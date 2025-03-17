"""
DOCX Format-Preserving Translator (Tối Ưu Hóa)
- Xử lý trực tiếp file DOCX như một tệp ZIP
- Sử dụng đa luồng để tăng tốc độ xử lý
- Sử dụng bộ nhớ đệm để tránh dịch lại nội dung trùng lặp
"""

import os
import zipfile
import tempfile
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import concurrent.futures
from typing import List, Dict, Tuple, Callable, Optional, Any
import hashlib
from tqdm import tqdm

# Namespace cho DOCX XML
NAMESPACES = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'wps': 'http://schemas.microsoft.com/office/word/2010/wordprocessingShape',
}

# Đăng ký tất cả namespaces
for prefix, uri in NAMESPACES.items():
    ET.register_namespace(prefix, uri)


class TranslationCache:
    """
    Bộ nhớ đệm để lưu trữ kết quả dịch, tránh dịch lại các đoạn văn bản giống nhau.
    Sử dụng cơ chế LRU đơn giản để quản lý kích thước bộ nhớ.
    """
    def __init__(self, max_size: int = 2000):
        self.cache: Dict[str, str] = {}
        self.max_size = max_size
        self.access_order: List[str] = []  # Để theo dõi thứ tự truy cập cho LRU
        self.hits = 0
        self.misses = 0
    
    def get(self, text: str) -> Optional[str]:
        """Lấy bản dịch từ cache, cập nhật thống kê và thứ tự truy cập"""
        key = self._get_key(text)
        
        if key in self.cache:
            # Cập nhật thứ tự truy cập (đưa key này lên cuối danh sách)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, text: str, translated_text: str) -> None:
        """Lưu bản dịch vào cache, thực hiện LRU nếu cần"""
        key = self._get_key(text)
        
        # Kiểm tra nếu cache đầy, xóa các mục ít được truy cập nhất
        if len(self.cache) >= self.max_size:
            # Xóa 20% các mục ít được truy cập nhất
            items_to_remove = int(self.max_size * 0.2)
            for _ in range(items_to_remove):
                if self.access_order:
                    oldest_key = self.access_order.pop(0)  # Lấy phần tử đầu tiên (ít được truy cập nhất)
                    if oldest_key in self.cache:
                        del self.cache[oldest_key]
        
        self.cache[key] = translated_text
        
        # Cập nhật thứ tự truy cập
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def _get_key(self, text: str) -> str:
        """Tạo key cho cache từ text bằng hàm băm MD5"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Trả về thống kê về hiệu quả của cache"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
        }


class DocxTranslator:
    def __init__(self, translate_func: Callable[[str], str] = None, max_workers: int = 4):
        """
        Khởi tạo DocxTranslator với hàm dịch tùy chỉnh và cấu hình đa luồng
        
        Tham số:
            translate_func (callable): Hàm dịch văn bản
            max_workers (int): Số lượng worker tối đa cho đa luồng
        """
        self.translate_func = translate_func
        self.max_workers = max_workers
        self.cache = TranslationCache()
    
    def translate_text(self, text: str) -> str:
        """
        Dịch văn bản sử dụng cache và hàm dịch được cung cấp
        """
        if not text or not text.strip():
            return text
        
        # Kiểm tra trong cache trước
        cached_translation = self.cache.get(text)
        if cached_translation:
            return cached_translation
        
        # Nếu không có trong cache, dịch bằng hàm translate_func
        if self.translate_func:
            translated = self.translate_func(text)
            # Lưu vào cache
            self.cache.set(text, translated)
            return translated
        
        # Nếu không có hàm dịch, trả về văn bản gốc
        return text
    
    def _translate_text_elements(self, elements: List[ET.Element]) -> int:
        """
        Dịch danh sách các phần tử văn bản XML
        
        Returns:
            int: Số lượng phần tử đã dịch
        """
        count = 0
        
        for elem in elements:
            if elem.text and elem.text.strip():
                # Dịch nội dung (sử dụng cache)
                elem.text = self.translate_text(elem.text)
                count += 1
        
        return count
    
    def _extract_xml_texts(self, xml_file: str) -> Tuple[List[ET.Element], ET.ElementTree]:
        """
        Trích xuất tất cả các phần tử văn bản từ file XML
        
        Returns:
            Tuple[List[ET.Element], ET.ElementTree]: 
            Danh sách các phần tử, và cây XML
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        elements = []
        
        for elem in root.findall('.//w:t', NAMESPACES):
            if elem.text and elem.text.strip():
                elements.append(elem)
        
        return elements, tree
    
    def _translate_xml_file(self, xml_file: str) -> Tuple[int, ET.ElementTree]:
        """
        Dịch nội dung văn bản trong file XML
        
        Returns:
            Tuple[int, ET.ElementTree]: Số lượng đoạn văn bản đã dịch và cây XML đã cập nhật
        """
        elements, tree = self._extract_xml_texts(xml_file)
        
        if not elements:
            return 0, tree
        
        # Dịch tất cả các phần tử văn bản
        count = self._translate_text_elements(elements)
        
        return count, tree
    
    def _translate_xml_chunk(self, elements_chunk: List[ET.Element]) -> int:
        """
        Dịch một nhóm các phần tử văn bản - sử dụng cho đa luồng
        
        Returns:
            int: Số lượng phần tử đã dịch
        """
        return self._translate_text_elements(elements_chunk)
    
    def _process_xml_file_with_threads(self, xml_file: str) -> Tuple[int, ET.ElementTree]:
        """
        Xử lý một file XML sử dụng đa luồng để dịch nội dung
        
        Returns:
            Tuple[int, ET.ElementTree]: Số lượng đoạn văn bản đã dịch và cây XML đã cập nhật
        """
        elements, tree = self._extract_xml_texts(xml_file)
        
        if not elements:
            return 0, tree
        
        total_elements = len(elements)
        
        # Nếu số lượng phần tử ít, không cần đa luồng
        if total_elements <= 10:
            count = self._translate_text_elements(elements)
            return count, tree
        
        # Chia nhỏ danh sách để xử lý đa luồng
        # Xác định kích thước chunk phù hợp
        chunk_size = max(5, total_elements // (self.max_workers * 2))
        chunks = [elements[i:i + chunk_size] for i in range(0, total_elements, chunk_size)]
        
        count = 0
        
        # Sử dụng ThreadPoolExecutor để xử lý song song
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Gửi các tác vụ dịch chunk
            futures = [executor.submit(self._translate_text_elements, chunk) for chunk in chunks]
            
            # Thu thập kết quả
            for future in concurrent.futures.as_completed(futures):
                try:
                    count += future.result()
                except Exception as e:
                    print(f"Lỗi khi xử lý chunk: {e}")
        
        return count, tree
    
    def _process_xml_files_parallel(self, xml_files: List[str]) -> Dict[str, Tuple[int, ET.ElementTree]]:
        """
        Xử lý song song nhiều file XML sử dụng ThreadPoolExecutor
        
        Returns:
            Dict[str, Tuple[int, ET.ElementTree]]: Từ điển ánh xạ đường dẫn file -> kết quả xử lý
        """
        results = {}
        
        # Tạo thanh tiến trình
        pbar = tqdm(total=len(xml_files), desc="Xử lý các file XML")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Gửi các tác vụ xử lý file XML
            future_to_file = {
                executor.submit(self._process_xml_file_with_threads, xml_file): xml_file
                for xml_file in xml_files
            }
            
            # Xử lý kết quả khi hoàn thành
            for future in concurrent.futures.as_completed(future_to_file):
                xml_file = future_to_file[future]
                try:
                    count, tree = future.result()
                    results[xml_file] = (count, tree)
                except Exception as e:
                    print(f"Lỗi khi xử lý file {xml_file}: {e}")
                    # Trong trường hợp lỗi, trả về cây XML gốc
                    try:
                        tree = ET.parse(xml_file)
                        results[xml_file] = (0, tree)
                    except Exception:
                        # Nếu không thể đọc file, bỏ qua
                        pass
                finally:
                    pbar.update(1)
        
        pbar.close()
        return results
    
    def translate_docx_complete(self, input_path, output_path):
        """
        Dịch DOCX với tối ưu hóa hiệu suất:
        - Xử lý song song nhiều file XML và phần tử trong mỗi file
        - Sử dụng cache để tránh dịch lại các đoạn trùng lặp
        """
        print(f"Đang xử lý đầy đủ file DOCX: {input_path}")
        start_time = time.time()
        
        # Tạo thư mục tạm để giải nén file DOCX
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Giải nén file DOCX vào thư mục tạm
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Thu thập tất cả các file XML cần xử lý
            xml_files = []
            
            # Thêm document.xml (file chính)
            document_xml_path = os.path.join(temp_dir, 'word', 'document.xml')
            if os.path.exists(document_xml_path):
                xml_files.append(document_xml_path)
            
            # Thêm các file header và footer
            word_dir = os.path.join(temp_dir, 'word')
            if os.path.exists(word_dir):
                for filename in os.listdir(word_dir):
                    if filename.startswith('header') or filename.startswith('footer'):
                        xml_files.append(os.path.join(word_dir, filename))
            
            # Thêm các file XML khác có thể chứa văn bản (footnotes, endnotes, etc.)
            for dirpath, _, filenames in os.walk(word_dir):
                for filename in filenames:
                    if filename.endswith('.xml') and filename not in ['document.xml', 'styles.xml', 
                                                                     'settings.xml', 'fontTable.xml', 
                                                                     'webSettings.xml', 'theme1.xml']:
                        xml_path = os.path.join(dirpath, filename)
                        if xml_path not in xml_files:  # Tránh trùng lặp
                            xml_files.append(xml_path)
            
            print(f"Tìm thấy {len(xml_files)} file XML cần xử lý")
            
            # Xử lý song song tất cả các file XML
            xml_results = self._process_xml_files_parallel(xml_files)
            
            # Tính tổng số đoạn văn bản đã xử lý
            total_texts = sum(count for count, _ in xml_results.values())
            print(f"Đã dịch {total_texts} đoạn văn bản")
            
            # Hiển thị thống kê về cache
            cache_stats = self.cache.get_stats()
            print(f"Hiệu quả cache: {cache_stats['hit_rate']:.2f}% hit rate ({cache_stats['hits']} hits, {cache_stats['misses']} misses)")
            
            # Lưu tất cả các file XML đã cập nhật
            for xml_file, (_, tree) in xml_results.items():
                tree.write(xml_file, encoding='utf-8', xml_declaration=True)
            
            # Tạo file DOCX mới từ thư mục đã cập nhật
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
                for root_dir, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root_dir, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zip_out.write(file_path, arcname)
            
            elapsed_time = time.time() - start_time
            print(f"Đã lưu file DOCX với định dạng được giữ nguyên hoàn toàn: {output_path}")
            print(f"Thời gian xử lý: {elapsed_time:.2f} giây")
            
            return output_path
            
        except Exception as e:
            print(f"Lỗi khi xử lý file DOCX: {e}")
            raise
            
        finally:
            # Dọn dẹp thư mục tạm
            shutil.rmtree(temp_dir)