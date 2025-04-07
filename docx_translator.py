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
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import concurrent.futures
from typing import List, Dict, Tuple, Callable, Optional, Any, Set
import hashlib
from tqdm import tqdm
import re

# Mở rộng NAMESPACES để hỗ trợ thêm các namespaces mới
NAMESPACES = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'wps': 'http://schemas.microsoft.com/office/word/2010/wordprocessingShape',
    'mc': 'http://schemas.openxmlformats.org/markup-compatibility/2006',
    'v': 'urn:schemas-microsoft-com:vml',
    'o': 'urn:schemas-microsoft-com:office:office',
    'w14': 'http://schemas.microsoft.com/office/word/2010/wordml',
    'w15': 'http://schemas.microsoft.com/office/word/2012/wordml',
}

# Đăng ký tất cả namespaces
for prefix, uri in NAMESPACES.items():
    ET.register_namespace(prefix, uri)

# Thêm trường hợp đặc biệt cần bảo vệ
SPECIAL_TEXT_PATTERNS = [
    re.compile(r'\d+(\.\d+)?'),  # Số (thập phân hoặc nguyên)
    re.compile(r'^[ivxlcdm]+$', re.IGNORECASE),  # Số La Mã
    re.compile(r'^[A-Z]\d+$'),  # Mã như A1, B2, C3
    re.compile(r'^\d{4}-\d{2}-\d{2}$'),  # Định dạng ngày tháng ISO
    re.compile(r'^(0?[1-9]|[12][0-9]|3[01])/(0?[1-9]|1[0-2])/\d{4}$'),  # dd/mm/yyyy
    re.compile(r'^([A-Z][a-z]*\s?)+$'),  # Tên riêng
]

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
        self.document_analysis = None  # Kết quả phân tích cấu trúc tài liệu
    
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
    
    def _is_roman_numeral(self, text: str) -> bool:
        """
        Kiểm tra xem một chuỗi có phải là số La Mã hay không
        
        Args:
            text: Chuỗi cần kiểm tra
            
        Returns:
            bool: True nếu là số La Mã, False nếu không phải
        """
        # Làm sạch chuỗi trước khi kiểm tra
        text = text.strip().lower()
        
        # Nếu chuỗi trống thì không phải số La Mã
        if not text:
            return False
        
        # Kiểm tra nếu chuỗi chỉ chứa các ký tự của số La Mã
        valid_chars = {'i', 'v', 'x', 'l', 'c', 'd', 'm'}
        if not all(c in valid_chars for c in text):
            return False
        
        # Kiểm tra chuỗi theo mẫu thông thường của số La Mã
        # Xử lý các trường hợp đặc biệt như i, ii, iii, iv, v, vi, vii, viii, ix, x...
        roman_pattern = re.compile(r'^(?=[ivxlcdm])m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$')
        return bool(roman_pattern.match(text))
    
    def _group_text_elements(self, text_elements: List[ET.Element], paragraph_elem: ET.Element) -> List[ET.Element]:
        """
        Gộp các phần tử văn bản trong cùng một đoạn để dịch theo ngữ cảnh
        Bỏ qua các phần tử là số khi gộp để tránh sự nhầm lẫn
        
        Args:
            text_elements: Danh sách các phần tử w:t trong một đoạn
            paragraph_elem: Phần tử w:p chứa các phần tử văn bản
            
        Returns:
            List[ET.Element]: Danh sách phần tử đã được xử lý
        """
        # Nếu không có phần tử nào có nội dung, trả về danh sách rỗng
        if not any(elem.text and elem.text.strip() for elem in text_elements):
            return []
        
        # Hàm kiểm tra xem một chuỗi có phải là số hay không
        def is_number(text):
            # Loại bỏ khoảng trắng trước và sau
            text = text.strip()
            # Kiểm tra xem có phải là một số (nguyên hoặc thập phân)
            try:
                # Thử chuyển thành số
                float(text.replace(',', '.'))
                return True
            except ValueError:
                # Kiểm tra thêm các trường hợp đặc biệt như năm (1999, 2023...)
                if text.isdigit() and len(text) >= 4:
                    return True
                return False
        
        # Hàm kiểm tra xem một chuỗi có phải là số La Mã hay không
        def is_roman_numeral(text):
            return self._is_roman_numeral(text)
            
        # Danh sách để lưu trữ phần tử kết quả
        result_elements = []
        
        # Phân loại các phần tử
        numeric_elements = []  # Phần tử chứa số
        roman_numeral_elements = []  # Phần tử chứa số La Mã
        text_content_elements = []  # Phần tử chứa văn bản thông thường
        
        # Phân loại các phần tử
        for elem in text_elements:
            if elem.text and elem.text.strip():
                if is_number(elem.text):
                    numeric_elements.append(elem)
                elif is_roman_numeral(elem.text):
                    roman_numeral_elements.append(elem)
                    print(f"Phát hiện số La Mã: {elem.text}")
                else:
                    text_content_elements.append(elem)
        
        # Xử lý các phần tử văn bản thông thường (gộp lại)
        if text_content_elements:
            # Gộp tất cả nội dung văn bản từ các phần tử không phải số
            full_text = ""
            for elem in text_content_elements:
                if elem.text:
                    full_text += elem.text
            
            # Nếu nội dung gộp không trống
            if full_text.strip():
                # Đặt tất cả văn bản vào phần tử đầu tiên
                first_elem = text_content_elements[0]
                first_elem.text = full_text
                
                # Đánh dấu các phần tử khác để bỏ qua khi dịch
                for elem in text_content_elements[1:]:
                    elem.text = ""
                
                # Thêm phần tử đầu tiên vào kết quả
                result_elements.append(first_elem)
        
        # Thêm các phần tử số vào kết quả (giữ nguyên)
        result_elements.extend(numeric_elements)
        
        # Thêm các phần tử số La Mã vào kết quả (giữ nguyên)
        result_elements.extend(roman_numeral_elements)
        
        return result_elements
    
    def _translate_text_elements(self, elements: List[ET.Element]) -> int:
        """
        Dịch danh sách các phần tử văn bản XML
        
        Returns:
            int: Số lượng phần tử đã dịch
        """
        count = 0
        
        # Hàm kiểm tra xem một chuỗi có phải là số hay không
        def is_number(text):
            text = text.strip()
            try:
                float(text.replace(',', '.'))
                return True
            except ValueError:
                if text.isdigit() and len(text) >= 4:
                    return True
                return False
        
        # Hàm kiểm tra xem một chuỗi có phải là số La Mã hay không
        def is_roman_numeral(text):
            return self._is_roman_numeral(text)
        
        # Phát hiện và xử lý các phần tử đặc biệt
        for elem in elements:
            if elem.text and elem.text.strip():
                # Nếu là số hoặc số La Mã, bỏ qua không dịch
                if is_number(elem.text):
                    print(f"Bỏ qua dịch phần tử số: {elem.text}")
                    continue
                elif is_roman_numeral(elem.text):
                    print(f"Bỏ qua dịch phần tử số La Mã: {elem.text}")
                    continue
                    
                # Xử lý đặc biệt cho văn bản quá ngắn
                if len(elem.text.strip()) <= 2:
                    # Bỏ qua dịch các phần tử chỉ có 1-2 ký tự nếu không phải số hoặc viết tắt
                    # Điều này giúp tránh việc dịch các ký tự riêng lẻ mà không có ngữ cảnh
                    if not elem.text.strip().isdigit() and '.' not in elem.text.strip():
                        continue
                
                # Dịch nội dung (sử dụng cache)
                orig_text = elem.text
                elem.text = self.translate_text(elem.text)
                
                # Phát hiện các thông báo lỗi từ API và khôi phục văn bản gốc
                error_indicators = [
                    "I'm sorry", "Sorry,", "but the text", "Please provide", 
                    "does not contain", "appears to be incomplete", 
                    "not in a recognizable", "not translatable"
                ]
                
                if any(indicator in elem.text for indicator in error_indicators):
                    # Khôi phục văn bản gốc nếu API trả về thông báo lỗi
                    elem.text = orig_text
                else:
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
        
        # Phiên bản cũ: lấy từng phần tử w:t riêng lẻ
        # for elem in root.findall('.//w:t', NAMESPACES):
        #     if elem.text and elem.text.strip():
        #         elements.append(elem)
        
        # Phiên bản cải tiến: gộp các phần tử w:t trong cùng một đoạn w:p
        paragraphs = root.findall('.//w:p', NAMESPACES)
        for paragraph in paragraphs:
            text_elements = paragraph.findall('.//w:t', NAMESPACES)
            if text_elements:
                # Thêm các phần tử có văn bản vào danh sách để xử lý theo ngữ cảnh
                elements.extend(self._group_text_elements(text_elements, paragraph))
        
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
        
        # Ưu tiên sử dụng phương pháp dịch theo đoạn văn
        # Phương pháp này áp dụng cho cả tài liệu thông thường và đặc biệt
        count = self._translate_by_paragraphs(tree, xml_file)
        if count > 0:
            return count, tree
        
        # Nếu không thể dịch theo đoạn văn, áp dụng phương pháp dịch theo phần tử
        total_elements = len(elements)
        
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
    
    def _translate_paragraph(self, paragraph_data: dict) -> dict:
        """
        Dịch một đoạn văn bản - được thiết kế cho xử lý đa luồng
        
        Args:
            paragraph_data: Dictionary chứa thông tin về đoạn văn cần dịch
            
        Returns:
            dict: Kết quả dịch thuật
        """
        paragraph = paragraph_data['paragraph']
        paragraph_text = paragraph_data['text']
        text_elements = paragraph_data['elements']
        
        # Hàm kiểm tra xem một chuỗi có phải là số hay không
        def is_number(text):
            text = text.strip()
            try:
                float(text.replace(',', '.'))
                return True
            except ValueError:
                if text.isdigit() and len(text) >= 4:
                    return True
                return False
        
        # Hàm kiểm tra xem một chuỗi có phải là số La Mã hay không
        def is_roman_numeral(text):
            return self._is_roman_numeral(text)
        
        # Phân loại các phần tử trong đoạn văn
        numeric_elements = []  # Phần tử chứa số
        roman_numeral_elements = []  # Phần tử chứa số La Mã
        text_content_elements = []  # Phần tử chứa văn bản thông thường
        
        # Phân loại các phần tử và nội dung văn bản của chúng
        for elem in text_elements:
            if elem.text and elem.text.strip():
                if is_number(elem.text):
                    numeric_elements.append(elem)
                elif is_roman_numeral(elem.text):
                    roman_numeral_elements.append(elem)
                else:
                    text_content_elements.append(elem)
        
        # Gộp văn bản của các phần tử không phải số hoặc số La Mã
        non_special_text = ""
        for elem in text_content_elements:
            if elem.text:
                non_special_text += elem.text
        
        # Chỉ dịch nếu đoạn có nội dung thông thường có ý nghĩa
        if not non_special_text.strip() or len(non_special_text.strip()) <= 2:
            return {'success': False, 'count': 0}
        
        # Dịch đoạn văn bản thông thường đã gộp
        translated_text = self.translate_text(non_special_text)
        
        # Phát hiện các thông báo lỗi từ API
        error_indicators = [
            "I'm sorry", "Sorry,", "but the text", "Please provide", 
            "does not contain", "appears to be incomplete", 
            "not in a recognizable", "not translatable"
        ]
        
        if any(indicator in translated_text for indicator in error_indicators):
            # Nếu có lỗi, giữ nguyên văn bản gốc
            return {'success': False, 'count': 0}
        
        # Áp dụng kết quả dịch vào tài liệu
        if text_content_elements:
            # Chiến lược thông minh để phân phối văn bản dịch
            if len(text_content_elements) == 1:
                # Nếu chỉ có một phần tử, đặt toàn bộ văn bản dịch vào đó
                text_content_elements[0].text = translated_text
            else:
                # Cố gắng phân phối văn bản dịch theo tỷ lệ của văn bản gốc
                # Để giữ định dạng tốt hơn
                try:
                    # Tính tổng độ dài văn bản gốc thông thường
                    total_original_length = sum(len(elem.text) if elem.text else 0 for elem in text_content_elements)
                    
                    if total_original_length > 0 and len(translated_text) > 0:
                        # Phát hiện phân đoạn và ký tự đặc biệt trong bản dịch
                        # (như dấu xuống dòng, dấu chấm, v.v.)
                        # Để quyết định cách phân phối lại văn bản
                        
                        # Nếu có các ký tự xuống dòng, sử dụng chúng để phân đoạn
                        if '\n' in translated_text and '\n' in non_special_text:
                            # Phân đoạn theo dòng
                            original_segments = non_special_text.split('\n')
                            translated_segments = translated_text.split('\n')
                            
                            # Đảm bảo có cùng số lượng phân đoạn
                            min_segments = min(len(original_segments), len(translated_segments))
                            
                            # Phân phối cho từng phần tử với số lượng dòng tương ứng
                            current_elem_index = 0
                            current_segment_index = 0
                            
                            while current_elem_index < len(text_content_elements) and current_segment_index < min_segments:
                                # Đặt phân đoạn dịch vào phần tử văn bản hiện tại
                                text_content_elements[current_elem_index].text = translated_segments[current_segment_index]
                                
                                # Di chuyển đến phần tử và phân đoạn tiếp theo
                                current_elem_index += 1
                                current_segment_index += 1
                            
                            # Nếu còn phân đoạn dịch nhưng hết phần tử
                            if current_segment_index < len(translated_segments) and current_elem_index > 0:
                                # Thêm vào phần tử cuối cùng
                                remaining_text = '\n'.join(translated_segments[current_segment_index:])
                                if text_content_elements[current_elem_index-1].text:
                                    text_content_elements[current_elem_index-1].text += '\n' + remaining_text
                                else:
                                    text_content_elements[current_elem_index-1].text = remaining_text
                        else:
                            # Phương pháp đơn giản: đặt toàn bộ vào phần tử đầu tiên
                            text_content_elements[0].text = translated_text
                            
                            # Xóa nội dung các phần tử khác
                            for elem in text_content_elements[1:]:
                                elem.text = ""
                except Exception as e:
                    # Nếu có lỗi, sử dụng phương pháp đơn giản
                    text_content_elements[0].text = translated_text
                    for elem in text_content_elements[1:]:
                        elem.text = ""
            
            # Các phần tử số và số La Mã được giữ nguyên, không cần thay đổi
            return {'success': True, 'count': 1}
        
        return {'success': False, 'count': 0}
    
    def _translate_by_paragraphs(self, tree: ET.ElementTree, xml_file: str) -> int:
        """
        Dịch tài liệu theo đơn vị đoạn văn thay vì từng phần tử riêng lẻ,
        áp dụng đa luồng để xử lý song song các đoạn văn
        
        Args:
            tree: Cây XML của tài liệu
            xml_file: Đường dẫn tới file XML
            
        Returns:
            int: Số lượng đoạn văn bản đã dịch
        """
        count = 0
        root = tree.getroot()
        
        # Tìm tất cả các đoạn văn
        paragraphs = root.findall('.//w:p', NAMESPACES)
        print(f"Tìm thấy {len(paragraphs)} đoạn văn trong file {os.path.basename(xml_file)}")
        
        # Chuẩn bị dữ liệu cho xử lý đa luồng
        paragraph_data_list = []
        
        for paragraph in paragraphs:
            # Bỏ qua các đoạn văn rỗng
            text_elements = paragraph.findall('.//w:t', NAMESPACES)
            if not text_elements:
                continue
            
            # Gộp văn bản của tất cả phần tử trong đoạn
            paragraph_text = ""
            effective_text_elements = []
            
            for elem in text_elements:
                if elem.text:
                    paragraph_text += elem.text
                    effective_text_elements.append(elem)
            
            if paragraph_text.strip():
                paragraph_data_list.append({
                    'paragraph': paragraph,
                    'text': paragraph_text,
                    'elements': effective_text_elements
                })
        
        # Nếu không có đoạn văn bản hợp lệ
        if not paragraph_data_list:
            print(f"Không tìm thấy đoạn văn hợp lệ trong file {os.path.basename(xml_file)}")
            # Thử tìm và xử lý văn bản trong file theo cách khác nếu là file document.xml
            if "document.xml" in xml_file:
                return self._process_document_xml_direct(tree, xml_file)
            return 0
        
        print(f"Dịch {len(paragraph_data_list)} đoạn văn với đa luồng")
        
        # In thông tin về 3 đoạn văn đầu tiên để debug
        for i, data in enumerate(paragraph_data_list[:3]):
            print(f"Đoạn {i+1}: '{data['text'][:50]}...' ({len(data['elements'])} phần tử)")
        
        # Áp dụng đa luồng để dịch song song các đoạn văn
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Gửi các tác vụ dịch
            futures = [executor.submit(self._translate_paragraph, data) for data in paragraph_data_list]
            
            # Theo dõi tiến độ với tqdm nếu có nhiều đoạn
            if len(futures) > 10:
                pbar = tqdm(total=len(futures), desc="Dịch các đoạn văn")
            else:
                pbar = None
            
            # Thu thập kết quả
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result['success']:
                        count += result['count']
                    if pbar:
                        pbar.update(1)
                except Exception as e:
                    print(f"Lỗi khi dịch đoạn văn: {e}")
                    if pbar:
                        pbar.update(1)
            
            if pbar:
                pbar.close()
        
        print(f"Đã dịch thành công {count}/{len(paragraph_data_list)} đoạn văn")
        return count
    
    def _process_document_xml_direct(self, tree: ET.ElementTree, xml_file: str) -> int:
        """
        Xử lý trực tiếp file document.xml khi không tìm thấy đoạn văn hợp lệ
        
        Args:
            tree: Cây XML của tài liệu
            xml_file: Đường dẫn tới file document.xml
            
        Returns:
            int: Số lượng phần tử đã dịch
        """
        print("Áp dụng phương pháp trực tiếp cho file document.xml")
        root = tree.getroot()
        count = 0
        
        # Tìm tất cả các phần tử văn bản
        all_text_elements = root.findall('.//w:t', NAMESPACES)
        text_elements_with_content = [elem for elem in all_text_elements if elem.text and elem.text.strip()]
        
        print(f"Tìm thấy {len(text_elements_with_content)}/{len(all_text_elements)} phần tử văn bản có nội dung")
        
        # In một số phần tử văn bản để debug
        for i, elem in enumerate(text_elements_with_content[:10]):
            print(f"Phần tử {i+1}: '{elem.text[:30]}...'")
        
        # Tìm các phần tử có nội dung trung bình hoặc dài
        longer_text_elements = [elem for elem in text_elements_with_content if len(elem.text.strip()) > 10]
        
        if longer_text_elements:
            print(f"Dịch {len(longer_text_elements)} phần tử văn bản dài")
            
            # Áp dụng đa luồng để dịch song song
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for elem in longer_text_elements:
                    # Tạo hàm dịch phần tử
                    def translate_element(element):
                        if element.text and element.text.strip():
                            original_text = element.text
                            element.text = self.translate_text(original_text)
                            
                            # Kiểm tra thông báo lỗi
                            error_indicators = [
                                "I'm sorry", "Sorry,", "but the text", "Please provide", 
                                "does not contain", "appears to be incomplete", 
                                "not in a recognizable", "not translatable"
                            ]
                            
                            if any(indicator in element.text for indicator in error_indicators):
                                element.text = original_text
                                return False
                            return True
                        return False
                    
                    # Gửi tác vụ dịch
                    futures.append(executor.submit(translate_element, elem))
                
                # Thu thập kết quả
                for future in concurrent.futures.as_completed(futures):
                    try:
                        if future.result():
                            count += 1
                    except Exception as e:
                        print(f"Lỗi khi dịch phần tử: {e}")
        
        print(f"Đã dịch thành công {count} phần tử văn bản")
        return count
    
    def _process_xml_files_parallel(self, xml_files: List[str], progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, Tuple[int, ET.ElementTree]]:
        """
        Xử lý song song nhiều file XML sử dụng ThreadPoolExecutor
        
        Args:
            xml_files: Danh sách các file XML cần xử lý
            progress_callback: Hàm callback để báo cáo tiến trình
            
        Returns:
            Dict[str, Tuple[int, ET.ElementTree]]: Từ điển ánh xạ đường dẫn file -> kết quả xử lý
        """
        results = {}
        total_files = len(xml_files)
        processed_files = 0
        
        # Tạo thanh tiến trình
        pbar = tqdm(total=total_files, desc="Xử lý các file XML")
        
        # Xử lý tuần tự để dễ theo dõi và gỡ lỗi cho nhiều file XML
        print(f"Bắt đầu xử lý {len(xml_files)} file XML...")
        for xml_file in xml_files:
            try:
                print(f"\nĐang xử lý file: {os.path.basename(xml_file)}")
                # Đọc nội dung file XML
                tree = ET.parse(xml_file)
                
                # Kiểm tra loại file XML
                is_document = "document.xml" in xml_file
                is_header = "header" in os.path.basename(xml_file)
                is_footer = "footer" in os.path.basename(xml_file)
                file_type = "document" if is_document else "header" if is_header else "footer" if is_footer else "other"
                print(f"Loại file XML: {file_type}")
                
                # Dịch nội dung
                count = self._translate_by_paragraphs(tree, xml_file)
                
                if count == 0 and is_document:
                    # Thử phương pháp thay thế cho document.xml
                    print(f"Không tìm thấy đoạn văn để dịch trong {os.path.basename(xml_file)}, thử phương pháp khác...")
                    count = self._process_document_xml_direct(tree, xml_file)
                
                # Lưu kết quả ngay lập tức để đảm bảo không bị mất
                if count > 0:
                    print(f"Đã dịch {count} đoạn trong {os.path.basename(xml_file)}")
                    tree.write(xml_file, encoding='utf-8', xml_declaration=True)
                else:
                    print(f"Không dịch được đoạn nào trong {os.path.basename(xml_file)}")
                
                # Lưu kết quả vào dict
                results[xml_file] = (count, tree)
                
                pbar.update(1)
                processed_files += 1
                
                if progress_callback:
                    progress_callback(processed_files, total_files)
                    
            except Exception as e:
                print(f"Lỗi khi xử lý file {xml_file}: {e}")
                try:
                    tree = ET.parse(xml_file)
                    results[xml_file] = (0, tree)
                except Exception:
                    pass
                
                pbar.update(1)
                processed_files += 1
                
                if progress_callback:
                    progress_callback(processed_files, total_files)
        
        pbar.close()
        
        # Hiển thị thống kê
        total_translated = sum(count for count, _ in results.values())
        print(f"Tổng cộng đã dịch {total_translated} đoạn từ {len(xml_files)} file XML")
        
        return results

    def _should_preserve_text(self, text: str) -> bool:
        """
        Kiểm tra xem văn bản có nên được giữ nguyên không dịch
        
        Args:
            text: Văn bản cần kiểm tra
            
        Returns:
            bool: True nếu nên giữ nguyên, False nếu nên dịch
        """
        if not text or not text.strip():
            return True
            
        text = text.strip()
        
        # Kiểm tra các mẫu đặc biệt
        for pattern in SPECIAL_TEXT_PATTERNS:
            if pattern.match(text):
                return True
                
        # Kiểm tra nếu văn bản quá ngắn
        if len(text) <= 2 and not text.isalpha():
            return True
            
        return False
        
    def _preserve_xml_structure(self, element: ET.Element) -> Dict[str, str]:
        """
        Bảo toàn cấu trúc XML của phần tử để khôi phục sau khi dịch
        
        Args:
            element: Phần tử XML cần bảo toàn
            
        Returns:
            Dict: Thông tin về cấu trúc XML
        """
        preserved = {
            'tag': element.tag,
            'attrib': {k: v for k, v in element.attrib.items()},
            'tail': element.tail,
        }
        return preserved
        
    def _restore_xml_structure(self, element: ET.Element, preserved: Dict[str, str]) -> None:
        """
        Khôi phục cấu trúc XML của phần tử sau khi dịch
        
        Args:
            element: Phần tử XML cần khôi phục
            preserved: Thông tin về cấu trúc đã bảo toàn
        """
        for k, v in preserved['attrib'].items():
            element.attrib[k] = v
        element.tail = preserved['tail']
    
    def _process_docx_metadata(self, temp_dir: str) -> None:
        """
        Xử lý metadata của file DOCX để đảm bảo tính nhất quán
        
        Args:
            temp_dir: Đường dẫn tới thư mục tạm chứa nội dung DOCX giải nén
        """
        # Xử lý core.xml để cập nhật thông tin về bản dịch
        core_path = os.path.join(temp_dir, 'docProps', 'core.xml')
        if os.path.exists(core_path):
            try:
                tree = ET.parse(core_path)
                root = tree.getroot()
                
                # Tìm namespace của Core properties
                cp_ns = '{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}'
                dc_ns = '{http://purl.org/dc/elements/1.1/}'
                
                # Cập nhật revision nếu có
                revision_elem = root.find(f'.//{cp_ns}revision')
                if revision_elem is not None:
                    try:
                        revision = int(revision_elem.text)
                        revision_elem.text = str(revision + 1)
                    except (ValueError, TypeError):
                        revision_elem.text = "1"
                
                # Cập nhật lastModifiedBy
                modified_by = root.find(f'.//{cp_ns}lastModifiedBy')
                if modified_by is not None:
                    modified_by.text = "DocxTranslator"
                
                # Cập nhật lastModified
                from datetime import datetime
                last_mod = root.find(f'.//{dc_ns}modified')
                if last_mod is not None:
                    last_mod.text = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                
                # Lưu lại file core.xml
                tree.write(core_path, encoding='utf-8', xml_declaration=True)
            except Exception as e:
                print(f"Lỗi khi xử lý metadata: {e}")
    
    def _handle_complex_tables(self, xml_file: str) -> None:
        """
        Xử lý đặc biệt cho các bảng phức tạp
        
        Args:
            xml_file: Đường dẫn tới file XML cần xử lý
        """
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Tìm tất cả các bảng lồng nhau
            nested_tables = []
            tables = root.findall('.//w:tbl', NAMESPACES)
            
            for table in tables:
                if table.find('.//w:tbl', NAMESPACES) is not None:
                    nested_tables.append(table)
            
            if nested_tables:
                print(f"Đã phát hiện {len(nested_tables)} bảng lồng nhau trong {os.path.basename(xml_file)}")
                
                # Đánh dấu các bảng lồng nhau để xử lý đặc biệt
                for i, table in enumerate(nested_tables):
                    # Thêm thuộc tính để đánh dấu là bảng lồng nhau
                    table.set('docx_translator_nested', f'table_{i}')
                
                # Lưu lại file sau khi đánh dấu
                tree.write(xml_file, encoding='utf-8', xml_declaration=True)
        except Exception as e:
            print(f"Lỗi khi xử lý bảng phức tạp: {e}")
    
    def _handle_special_elements(self, xml_file: str) -> None:
        """
        Xử lý các phần tử đặc biệt như dropcap, textbox, etc.
        
        Args:
            xml_file: Đường dẫn tới file XML cần xử lý
        """
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Xử lý dropcaps
            dropcaps = root.findall('.//*[@w:dropCap]', NAMESPACES)
            if dropcaps:
                print(f"Đã phát hiện {len(dropcaps)} dropcap trong {os.path.basename(xml_file)}")
                for dropcap in dropcaps:
                    # Đánh dấu dropcap để xử lý đặc biệt
                    dropcap.set('docx_translator_special', 'dropcap')
            
            # Xử lý textbox và các phần tử vml
            textboxes = root.findall('.//v:textbox', NAMESPACES)
            if textboxes:
                print(f"Đã phát hiện {len(textboxes)} textbox trong {os.path.basename(xml_file)}")
                for textbox in textboxes:
                    # Đánh dấu textbox để xử lý đặc biệt
                    container = textbox.getparent()
                    if container is not None:
                        container.set('docx_translator_special', 'textbox')
            
            # Lưu lại file sau khi đánh dấu
            tree.write(xml_file, encoding='utf-8', xml_declaration=True)
        except Exception as e:
            print(f"Lỗi khi xử lý phần tử đặc biệt: {e}")

    def translate_docx_complete(self, input_path, output_path, progress_callback: Optional[Callable[[int, int], None]] = None):
        """
        Dịch DOCX với tối ưu hóa hiệu suất và bảo toàn định dạng
        
        Args:
            input_path: Đường dẫn đến file DOCX cần dịch
            output_path: Đường dẫn lưu file DOCX đã dịch
            progress_callback: Hàm callback để báo cáo tiến trình
        """
        print(f"Đang xử lý đầy đủ file DOCX: {input_path}")
        start_time = time.time()
        
        # Tạo thư mục tạm để giải nén file DOCX
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Giải nén file DOCX vào thư mục tạm
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Tạo bản sao tạm trước khi xử lý
            backup_dir = tempfile.mkdtemp()
            # Sao chép toàn bộ thư mục temp_dir sang backup_dir
            for item in os.listdir(temp_dir):
                s = os.path.join(temp_dir, item)
                d = os.path.join(backup_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
            
            # Xử lý metadata của tài liệu
            self._process_docx_metadata(temp_dir)
            
            # Thu thập tất cả các file XML cần xử lý
            xml_files = []
            
            # Thêm document.xml (file chính)
            document_xml_path = os.path.join(temp_dir, 'word', 'document.xml')
            if os.path.exists(document_xml_path):
                xml_files.append(document_xml_path)
                
                # Xử lý các phần tử đặc biệt và bảng phức tạp
                self._handle_special_elements(document_xml_path)
                self._handle_complex_tables(document_xml_path)
                
                # Phân tích cấu trúc tài liệu để thông tin
                self.document_analysis = self._analyze_document_structure(document_xml_path)
                num_paragraphs = self.document_analysis.get('num_paragraphs', 0)
                avg_text_length = self.document_analysis.get('avg_text_length', 0.0)
                print(f"Phân tích cấu trúc tài liệu: {num_paragraphs} đoạn, độ dài trung bình: {avg_text_length:.2f} ký tự.")
            
            # Thêm các file XML khác cần dịch
            word_dir = os.path.join(temp_dir, 'word')
            if os.path.exists(word_dir):
                # Headers và footers
                for filename in os.listdir(word_dir):
                    if filename.startswith('header') or filename.startswith('footer'):
                        xml_path = os.path.join(word_dir, filename)
                        xml_files.append(xml_path)
                        self._handle_special_elements(xml_path)
                
                # Footnotes và endnotes
                for special_file in ['footnotes.xml', 'endnotes.xml']:
                    special_path = os.path.join(word_dir, special_file)
                    if os.path.exists(special_path):
                        xml_files.append(special_path)
                        self._handle_special_elements(special_path)
            
            # Xử lý các thuộc tính tài liệu trong docProps
            doc_props_dir = os.path.join(temp_dir, 'docProps')
            if os.path.exists(doc_props_dir):
                # Cập nhật thông tin về tài liệu dịch trong app.xml
                app_xml_path = os.path.join(doc_props_dir, 'app.xml')
                if os.path.exists(app_xml_path):
                    try:
                        tree = ET.parse(app_xml_path)
                        root = tree.getroot()
                        
                        # Tìm tất cả các phần tử văn bản chứa nội dung cần dịch
                        text_elements = []
                        for elem in root.iter():
                            if elem.text and elem.text.strip() and not self._should_preserve_text(elem.text):
                                text_elements.append(elem)
                        
                        # Dịch các phần tử văn bản tìm thấy
                        if text_elements:
                            print(f"Dịch {len(text_elements)} phần tử văn bản trong app.xml")
                            for elem in text_elements:
                                preserved = self._preserve_xml_structure(elem)
                                elem.text = self.translate_text(elem.text)
                                self._restore_xml_structure(elem, preserved)
                            
                            # Lưu lại file sau khi dịch
                            tree.write(app_xml_path, encoding='utf-8', xml_declaration=True)
                    except Exception as e:
                        print(f"Lỗi khi xử lý app.xml: {e}")
            
            print(f"Tìm thấy {len(xml_files)} file XML cần xử lý")
            
            # Kiểm tra nếu DOCX chỉ có 1 file XML (trường hợp đặc biệt)
            if len(xml_files) <= 1:
                print("Phát hiện file DOCX đơn giản (chỉ có 1 file XML). Áp dụng phương pháp bảo toàn định dạng đặc biệt...")
                
                # Báo cáo tiến trình bắt đầu nếu có callback
                if progress_callback:
                    progress_callback(0, 1)  # Chỉ có 1 file cần xử lý
                
                # Xử lý file document.xml bằng phương pháp đặc biệt
                total_texts = self._process_simple_docx(document_xml_path, backup_dir, temp_dir)
                
                # Báo cáo tiến trình hoàn tất
                if progress_callback:
                    progress_callback(1, 1)
            else:
                print(f"Phát hiện file DOCX phức tạp ({len(xml_files)} file XML). Áp dụng phương pháp xử lý đa file...")
                
                # Báo cáo tiến trình bắt đầu nếu có callback
                if progress_callback:
                    progress_callback(0, len(xml_files))
                
                # Xử lý song song tất cả các file XML
                xml_results = self._process_xml_files_parallel(xml_files, progress_callback)
                
                # Tính tổng số đoạn văn bản đã xử lý
                total_texts = sum(count for count, _ in xml_results.values())
                
                # Đảm bảo tất cả các file XML đã được lưu
                for xml_file, (count, tree) in xml_results.items():
                    if count > 0:  # Chỉ lưu lại các file đã được dịch (thay đổi)
                        print(f"Đang lưu file {os.path.basename(xml_file)} với {count} đoạn đã dịch")
                        tree.write(xml_file, encoding='utf-8', xml_declaration=True)
            
            print(f"Đã dịch tổng cộng {total_texts} đoạn văn bản")
            
            # Hiển thị thống kê về cache
            cache_stats = self.cache.get_stats()
            print(f"Hiệu quả cache: {cache_stats['hit_rate']:.2f}% hit rate ({cache_stats['hits']} hits, {cache_stats['misses']} misses)")
            
            # Tạo file DOCX mới từ thư mục đã cập nhật
            print(f"Đang tạo file DOCX mới: {output_path}")
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
                for root_dir, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root_dir, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zip_out.write(file_path, arcname)
                        
            # Xác nhận file đã được tạo thành công
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path)
                print(f"File DOCX đã được tạo thành công: {output_path} (kích thước: {output_size/1024:.2f} KB)")
            else:
                print(f"CẢNH BÁO: Không thể xác nhận file đầu ra {output_path}")
            
            elapsed_time = time.time() - start_time
            print(f"Đã lưu file DOCX với định dạng được giữ nguyên hoàn toàn: {output_path}")
            print(f"Thời gian xử lý: {elapsed_time:.2f} giây")
            
            # Báo cáo tiến trình hoàn thành nếu có callback
            if progress_callback:
                progress_callback(len(xml_files), len(xml_files))
            
            return output_path
            
        except Exception as e:
            print(f"Lỗi khi xử lý file DOCX: {e}")
            import traceback
            traceback.print_exc()  # In thông tin chi tiết về lỗi
            raise
            
        finally:
            # Dọn dẹp thư mục tạm
            shutil.rmtree(temp_dir)
            if 'backup_dir' in locals():
                shutil.rmtree(backup_dir)

    def _analyze_document_structure(self, document_xml_path: str) -> Dict[str, Any]:
        """
        Phân tích cấu trúc tài liệu để xác định chiến lược xử lý phù hợp
        
        Args:
            document_xml_path: Đường dẫn đến file document.xml
            
        Returns:
            Dict[str, Any]: Thông tin về cấu trúc tài liệu
        """
        try:
            tree = ET.parse(document_xml_path)
            root = tree.getroot()
            
            # Phân tích thống kê về các đoạn và phần tử văn bản
            paragraphs = root.findall('.//w:p', NAMESPACES)
            text_elements = root.findall('.//w:t', NAMESPACES)
            runs = root.findall('.//w:r', NAMESPACES)
            
            # Số lượng đoạn, phần tử văn bản, và runs
            num_paragraphs = len(paragraphs)
            num_text_elements = len(text_elements)
            num_runs = len(runs)
            
            # Thống kê về độ dài văn bản trong các phần tử
            text_lengths = [len(elem.text.strip()) if elem.text else 0 for elem in text_elements]
            avg_text_length = sum(text_lengths) / max(1, len(text_lengths))
            
            # Phát hiện các đoạn có cấu trúc đặc biệt
            special_paragraphs = 0
            for p in paragraphs:
                p_text_elements = p.findall('.//w:t', NAMESPACES)
                p_runs = p.findall('.//w:r', NAMESPACES)
                
                # Kiểm tra các đoạn văn có nhiều run nhưng mỗi run chứa ít ký tự
                if len(p_runs) >= 3:
                    char_per_run_count = 0
                    for run in p_runs[:5]:
                        t_elems = run.findall('.//w:t', NAMESPACES)
                        for t in t_elems:
                            if t.text and len(t.text.strip()) == 1:
                                char_per_run_count += 1
                    
                    if char_per_run_count >= 3:
                        special_paragraphs += 1
            
            # Phát hiện tệp cần phương pháp gộp văn bản
            needs_text_grouping = (
                special_paragraphs >= 3 or  # Nhiều đoạn đặc biệt
                (num_text_elements > 0 and avg_text_length < 5) or  # Văn bản ngắn
                (num_runs > 0 and num_text_elements / num_runs < 0.8)  # Nhiều run ít văn bản
            )
            
            return {
                "num_paragraphs": num_paragraphs,
                "num_text_elements": num_text_elements,
                "num_runs": num_runs,
                "avg_text_length": avg_text_length,
                "special_paragraphs": special_paragraphs,
                "needs_text_grouping": needs_text_grouping
            }
        except Exception as e:
            print(f"Lỗi khi phân tích cấu trúc tài liệu: {e}")
            return {
                "error": str(e),
                "needs_text_grouping": True  # Mặc định là true nếu có lỗi
            }

    def _process_simple_docx(self, document_xml_path, backup_dir, temp_dir):
        """
        Xử lý file DOCX đơn giản - chỉ có một file XML chính
        Áp dụng các kỹ thuật đặc biệt để giữ định dạng
        
        Args:
            document_xml_path: Đường dẫn đến file document.xml
            backup_dir: Thư mục chứa bản sao gốc
            temp_dir: Thư mục chứa file được xử lý
            
        Returns:
            int: Số lượng đoạn văn bản đã dịch
        """
        # Đọc file document.xml
        tree = ET.parse(document_xml_path)
        
        # Áp dụng phương pháp dịch theo đoạn văn
        count = self._translate_by_paragraphs(tree, document_xml_path)
        
        if count == 0:
            # Nếu không dịch được đoạn nào, thử phương pháp dịch cách khác
            count = self._process_document_xml_direct(tree, document_xml_path)
        
        # Lưu file document.xml đã cập nhật
        tree.write(document_xml_path, encoding='utf-8', xml_declaration=True)
        
        # Khôi phục các file định dạng từ bản sao
        backup_word_dir = os.path.join(backup_dir, 'word')
        temp_word_dir = os.path.join(temp_dir, 'word')
        
        if os.path.exists(backup_word_dir):
            for item in os.listdir(backup_word_dir):
                # Chỉ sao chép lại các file định dạng, không sao chép document.xml
                if item != 'document.xml' and item.endswith('.xml'):
                    src_path = os.path.join(backup_word_dir, item)
                    dest_path = os.path.join(temp_word_dir, item)
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dest_path)
        
        # Đảm bảo file [Content_Types].xml và .rels được giữ nguyên
        for root_item in os.listdir(backup_dir):
            if root_item in ['[Content_Types].xml', '_rels']:
                src_path = os.path.join(backup_dir, root_item)
                dest_path = os.path.join(temp_dir, root_item)
                if os.path.isdir(src_path):
                    # Xóa thư mục đích nếu tồn tại
                    if os.path.exists(dest_path):
                        shutil.rmtree(dest_path)
                    shutil.copytree(src_path, dest_path)
                else:
                    shutil.copy2(src_path, dest_path)
        
        return count