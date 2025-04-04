#!/usr/bin/env python3
"""
Script kiểm tra nội dung file DOCX
"""

import sys
import zipfile
import xml.etree.ElementTree as ET
import os
from pathlib import Path

# Namespace cho DOCX XML
NAMESPACES = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
}

def extract_paragraphs(docx_path, max_paragraphs=10, offset=0):
    """
    Trích xuất và in ra các đoạn văn trong file DOCX
    
    Args:
        docx_path: Đường dẫn đến file DOCX
        max_paragraphs: Số lượng đoạn văn tối đa cần in
        offset: Vị trí bắt đầu hiển thị đoạn văn
    """
    if not os.path.exists(docx_path):
        print(f"Lỗi: Không tìm thấy file {docx_path}")
        return
    
    print(f"Kiểm tra file: {docx_path}")
    
    try:
        # Mở file DOCX như một zip file
        with zipfile.ZipFile(docx_path, 'r') as zip_ref:
            # Kiểm tra xem document.xml có tồn tại không
            if 'word/document.xml' not in zip_ref.namelist():
                print("File DOCX không chứa word/document.xml")
                return
            
            # Đọc nội dung file document.xml
            with zip_ref.open('word/document.xml') as xml_file:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Tìm tất cả đoạn văn
                paragraphs = root.findall('.//w:p', NAMESPACES)
                
                print(f"Tổng số đoạn văn: {len(paragraphs)}")
                print(f"\nDanh sách đoạn văn (từ vị trí {offset}, tối đa {max_paragraphs} đoạn):")
                
                # Lọc ra các đoạn có nội dung
                valid_paragraphs = []
                for i, paragraph in enumerate(paragraphs):
                    # Bỏ qua các đoạn rỗng
                    text_elements = paragraph.findall('.//w:t', NAMESPACES)
                    if not text_elements:
                        continue
                    
                    # Gộp văn bản trong đoạn
                    paragraph_text = ""
                    for elem in text_elements:
                        if elem.text:
                            paragraph_text += elem.text
                    
                    # Chỉ đưa vào danh sách các đoạn không rỗng
                    if paragraph_text.strip():
                        valid_paragraphs.append((i+1, paragraph_text))
                
                # Hiển thị các đoạn từ offset
                count = 0
                for i in range(offset, min(offset + max_paragraphs, len(valid_paragraphs))):
                    para_num, text = valid_paragraphs[i]
                    print(f"Đoạn {para_num}: {text[:100]}{'...' if len(text) > 100 else ''}")
                    count += 1
                
                if offset + count < len(valid_paragraphs):
                    print(f"\n... và {len(valid_paragraphs) - (offset + count)} đoạn khác")
                
                # In một số thông tin khác trong file
                print("\nTên các file trong DOCX:")
                for name in zip_ref.namelist()[:10]:
                    print(f"- {name}")
                
                if len(zip_ref.namelist()) > 10:
                    print(f"... và {len(zip_ref.namelist()) - 10} file khác")
    
    except Exception as e:
        print(f"Lỗi khi đọc file DOCX: {e}")

def main():
    # Kiểm tra đầu vào
    if len(sys.argv) < 2:
        print("Cách sử dụng: python check_docx.py <đường_dẫn_docx> [<số_đoạn_tối_đa>] [<vị_trí_bắt_đầu>]")
        return 1
    
    docx_path = sys.argv[1]
    max_paragraphs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    
    extract_paragraphs(docx_path, max_paragraphs, offset)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 