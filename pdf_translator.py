#!/usr/bin/env python3
"""
Chuyển đổi file PDF sang Word (DOCX) mà không sử dụng Microsoft Word
Sử dụng thư viện pdf2docx để chuyển đổi.

Cách dùng:
    python convert_pdf_to_word.py input.pdf output.docx
"""

import argparse
from pdf2docx import Converter

def convert_pdf_to_docx(pdf_file, docx_file):
    try:
        print(f"Đang chuyển đổi {pdf_file} sang {docx_file}...")
        # Khởi tạo Converter với file PDF
        cv = Converter(pdf_file)
        # Chuyển đổi toàn bộ trang (start=0, end=None)
        cv.convert(docx_file, start=0, end=None)
        cv.close()
        print(f"Đã chuyển đổi thành công: {docx_file}")
    except Exception as e:
        print(f"Lỗi khi chuyển đổi: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Chuyển đổi file PDF sang DOCX mà không sử dụng MS Word."
    )
    parser.add_argument("input_pdf", help="Đường dẫn đến file PDF cần chuyển đổi")
    parser.add_argument("output_docx", help="Đường dẫn đến file DOCX đầu ra")
    args = parser.parse_args()

    convert_pdf_to_docx(args.input_pdf, args.output_docx)

if __name__ == "__main__":
    main()
