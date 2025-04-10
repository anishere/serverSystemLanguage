#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chương trình dịch văn bản từ ảnh sử dụng OCR và OpenAI GPT-4o-mini

Quy trình:
1. Đọc ảnh và tự động nhận dạng ngôn ngữ nguồn
2. Phát hiện vùng bounding box chứa văn bản (EasyOCR hoặc PaddleOCR)
3. Crop vùng bounding box
4. Tiền xử lý ảnh đã cắt (lọc nhiễu, tăng độ tương phản, làm sắc nét)
5. Sử dụng GPT để trích xuất văn bản từ ảnh đã xử lý
6. Dịch văn bản trích xuất
7. Vẽ văn bản đã dịch vào vùng crop (căn giữa)
8. Chèn vùng đã xử lý lại vào ảnh gốc

Yêu cầu:
- opencv-python
- numpy
- requests
- Pillow
- python-dotenv
- matplotlib
- easyocr

Cách sử dụng:
python main.py --image_path anh.jpg [--target_lang Vietnamese] [--show]
"""

import os
import argparse
import time
import shutil
import cv2
import sys
import numpy as np
from dotenv import load_dotenv

# Thêm thư mục steps vào đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
steps_dir = os.path.join(current_dir, "steps")
sys.path.append(steps_dir)

# Nhập các module từ các file bước
from step1_read_image import read_image
from detect_language import detect_language_from_image
from step2_text_detection import detect_text_areas
from step3_crop_boxes import crop_text_areas, whiten_text_areas
from step4_preprocess_image import process_images as preprocess_images
from step5_extract_text import extract_text_with_gpt, process_crops
from step6_translate_text import translate_text, process_translations
from step7_draw_translated_text import process_images
from step8_merge_image import merge_translated_regions

# Tải biến môi trường
load_dotenv()

def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Dịch văn bản từ ảnh')
    parser.add_argument('--image_path', required=True, help='Đường dẫn đến ảnh đầu vào')
    parser.add_argument('--output_dir', help='Thư mục lưu kết quả (mặc định: thư mục cùng với ảnh)')
    parser.add_argument('--source_lang', help='Ngôn ngữ nguồn (tự động phát hiện nếu không cung cấp)')
    parser.add_argument('--target_lang', default='Vietnamese', help='Ngôn ngữ đích (mặc định: tiếng Việt)')
    parser.add_argument('--use_easyocr', action='store_true', help='Sử dụng EasyOCR thay vì PaddleOCR')
    parser.add_argument('--use_fallback', action='store_true', help='Sử dụng phương pháp dự phòng nếu OCR không hoạt động')
    parser.add_argument('--skip_preprocess', action='store_true', help='Bỏ qua bước tiền xử lý ảnh')
    parser.add_argument('--api_key', help='OpenAI API key (nếu không cung cấp, sẽ lấy từ biến môi trường)')
    parser.add_argument('--bg_color', default='white', help='Màu nền cho văn bản dịch')
    parser.add_argument('--text_color', default='black', help='Màu chữ cho văn bản dịch')
    parser.add_argument('--show', action='store_true', help='Hiển thị kết quả')
    return parser.parse_args()

# Bảng chuyển đổi từ tên ngôn ngữ đầy đủ sang mã OCR
language_to_ocr_code = {
    'Vietnamese': 'vi',
    'English': 'en',
    'Chinese': 'ch',
    'Chinese Traditional': 'ch_tra',
    'Japanese': 'ja',
    'Korean': 'ko',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Spanish': 'es',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Arabic': 'ar',
    'Hindi': 'hi',
    'Thai': 'th'
}

def create_output_dirs(image_path, output_dir=None):
    """
    Tạo các thư mục đầu ra.
    
    Args:
        image_path: Đường dẫn đến ảnh đầu vào
        output_dir: Thư mục đầu ra (nếu None, sẽ tạo thư mục mới trong cùng thư mục với ảnh)
        
    Returns:
        Dict chứa đường dẫn các thư mục
    """
    # Lấy tên file từ đường dẫn
    basename = os.path.basename(image_path)
    filename = os.path.splitext(basename)[0]
    
    # Xác định thư mục gốc
    if output_dir:
        root_dir = output_dir
    else:
        root_dir = os.path.join(os.path.dirname(image_path), f"{filename}_translated")
    
    # Tạo thư mục gốc
    os.makedirs(root_dir, exist_ok=True)
    
    # Tạo các thư mục con
    crops_dir = os.path.join(root_dir, "crops")
    preprocessed_dir = os.path.join(root_dir, "preprocessed")
    translated_dir = os.path.join(root_dir, "translated_crops")
    
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(translated_dir, exist_ok=True)
    
    # Tạo đường dẫn cho các file
    paths = {
        "root_dir": root_dir,
        "crops_dir": crops_dir,
        "preprocessed_dir": preprocessed_dir,
        "translated_dir": translated_dir,
        "whitened_image": os.path.join(root_dir, "whitened_image.jpg"),
        "detected_boxes": os.path.join(root_dir, "detected_boxes.jpg"),
        "cropped_info": os.path.join(crops_dir, "cropped_info.txt"),
        "preprocessed_info": os.path.join(preprocessed_dir, "processed_info.txt"),
        "extracted_texts": os.path.join(root_dir, "extracted_texts.txt"),
        "translated_texts": os.path.join(root_dir, "translated_texts.txt"),
        "translated_info": os.path.join(translated_dir, "translated_info.txt"),
        "final_image": os.path.join(root_dir, "final_translated_image.jpg")
    }
    
    return paths

def save_detected_boxes_image(image_path, text_areas, output_path):
    """
    Lưu ảnh với các vùng đã phát hiện.
    
    Args:
        image_path: Đường dẫn đến ảnh gốc
        text_areas: Danh sách các vùng văn bản
        output_path: Đường dẫn lưu ảnh kết quả
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return
    
    # Tạo bản sao của ảnh để vẽ lên
    result_image = image.copy()
    
    # Vẽ bounding box cho từng vùng văn bản
    for idx, (box, text, confidence) in enumerate(text_areas):
        # Chuyển box thành dạng points cho hàm vẽ
        points = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        
        # Vẽ bounding box
        cv2.polylines(result_image, [points], True, (0, 255, 0), 2)
        
        # Vẽ số thứ tự
        x, y = points[0][0]
        cv2.putText(result_image, f"#{idx+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Lưu ảnh kết quả
    cv2.imwrite(output_path, result_image)
    print(f"Đã lưu ảnh kết quả vào: {output_path}")

def main():
    """Hàm chính của chương trình."""
    # Phân tích tham số
    args = parse_args()
    
    # Kiểm tra file ảnh đầu vào
    if not os.path.exists(args.image_path):
        print(f"Lỗi: Không tìm thấy ảnh {args.image_path}")
        return
    
    # Lấy API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Lỗi: Không tìm thấy OpenAI API key. Cung cấp qua --api_key hoặc biến môi trường OPENAI_API_KEY")
        return
    
    # Tạo thư mục đầu ra
    paths = create_output_dirs(args.image_path, args.output_dir)
    
    print("\n===== BẮT ĐẦU QUY TRÌNH DỊCH VĂN BẢN TỪ ẢNH =====\n")
    start_time = time.time()
    
    # Bước 1: Đọc ảnh
    print("\n----- Bước 1: Đọc ảnh -----")
    image = read_image(args.image_path)
    if image is None:
        return
    
    # Sao chép ảnh gốc vào thư mục đầu ra
    original_copy = os.path.join(paths["root_dir"], os.path.basename(args.image_path))
    shutil.copy2(args.image_path, original_copy)
    print(f"Đã sao chép ảnh gốc vào: {original_copy}")
    
    # Nhận dạng ngôn ngữ nguồn nếu không được cung cấp
    if not args.source_lang:
        print("\n----- Phát hiện ngôn ngữ nguồn -----")
        source_lang_code, source_lang_name = detect_language_from_image(args.image_path, api_key)
    else:
        # Nếu đã cung cấp nguồn ngữ, chuyển đổi về dạng phù hợp
        if args.source_lang in language_to_ocr_code:
            # Nếu là tên đầy đủ
            source_lang_name = args.source_lang
            source_lang_code = language_to_ocr_code[args.source_lang]
        else:
            # Nếu là mã ngôn ngữ
            source_lang_code = args.source_lang
            # Tìm tên đầy đủ (nếu có)
            source_lang_name = next((name for name, code in language_to_ocr_code.items() 
                                if code == source_lang_code), source_lang_code)
        
        print(f"Sử dụng ngôn ngữ nguồn: {source_lang_name} ({source_lang_code})")
    
    # Bước 2: Phát hiện vùng văn bản (sử dụng OCR)
    print("\n----- Bước 2: Phát hiện vùng văn bản -----")
    # Sử dụng mã ngôn ngữ cho OCR
    text_areas = detect_text_areas(args.image_path, source_lang_code, args.use_easyocr, args.use_fallback)
    
    if not text_areas:
        print("Không phát hiện được vùng văn bản nào")
        return
    
    # Lưu ảnh với các vùng đã phát hiện
    save_detected_boxes_image(args.image_path, text_areas, paths["detected_boxes"])
    
    # Bước 3: Cắt vùng bounding box và tô trắng
    print("\n----- Bước 3: Cắt vùng bounding box và tô trắng -----")
    # Cắt vùng văn bản
    cropped_images = crop_text_areas(image, text_areas, paths["crops_dir"])
    
    # Tô trắng vùng văn bản
    whitened_image = whiten_text_areas(image, text_areas)
    cv2.imwrite(paths["whitened_image"], whitened_image)
    
    # Bước 4: Tiền xử lý ảnh
    processed_images = cropped_images
    if not args.skip_preprocess:
        print("\n----- Bước 4: Tiền xử lý ảnh -----")
        processed_images = preprocess_images(cropped_images, paths["preprocessed_dir"], args.show)
    else:
        print("\n----- Bỏ qua bước 4: Tiền xử lý ảnh -----")
    
    # Bước 5: Trích xuất văn bản từ các ảnh đã xử lý
    print("\n----- Bước 5: Trích xuất văn bản -----")
    processed_crops = process_crops(processed_images, api_key, paths["extracted_texts"])
    
    # Bước 6: Dịch văn bản đã trích xuất
    print("\n----- Bước 6: Dịch văn bản -----")
    # Sử dụng tên ngôn ngữ đầy đủ cho GPT
    translated_texts = process_translations(processed_crops, args.target_lang, api_key, paths["translated_texts"])
    
    # Bước 7: Vẽ văn bản đã dịch vào vùng crop
    print("\n----- Bước 7: Vẽ văn bản đã dịch -----")
    processed_images = process_images(
        translated_texts,
        paths["crops_dir"],
        paths["translated_dir"],
        args.bg_color,
        args.text_color,
        0  # Kích thước font tự động
    )
    
    # Bước 8: Chèn vùng đã xử lý vào ảnh gốc
    print("\n----- Bước 8: Chèn vùng đã xử lý vào ảnh gốc -----")
    result_image = merge_translated_regions(whitened_image, processed_images, cropped_images)
    cv2.imwrite(paths["final_image"], result_image)
    
    end_time = time.time()
    print(f"\nTổng thời gian xử lý: {end_time - start_time:.2f} giây")
    print(f"Đã lưu ảnh kết quả cuối cùng vào: {paths['final_image']}")
    
    # Hiển thị kết quả nếu yêu cầu
    if args.show:
        cv2.imshow('Original Image', image)
        cv2.imshow('Translated Image', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n===== QUY TRÌNH HOÀN THÀNH =====\n")

if __name__ == "__main__":
    main() 