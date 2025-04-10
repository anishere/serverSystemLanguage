#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bước 3: Cắt vùng bounding box từ ảnh gốc
"""

import os
import cv2
import numpy as np
import argparse
import sys

# Thêm thư mục hiện tại vào path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from step1_read_image import read_image

# Import từ module OCR mới
try:
    from step2_text_detection import detect_text_areas
    ocr_available = True
except ImportError:
    ocr_available = False

def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Cắt vùng bounding box từ ảnh gốc')
    parser.add_argument('--image_path', required=True, help='Đường dẫn đến ảnh đầu vào')
    parser.add_argument('--text_areas_file', help='Đường dẫn đến file chứa thông tin vùng văn bản')
    parser.add_argument('--output_dir', default='crops', help='Thư mục lưu các ảnh đã cắt')
    parser.add_argument('--lang', default='en', help='Ngôn ngữ cho OCR: en, vi, ja, ko...')
    parser.add_argument('--use_easyocr', action='store_true', help='Sử dụng EasyOCR thay vì PaddleOCR')
    parser.add_argument('--show', action='store_true', help='Hiển thị ảnh')
    return parser.parse_args()

def load_text_areas(text_areas_file):
    """
    Đọc thông tin vùng văn bản từ file.
    
    Args:
        text_areas_file: Đường dẫn đến file chứa thông tin vùng văn bản
        
    Returns:
        List các vùng văn bản (bounding_box, text, confidence)
    """
    text_areas = []
    
    try:
        with open(text_areas_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    idx = int(parts[0])
                    box_str = parts[1]
                    text = parts[2]
                    confidence = float(parts[3])
                    
                    # Chuyển đổi chuỗi box thành danh sách tọa độ
                    coords = box_str.split(',')
                    box = []
                    for i in range(0, len(coords), 2):
                        box.append([float(coords[i]), float(coords[i+1])])
                    
                    text_areas.append((box, text, confidence))
        
        print(f"Đã đọc {len(text_areas)} vùng văn bản từ file {text_areas_file}")
    
    except Exception as e:
        print(f"Lỗi khi đọc file vùng văn bản: {e}")
    
    return text_areas

def crop_text_areas(image, text_areas, output_dir):
    """
    Cắt vùng văn bản từ ảnh gốc.
    
    Args:
        image: Ảnh gốc
        text_areas: Danh sách vùng văn bản
        output_dir: Thư mục lưu ảnh đã cắt
        
    Returns:
        Danh sách đường dẫn đến các ảnh đã cắt
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Danh sách lưu đường dẫn các ảnh đã cắt
    cropped_images_paths = []
    
    # Thông tin ảnh
    height, width = image.shape[:2]
    
    # Cắt và lưu từng vùng văn bản
    for idx, (box, text, confidence) in enumerate(text_areas):
        # Chuyển đổi box thành dạng số nguyên
        box_points = np.array(box, dtype=np.int32)
        
        # Tính toán bounding rectangle
        rect = cv2.boundingRect(box_points)
        x, y, w, h = rect
        
        # Đảm bảo kích thước hợp lệ
        x = max(0, x)
        y = max(0, y)
        w = min(width - x, w)
        h = min(height - y, h)
        
        # Cắt vùng ảnh
        cropped = image[y:y+h, x:x+w]
        
        # Tạo đường dẫn lưu ảnh
        output_path = os.path.join(output_dir, f"crop_{idx+1}.jpg")
        
        # Lưu ảnh đã cắt
        cv2.imwrite(output_path, cropped)
        
        # Lưu thông tin
        cropped_images_paths.append({
            'path': output_path,
            'index': idx+1,
            'box': box,
            'text': text,
            'confidence': confidence,
            'rect': rect  # Lưu thông tin rectangle để dễ dàng chèn lại sau này
        })
        
        print(f"Đã cắt và lưu vùng #{idx+1} vào: {output_path}")
    
    return cropped_images_paths

def whiten_text_areas(image, text_areas):
    """
    Tô trắng vùng chữ trong ảnh gốc.
    
    Args:
        image: Ảnh gốc
        text_areas: Danh sách vùng văn bản
        
    Returns:
        Ảnh sau khi tô trắng các vùng văn bản
    """
    # Tạo bản sao của ảnh
    whitened_image = image.copy()
    
    # Tô trắng từng vùng văn bản
    for box, _, _ in text_areas:
        # Chuyển đổi box thành dạng số nguyên
        points = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        
        # Tô trắng vùng đa giác
        cv2.fillPoly(whitened_image, [points], (255, 255, 255))
    
    return whitened_image

def main():
    # Phân tích tham số
    args = parse_args()
    
    # Đọc ảnh gốc
    image = read_image(args.image_path)
    if image is None:
        return False
    
    # Lấy thông tin vùng văn bản
    if args.text_areas_file and os.path.exists(args.text_areas_file):
        # Đọc từ file nếu có
        text_areas = load_text_areas(args.text_areas_file)
    else:
        # Phát hiện mới nếu không có file
        print("Không tìm thấy file thông tin vùng văn bản, tiến hành phát hiện mới...")
        if ocr_available:
            text_areas = detect_text_areas(args.image_path, args.lang, args.use_easyocr)
        else:
            print("Lỗi: Không tìm thấy module OCR. Cài đặt cần thiết và kiểm tra step2_text_detection.py")
            return False
    
    if not text_areas:
        print("Không có vùng văn bản nào để xử lý")
        return False
    
    # Cắt các vùng văn bản
    output_dir = os.path.join(os.path.dirname(args.image_path), args.output_dir)
    cropped_images = crop_text_areas(image, text_areas, output_dir)
    
    # Tô trắng các vùng văn bản trong ảnh gốc
    whitened_image = whiten_text_areas(image, text_areas)
    
    # Lưu ảnh đã tô trắng
    whitened_path = os.path.join(os.path.dirname(args.image_path), "whitened_image.jpg")
    cv2.imwrite(whitened_path, whitened_image)
    print(f"Đã lưu ảnh đã tô trắng vào: {whitened_path}")
    
    # Lưu thông tin các ảnh đã cắt vào file
    cropped_info_path = os.path.join(output_dir, "cropped_info.txt")
    with open(cropped_info_path, 'w', encoding='utf-8') as f:
        for crop_info in cropped_images:
            rect = crop_info['rect']
            box_str = ",".join([f"{p[0]},{p[1]}" for p in crop_info['box']])
            f.write(f"{crop_info['index']}|{crop_info['path']}|{box_str}|{rect[0]},{rect[1]},{rect[2]},{rect[3]}|{crop_info['text']}|{crop_info['confidence']}\n")
    
    print(f"Đã lưu thông tin các ảnh đã cắt vào: {cropped_info_path}")
    
    # Hiển thị ảnh đã tô trắng nếu yêu cầu
    if args.show:
        cv2.imshow('Whitened Image', whitened_image)
        
        # Hiển thị các ảnh đã cắt
        for crop_info in cropped_images:
            crop_img = cv2.imread(crop_info['path'])
            if crop_img is not None:
                cv2.imshow(f"Crop #{crop_info['index']}", crop_img)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return True

if __name__ == "__main__":
    main() 