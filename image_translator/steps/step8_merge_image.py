#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bước 8: Chèn vùng đã xử lý trở lại ảnh gốc
"""

import os
import cv2
import numpy as np
import argparse
from step1_read_image import read_image

def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Chèn vùng đã xử lý trở lại ảnh gốc')
    parser.add_argument('--image_path', required=True, help='Đường dẫn đến ảnh gốc')
    parser.add_argument('--whitened_image', help='Đường dẫn đến ảnh đã tô trắng')
    parser.add_argument('--translated_dir', required=True, help='Thư mục chứa các ảnh đã dịch')
    parser.add_argument('--translated_info', help='Đường dẫn đến file thông tin các ảnh đã dịch')
    parser.add_argument('--cropped_info', help='Đường dẫn đến file thông tin các ảnh đã cắt')
    parser.add_argument('--output_path', default='final_translated_image.jpg', help='Đường dẫn lưu ảnh kết quả')
    parser.add_argument('--show', action='store_true', help='Hiển thị ảnh')
    return parser.parse_args()

def load_translated_info(translated_info_path):
    """
    Đọc thông tin các ảnh đã dịch từ file.
    
    Args:
        translated_info_path: Đường dẫn đến file thông tin
        
    Returns:
        List thông tin các ảnh đã dịch
    """
    translated_images = []
    
    try:
        with open(translated_info_path, 'r', encoding='utf-8') as f:
            # Bỏ qua dòng header
            header = f.readline()
            
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 5:
                    translated_images.append({
                        'index': int(parts[0]),
                        'crop_path': parts[1],
                        'output_path': parts[2],
                        'extracted_text': parts[3],
                        'translated_text': parts[4]
                    })
        
        print(f"Đã đọc thông tin {len(translated_images)} ảnh đã dịch từ {translated_info_path}")
    
    except Exception as e:
        print(f"Lỗi khi đọc file thông tin các ảnh đã dịch: {e}")
    
    return translated_images

def load_cropped_info(cropped_info_path):
    """
    Đọc thông tin các ảnh đã cắt từ file.
    
    Args:
        cropped_info_path: Đường dẫn đến file thông tin
        
    Returns:
        List thông tin các ảnh đã cắt
    """
    cropped_images = []
    
    try:
        with open(cropped_info_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 6:
                    index = int(parts[0])
                    path = parts[1]
                    box_str = parts[2]
                    rect_str = parts[3]
                    text = parts[4]
                    confidence = float(parts[5])
                    
                    # Chuyển đổi chuỗi box thành danh sách tọa độ
                    coords = box_str.split(',')
                    box = []
                    for i in range(0, len(coords), 2):
                        box.append([float(coords[i]), float(coords[i+1])])
                    
                    # Chuyển đổi chuỗi rect thành danh sách
                    rect_parts = rect_str.split(',')
                    rect = [int(rect_parts[0]), int(rect_parts[1]), int(rect_parts[2]), int(rect_parts[3])]
                    
                    cropped_images.append({
                        'index': index,
                        'path': path,
                        'box': box,
                        'rect': rect,
                        'text': text,
                        'confidence': confidence
                    })
        
        print(f"Đã đọc thông tin {len(cropped_images)} ảnh đã cắt từ {cropped_info_path}")
    
    except Exception as e:
        print(f"Lỗi khi đọc file thông tin các ảnh đã cắt: {e}")
    
    return cropped_images

def merge_translated_regions(original_image, translated_info, cropped_info=None):
    """
    Chèn các vùng đã dịch vào ảnh gốc.
    
    Args:
        original_image: Ảnh gốc
        translated_info: Thông tin các ảnh đã dịch
        cropped_info: Thông tin các ảnh đã cắt (nếu có)
        
    Returns:
        Ảnh kết quả
    """
    # Tạo bản sao của ảnh gốc
    result_image = original_image.copy()
    
    # Nếu có thông tin các ảnh đã cắt, sử dụng thông tin đó để lấy vị trí chèn
    if cropped_info:
        # Tạo ánh xạ giữa chỉ số và thông tin crop
        crop_map = {item['index']: item for item in cropped_info}
        
        # Chèn từng vùng đã dịch
        for item in translated_info:
            idx = item['index']
            
            # Lấy thông tin về vùng đã cắt
            if idx in crop_map:
                crop_item = crop_map[idx]
                x, y, w, h = crop_item['rect']
                
                # Đọc ảnh đã dịch
                translated_img = cv2.imread(item['output_path'])
                
                if translated_img is not None:
                    # Đảm bảo ảnh đã dịch có đúng kích thước với vùng cắt
                    if translated_img.shape[:2] != (h, w):
                        # Resize chính xác với kích thước gốc
                        translated_img = cv2.resize(translated_img, (w, h), interpolation=cv2.INTER_AREA)
                    
                    # Kiểm tra ranh giới để tránh lỗi
                    if y >= 0 and x >= 0 and y + h <= result_image.shape[0] and x + w <= result_image.shape[1]:
                        # Chèn vào ảnh gốc
                        result_image[y:y+h, x:x+w] = translated_img
                        print(f"Đã chèn vùng {idx} vào vị trí ({x}, {y}) với kích thước {w}x{h}")
                    else:
                        print(f"Cảnh báo: Vùng {idx} có tọa độ nằm ngoài ảnh gốc, bỏ qua")
    else:
        # Không có thông tin crop, dựa vào tên file để xác định vị trí chèn
        for item in translated_info:
            # Trích xuất thông tin từ tên file
            basename = os.path.basename(item['crop_path'])
            if basename.startswith('crop_') and '_' in basename:
                try:
                    # Phân tích tên file để lấy chỉ số
                    idx = int(basename.split('_')[1].split('.')[0])
                    
                    # Tìm vùng cắt trong ảnh gốc (đơn giản hóa, cần cải thiện)
                    # Giả định rằng các vùng được cắt từ trái sang phải, trên xuống dưới
                    height, width = original_image.shape[:2]
                    rows, cols = 3, 3  # Giả định có 3x3 vùng
                    cell_height, cell_width = height // rows, width // cols
                    
                    row = (idx - 1) // cols
                    col = (idx - 1) % cols
                    
                    x = col * cell_width
                    y = row * cell_height
                    w, h = cell_width, cell_height
                    
                    # Đọc ảnh đã dịch
                    translated_img = cv2.imread(item['output_path'])
                    
                    if translated_img is not None:
                        # Đảm bảo kích thước ảnh đã dịch khớp với vùng cắt
                        if translated_img.shape[:2] != (h, w):
                            translated_img = cv2.resize(translated_img, (w, h), interpolation=cv2.INTER_AREA)
                        
                        # Kiểm tra ranh giới
                        if y >= 0 and x >= 0 and y + h <= result_image.shape[0] and x + w <= result_image.shape[1]:
                            # Chèn vào ảnh gốc
                            result_image[y:y+h, x:x+w] = translated_img
                            print(f"Đã chèn vùng {idx} vào vị trí ({x}, {y}) với kích thước {w}x{h}")
                        else:
                            print(f"Cảnh báo: Vùng {idx} có tọa độ nằm ngoài ảnh gốc, bỏ qua")
                except (ValueError, IndexError) as e:
                    print(f"Lỗi khi xử lý file {basename}: {e}")
    
    return result_image

def main():
    # Phân tích tham số
    args = parse_args()
    
    # Đọc ảnh gốc
    original_image = read_image(args.image_path)
    if original_image is None:
        return False
    
    # Sử dụng ảnh đã tô trắng nếu có
    if args.whitened_image and os.path.exists(args.whitened_image):
        print(f"Sử dụng ảnh đã tô trắng: {args.whitened_image}")
        original_image = cv2.imread(args.whitened_image)
    
    # Thông tin các ảnh đã dịch
    translated_info = []
    
    # Đọc thông tin từ file nếu có
    translated_info_path = args.translated_info or os.path.join(args.translated_dir, "translated_info.txt")
    if os.path.exists(translated_info_path):
        translated_info = load_translated_info(translated_info_path)
    
    # Nếu không có thông tin, tìm các file ảnh trong thư mục
    if not translated_info:
        print(f"Không tìm thấy file thông tin {translated_info_path}, tìm ảnh trong thư mục...")
        
        # Tìm các file ảnh đã dịch trong thư mục
        translated_files = []
        for filename in os.listdir(args.translated_dir):
            if filename.lower().startswith('translated_') and filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                try:
                    # Lấy chỉ số từ tên file
                    idx = int(filename.split('_')[1].split('.')[0])
                    translated_files.append((idx, os.path.join(args.translated_dir, filename)))
                except (ValueError, IndexError):
                    continue
        
        # Sắp xếp theo chỉ số
        translated_files.sort()
        
        # Tạo thông tin từ các file đã tìm thấy
        for idx, file_path in translated_files:
            translated_info.append({
                'index': idx,
                'crop_path': f"crop_{idx}.jpg",
                'output_path': file_path,
                'extracted_text': "",
                'translated_text': ""
            })
    
    if not translated_info:
        print("Không tìm thấy thông tin các ảnh đã dịch")
        return False
    
    # Thông tin các ảnh đã cắt
    cropped_info = []
    
    # Đọc thông tin từ file nếu có
    if args.cropped_info and os.path.exists(args.cropped_info):
        cropped_info = load_cropped_info(args.cropped_info)
    
    # Chèn các vùng đã dịch vào ảnh gốc
    result_image = merge_translated_regions(original_image, translated_info, cropped_info)
    
    # Lưu ảnh kết quả
    output_path = os.path.join(os.path.dirname(args.image_path), args.output_path)
    cv2.imwrite(output_path, result_image)
    print(f"Đã lưu ảnh kết quả vào: {output_path}")
    
    # Hiển thị kết quả nếu yêu cầu
    if args.show:
        # Hiển thị ảnh gốc và ảnh kết quả
        cv2.imshow('Original Image', original_image)
        cv2.imshow('Translated Image', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return True

if __name__ == "__main__":
    main() 