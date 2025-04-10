#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bước 4: Tiền xử lý ảnh đã cắt để cải thiện độ chính xác khi trích xuất văn bản
"""

import os
import cv2
import numpy as np
import argparse
import sys

# Thêm thư mục hiện tại vào path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from step1_read_image import read_image

def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Tiền xử lý ảnh đã cắt')
    parser.add_argument('--crop_dir', required=True, help='Thư mục chứa các ảnh đã cắt')
    parser.add_argument('--cropped_info', help='Đường dẫn đến file chứa thông tin các ảnh đã cắt')
    parser.add_argument('--output_dir', default='preprocessed', help='Thư mục lưu các ảnh đã xử lý')
    parser.add_argument('--show', action='store_true', help='Hiển thị ảnh kết quả')
    return parser.parse_args()

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

def get_image_files(directory):
    """
    Lấy danh sách các file ảnh trong thư mục.
    
    Args:
        directory: Đường dẫn thư mục
        
    Returns:
        List đường dẫn các file ảnh
    """
    image_files = []
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(directory, filename))
    
    # Sắp xếp theo số thứ tự trong tên file
    def extract_number(filename):
        try:
            # Trích xuất số từ tên file crop_X.jpg
            return int(os.path.basename(filename).split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            return 0
    
    return sorted(image_files, key=extract_number)

def adaptive_threshold(image):
    """
    Áp dụng ngưỡng thích ứng để cải thiện độ tương phản của văn bản.
    
    Args:
        image: Ảnh đầu vào
        
    Returns:
        Ảnh đã áp dụng ngưỡng
    """
    # Chuyển về ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Áp dụng làm mờ Gaussian để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Áp dụng ngưỡng thích ứng
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh

def denoise_image(image):
    """
    Loại bỏ nhiễu từ ảnh.
    
    Args:
        image: Ảnh đầu vào
        
    Returns:
        Ảnh đã loại bỏ nhiễu
    """
    # Loại bỏ nhiễu không phá hủy cạnh
    if len(image.shape) == 3:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:
        denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    return denoised

def enhance_contrast(image):
    """
    Cải thiện độ tương phản của ảnh.
    
    Args:
        image: Ảnh đầu vào
        
    Returns:
        Ảnh đã cải thiện độ tương phản
    """
    # Chuyển đổi không gian màu
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
    else:
        l = image.copy()
    
    # Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Hợp nhất các kênh và chuyển lại không gian màu
    if len(image.shape) == 3:
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        enhanced = cl
    
    return enhanced

def sharpen_image(image):
    """
    Làm sắc nét ảnh.
    
    Args:
        image: Ảnh đầu vào
        
    Returns:
        Ảnh đã làm sắc nét
    """
    # Tạo kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    # Áp dụng kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    
    return sharpened

def preprocess_image(image_path, output_path):
    """
    Tiền xử lý ảnh để cải thiện độ chính xác khi trích xuất văn bản.
    
    Args:
        image_path: Đường dẫn đến ảnh đầu vào
        output_path: Đường dẫn lưu ảnh đã xử lý
        
    Returns:
        Ảnh đã xử lý
    """
    print(f"Đang xử lý ảnh: {image_path}")
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
        return None
    
    # Lưu ảnh gốc
    original = image.copy()
    
    # Loại bỏ nhiễu
    denoised = denoise_image(image)
    
    # Cải thiện độ tương phản
    enhanced = enhance_contrast(denoised)
    
    # Làm sắc nét
    sharpened = sharpen_image(enhanced)
    
    # Lưu ảnh đã xử lý
    cv2.imwrite(output_path, sharpened)
    print(f"Đã lưu ảnh đã xử lý vào: {output_path}")
    
    return sharpened, original

def process_images(cropped_images, output_dir, show=False):
    """
    Xử lý tất cả các ảnh đã cắt.
    
    Args:
        cropped_images: Danh sách thông tin các ảnh đã cắt
        output_dir: Thư mục lưu các ảnh đã xử lý
        show: Hiển thị kết quả
    
    Returns:
        Danh sách thông tin các ảnh đã xử lý
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Danh sách lưu thông tin các ảnh đã xử lý
    processed_images = []
    
    # Xử lý từng ảnh
    for crop_info in cropped_images:
        # Tạo đường dẫn đầu ra
        output_path = os.path.join(output_dir, f"processed_{crop_info['index']}.jpg")
        
        # Tiền xử lý ảnh
        result, original = preprocess_image(crop_info['path'], output_path)
        
        if result is not None:
            # Lưu thông tin
            processed_info = crop_info.copy()
            processed_info['processed_path'] = output_path
            processed_images.append(processed_info)
            
            # Hiển thị kết quả nếu yêu cầu
            if show:
                # Đặt kích thước phù hợp cho cửa sổ hiển thị
                scale = 1
                if original.shape[1] > 800:
                    scale = 800 / original.shape[1]
                
                resized_original = cv2.resize(original, (0, 0), fx=scale, fy=scale)
                resized_result = cv2.resize(result, (0, 0), fx=scale, fy=scale)
                
                cv2.imshow(f"Original #{crop_info['index']}", resized_original)
                cv2.imshow(f"Processed #{crop_info['index']}", resized_result)
    
    # Lưu thông tin các ảnh đã xử lý
    processed_info_path = os.path.join(output_dir, "processed_info.txt")
    with open(processed_info_path, 'w', encoding='utf-8') as f:
        for info in processed_images:
            rect = info['rect']
            box_str = ",".join([f"{p[0]},{p[1]}" for p in info['box']])
            f.write(f"{info['index']}|{info['path']}|{info['processed_path']}|{box_str}|{rect[0]},{rect[1]},{rect[2]},{rect[3]}|{info['text']}|{info['confidence']}\n")
    
    print(f"Đã lưu thông tin các ảnh đã xử lý vào: {processed_info_path}")
    
    return processed_images

def main():
    # Phân tích tham số
    args = parse_args()
    
    # Kiểm tra thư mục ảnh đã cắt
    if not os.path.exists(args.crop_dir):
        print(f"Lỗi: Không tìm thấy thư mục {args.crop_dir}")
        return False
    
    # Đọc thông tin các ảnh đã cắt
    cropped_images = []
    cropped_info_path = args.cropped_info or os.path.join(args.crop_dir, "cropped_info.txt")
    
    if os.path.exists(cropped_info_path):
        # Đọc từ file thông tin
        cropped_images = load_cropped_info(cropped_info_path)
    else:
        # Nếu không có file thông tin, tạo từ danh sách file ảnh
        print(f"Không tìm thấy file thông tin {cropped_info_path}, đang tạo thông tin từ các file ảnh...")
        
        image_files = get_image_files(args.crop_dir)
        for idx, image_path in enumerate(image_files):
            # Đọc ảnh để lấy kích thước
            image = cv2.imread(image_path)
            if image is not None:
                height, width = image.shape[:2]
                
                # Tạo thông tin giả về bounding box (hình chữ nhật đầy đủ)
                box = [[0, 0], [width, 0], [width, height], [0, height]]
                rect = [0, 0, width, height]
                
                cropped_images.append({
                    'index': idx + 1,
                    'path': image_path,
                    'box': box,
                    'rect': rect,
                    'text': f"Unknown text",
                    'confidence': 0.0
                })
    
    if not cropped_images:
        print("Không có ảnh nào để xử lý")
        return False
    
    # Tạo thư mục đầu ra
    output_dir = os.path.join(os.path.dirname(args.crop_dir), args.output_dir)
    
    # Xử lý các ảnh đã cắt
    processed_images = process_images(cropped_images, output_dir, args.show)
    
    # Hiển thị kết quả nếu yêu cầu
    if args.show:
        print("Nhấn phím bất kỳ để đóng cửa sổ hiển thị...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f"Đã xử lý {len(processed_images)} ảnh")
    
    return True

if __name__ == "__main__":
    main() 