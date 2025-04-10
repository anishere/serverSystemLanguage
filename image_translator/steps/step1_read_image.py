#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bước 1: Đọc ảnh và hiển thị thông tin ảnh
"""

import os
import cv2
import argparse

def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Đọc ảnh và hiển thị thông tin')
    parser.add_argument('--image_path', required=True, help='Đường dẫn đến ảnh đầu vào')
    parser.add_argument('--show', action='store_true', help='Hiển thị ảnh')
    return parser.parse_args()

def read_image(image_path):
    """
    Đọc ảnh từ đường dẫn và trả về thông tin cơ bản.
    
    Args:
        image_path: Đường dẫn đến ảnh
        
    Returns:
        image: Ảnh đã đọc
    """
    print(f"Đọc ảnh từ: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy ảnh tại đường dẫn {image_path}")
        return None
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
        return None
    
    # Lấy thông tin ảnh
    height, width, channels = image.shape
    
    print(f"Thông tin ảnh:")
    print(f"- Kích thước: {width} x {height} pixels")
    print(f"- Số kênh màu: {channels}")
    print(f"- Loại dữ liệu: {image.dtype}")
    
    return image

def main():
    args = parse_args()
    
    # Đọc ảnh
    image = read_image(args.image_path)
    
    if image is not None and args.show:
        # Hiển thị ảnh
        print("Hiển thị ảnh (nhấn phím bất kỳ để đóng)...")
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return image is not None

if __name__ == "__main__":
    main() 