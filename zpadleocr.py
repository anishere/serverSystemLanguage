#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chương trình Phát hiện Văn bản bằng PaddleOCR (Phiên bản đơn giản)

Yêu cầu:
    - paddle==2.5.2
    - paddleocr>=2.6.0.1
    - opencv-python
    - numpy

Cách sử dụng:
    python simple_paddle_ocr.py --image_path test.jpg --output_path result.jpg --lang vi
"""

import os
import argparse
import cv2
import numpy as np
import logging
import time

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def timer(name):
    """Decorator đo thời gian thực thi của hàm."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{name}: {end_time - start_time:.2f} giây")
            return result
        return wrapper
    return decorator

def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Phát hiện văn bản sử dụng PaddleOCR')
    
    # Tham số cơ bản
    parser.add_argument('--image_path', required=True, help='Đường dẫn đến hình ảnh đầu vào')
    parser.add_argument('--output_path', default='output_result.jpg', help='Đường dẫn để lưu hình ảnh kết quả')
    parser.add_argument('--bbox_file', default='bbox_coordinates.txt', help='Đường dẫn để lưu tọa độ bounding box')
    parser.add_argument('--show', action='store_true', help='Hiển thị kết quả')
    parser.add_argument('--lang', default='en', help='Ngôn ngữ cho OCR: en, vi, ch, ja, ko...')
    parser.add_argument('--use_angle_cls', action='store_true', help='Sử dụng bộ phân loại góc cho hướng văn bản')
    
    return parser.parse_args()

def find_font():
    """Tìm font phù hợp để hiển thị kết quả."""
    possible_font_paths = [
        './fonts/simfang.ttf',
        './simfang.ttf',
        'C:/Windows/Fonts/Arial.ttf',
        'C:/Windows/Fonts/calibri.ttf',
        'C:/Windows/Fonts/tahoma.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/Library/Fonts/Arial.ttf',
    ]
    
    for path in possible_font_paths:
        if os.path.exists(path):
            return path
    
    return None

@timer("Thực hiện OCR")
def detect_text(image_path, lang='en', use_angle_cls=False):
    """
    Phát hiện văn bản trong ảnh.
    
    Args:
        image_path: Đường dẫn đến ảnh
        lang: Ngôn ngữ OCR
        use_angle_cls: Sử dụng bộ phân loại góc
        
    Returns:
        Kết quả OCR hoặc None nếu có lỗi
    """
    try:
        from paddleocr import PaddleOCR
        
        # Cấu hình OCR đơn giản
        config = {
            'use_angle_cls': use_angle_cls,
            'lang': lang,
            'use_gpu': False,
            'show_log': False
        }
        
        # Tối ưu cho một số ngôn ngữ
        if lang == 'vi':
            config['det_db_box_thresh'] = 0.6
            config['det_db_thresh'] = 0.3
        
        # Khởi tạo PaddleOCR
        ocr = PaddleOCR(**config)
        logger.info(f"Đang thực hiện OCR trên hình ảnh: {image_path}")
        
        # Thực hiện OCR
        result = ocr.ocr(image_path, cls=use_angle_cls)
        
        # Kiểm tra kết quả
        if not result or len(result) == 0:
            logger.warning("Không phát hiện văn bản trong hình ảnh")
            return []
        
        # Xử lý kết quả
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                return result[0]
            else:
                return result
        
        return []
    
    except ImportError:
        logger.error("Thiếu thư viện PaddleOCR. Cài đặt: pip install paddlepaddle==2.5.2 paddleocr")
        return None
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện OCR: {e}")
        return None

@timer("Lưu kết quả")
def save_results(image_path, det_results, output_path, bbox_file):
    """
    Lưu kết quả OCR.
    
    Args:
        image_path: Đường dẫn đến ảnh gốc
        det_results: Kết quả phát hiện
        output_path: Đường dẫn lưu ảnh kết quả
        bbox_file: Đường dẫn lưu thông tin bounding box
    """
    if not det_results:
        logger.warning("Không có kết quả để lưu")
        return
    
    try:
        from paddleocr import draw_ocr
        
        # Đọc ảnh gốc
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Không thể đọc ảnh từ {image_path}")
            return
        
        # Tìm font hiển thị
        font_path = find_font()
        
        # Chuẩn bị dữ liệu cho draw_ocr
        boxes = [line[0] for line in det_results]
        txts = [line[1][0] for line in det_results]
        scores = [line[1][1] for line in det_results]
        
        # Vẽ kết quả OCR lên ảnh
        im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
        cv2.imwrite(output_path, im_show)
        logger.info(f"Đã lưu ảnh kết quả vào {output_path}")
        
        # Lưu thông tin bounding box
        with open(bbox_file, 'w', encoding='utf-8') as f:
            # Thêm header cho dễ đọc
            f.write("="*50 + "\n")
            f.write(" DANH SÁCH BOUNDING BOX VÀ VĂN BẢN PHÁT HIỆN ĐƯỢC \n")
            f.write("="*50 + "\n\n")
            
            for idx, (box, text) in enumerate(zip(boxes, txts)):
                pts = np.array(box).astype(np.int32).reshape(-1, 2)
                # Tạo chuỗi tọa độ, ví dụ: "(x1,y1) (x2,y2) (x3,y3) (x4,y4)"
                coord_str = " ".join([f"({p[0]},{p[1]})" for p in pts])
                f.write(f"Bounding box {idx+1}:\n")
                f.write(f"  Tọa độ: {coord_str}\n")
                f.write(f"  Văn bản: {text}\n\n")
            
            # Thêm footer
            f.write("="*50 + "\n")
        
        logger.info(f"Đã lưu tọa độ bounding box vào {bbox_file}")
        print(f"\nKết quả chi tiết đã được lưu vào file: {bbox_file}")
        print(f"Ảnh kết quả đã được lưu vào: {output_path}")
    
    except ImportError:
        logger.error("Thiếu thư viện PaddleOCR để vẽ kết quả")
    except Exception as e:
        logger.error(f"Lỗi khi lưu kết quả: {e}")

def display_results(output_path):
    """Hiển thị kết quả."""
    try:
        img = cv2.imread(output_path)
        cv2.imshow('OCR Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        logger.error(f"Không thể hiển thị ảnh: {e}")

def main():
    """Hàm chính của chương trình."""
    args = parse_args()
    
    # Kiểm tra file đầu vào
    if not os.path.exists(args.image_path):
        logger.error(f"Không tìm thấy hình ảnh: {args.image_path}")
        return
    
    # Phát hiện văn bản
    det_results = detect_text(args.image_path, args.lang, args.use_angle_cls)
    
    if det_results:
        # In kết quả phát hiện
        logger.info(f"Đã phát hiện {len(det_results)} dòng văn bản:")
        
        # In rõ ràng danh sách bounding box và văn bản
        print("\n" + "="*50)
        print(" DANH SÁCH BOUNDING BOX VÀ VĂN BẢN PHÁT HIỆN ĐƯỢC ")
        print("="*50)
        
        for idx, (bbox, (text, confidence)) in enumerate(det_results):
            pts = np.array(bbox).astype(np.int32).reshape(-1, 2)
            coord_str = " ".join([f"({p[0]},{p[1]})" for p in pts])
            
            print(f"\nBounding box {idx+1}:")
            print(f"  Tọa độ: {coord_str}")
            print(f"  Văn bản: {text}")
            print(f"  Độ tin cậy: {confidence:.4f}")
        
        print("="*50 + "\n")
        
        # Lưu kết quả
        save_results(args.image_path, det_results, args.output_path, args.bbox_file)
        
        # Hiển thị kết quả nếu yêu cầu
        if args.show:
            display_results(args.output_path)
    else:
        logger.error("Không thể thực hiện OCR hoặc không có kết quả")

if __name__ == "__main__":
    main()