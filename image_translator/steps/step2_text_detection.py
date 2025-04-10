#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bước 2: Phát hiện vùng bounding box chứa văn bản sử dụng PaddleOCR hoặc EasyOCR
"""

import os
import cv2
import numpy as np
import argparse
import time
import logging
from step1_read_image import read_image

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Phát hiện văn bản bằng OCR')
    parser.add_argument('--image_path', required=True, help='Đường dẫn đến ảnh đầu vào')
    parser.add_argument('--output_path', default='output_boxes.jpg', help='Đường dẫn để lưu ảnh kết quả')
    parser.add_argument('--lang', default='en', help='Ngôn ngữ cho OCR: en, vi, ja, ko...')
    parser.add_argument('--use_easyocr', action='store_true', help='Sử dụng EasyOCR thay vì PaddleOCR')
    parser.add_argument('--use_fallback', action='store_true', help='Sử dụng phương pháp dự phòng nếu OCR không hoạt động')
    parser.add_argument('--show', action='store_true', help='Hiển thị ảnh')
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

def detect_text_with_paddleocr(image_path, lang='en'):
    """
    Phát hiện vùng văn bản trong ảnh sử dụng PaddleOCR.
    
    Args:
        image_path: Đường dẫn đến ảnh
        lang: Ngôn ngữ OCR
        
    Returns:
        List các tuples (bounding_box, text, confidence)
    """
    print(f"Phát hiện vùng văn bản trong ảnh bằng PaddleOCR: {image_path}")
    
    start_time = time.time()
    
    try:
        from paddleocr import PaddleOCR
        
        # Cấu hình PaddleOCR
        config = {
            'use_angle_cls': False,  # Tắt phân loại góc để giảm lỗi
            'lang': lang,
            'use_gpu': False,
            'show_log': True        # Hiển thị log để theo dõi
        }
        
        # Khởi tạo PaddleOCR
        print("Đang khởi tạo PaddleOCR...")
        ocr = PaddleOCR(**config)
        print("Đã khởi tạo PaddleOCR, đang thực hiện OCR...")
        
        # Thực hiện OCR
        result = ocr.ocr(image_path, cls=False)  # Tắt cls
        
        # Kiểm tra kết quả
        if not result or len(result) == 0:
            print("Không phát hiện văn bản trong ảnh")
            return []
        
        # Xử lý kết quả
        text_areas = []
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                det_results = result[0]
            else:
                det_results = result
                
            # Chuyển đổi kết quả thành danh sách (bbox, text, confidence)
            for box, (text, confidence) in det_results:
                text_areas.append((box, text, confidence))
        
        end_time = time.time()
        print(f"Thời gian phát hiện: {end_time - start_time:.2f} giây")
        print(f"Đã phát hiện {len(text_areas)} vùng văn bản")
        
        return text_areas
    
    except ImportError as e:
        print(f"Lỗi: Thiếu thư viện PaddleOCR. Chi tiết: {e}")
        print("Cài đặt: pip install paddlepaddle==2.5.2 paddleocr")
        return []
    except Exception as e:
        print(f"Lỗi khi phát hiện văn bản: {e}")
        return []

def detect_text_with_easyocr(image_path, lang='en'):
    """
    Phát hiện vùng văn bản trong ảnh sử dụng EasyOCR.
    
    Args:
        image_path: Đường dẫn đến ảnh
        lang: Ngôn ngữ OCR
        
    Returns:
        List các tuples (bounding_box, text, confidence)
    """
    print(f"Phát hiện vùng văn bản trong ảnh bằng EasyOCR: {image_path}")
    
    start_time = time.time()
    
    try:
        import easyocr
        
        # Xác định ngôn ngữ
        languages = [lang]
        if lang != 'en':
            languages = ['en', lang]  # Thêm tiếng Anh để tăng độ chính xác
        
        # Khởi tạo EasyOCR reader
        print("Đang khởi tạo EasyOCR...")
        reader = easyocr.Reader(languages, gpu=False)
        print("Đã khởi tạo EasyOCR, đang thực hiện OCR...")
        
        # Thực hiện OCR
        result = reader.readtext(image_path)
        
        # Chuyển đổi kết quả về định dạng thống nhất
        text_areas = []
        
        for detection in result:
            bbox = detection[0]  # Bounding box dạng [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            text = detection[1]  # Văn bản
            confidence = detection[2]  # Độ tin cậy
            
            # Chuyển đổi bbox từ 4 điểm thành danh sách tọa độ
            box = [
                [int(bbox[0][0]), int(bbox[0][1])],  # top-left
                [int(bbox[1][0]), int(bbox[1][1])],  # top-right
                [int(bbox[2][0]), int(bbox[2][1])],  # bottom-right
                [int(bbox[3][0]), int(bbox[3][1])]   # bottom-left
            ]
            
            text_areas.append((box, text, confidence))
        
        end_time = time.time()
        print(f"Thời gian phát hiện: {end_time - start_time:.2f} giây")
        print(f"Đã phát hiện {len(text_areas)} vùng văn bản")
        
        return text_areas
    
    except ImportError as e:
        print(f"Lỗi: Thiếu thư viện EasyOCR. Chi tiết: {e}")
        print("Cài đặt: pip install easyocr")
        return []
    except Exception as e:
        print(f"Lỗi khi phát hiện văn bản: {e}")
        return []

def detect_text_areas(image_path, lang='en', use_easyocr=False, use_fallback=False):
    """
    Phát hiện vùng văn bản trong ảnh sử dụng PaddleOCR hoặc EasyOCR.
    
    Args:
        image_path: Đường dẫn đến ảnh
        lang: Ngôn ngữ OCR
        use_easyocr: Sử dụng EasyOCR thay vì PaddleOCR
        use_fallback: Sử dụng phương pháp dự phòng nếu OCR không hoạt động
        
    Returns:
        List các tuples (bounding_box, text, confidence)
    """
    try:
        if use_easyocr:
            return detect_text_with_easyocr(image_path, lang)
        else:
            return detect_text_with_paddleocr(image_path, lang)
    except Exception as e:
        print(f"Lỗi khi phát hiện văn bản: {e}")
        
        # Nếu thử phương pháp đầu tiên không thành công, thử phương pháp còn lại
        if not use_easyocr and not use_fallback:
            print("Thử sử dụng EasyOCR...")
            try:
                return detect_text_with_easyocr(image_path, lang)
            except Exception as e:
                print(f"Cả hai phương pháp OCR đều thất bại: {e}")
        
        # Nếu cả hai phương pháp đều thất bại và cho phép dùng phương pháp dự phòng, tạo vùng giả
        if use_fallback:
            print("Sử dụng phương pháp dự phòng: chia ảnh thành các vùng")
            return generate_fallback_text_areas(image_path)
        
        return []

def generate_fallback_text_areas(image_path):
    """
    Tạo các vùng văn bản giả khi OCR không hoạt động.
    Phương pháp dự phòng này chia ảnh thành lưới các vùng.
    
    Args:
        image_path: Đường dẫn đến ảnh
        
    Returns:
        List các tuples (bounding_box, text, confidence)
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    height, width = image.shape[:2]
    
    # Tạo lưới chia ảnh (3x3, 4x2 hoặc 2x2 tùy thuộc vào kích thước ảnh)
    if width > height * 2:  # Ảnh rộng hơn nhiều so với chiều cao
        rows, cols = 2, 4
    elif width > height:  # Ảnh rộng hơn một chút so với chiều cao
        rows, cols = 2, 3
    elif height > width * 2:  # Ảnh cao hơn nhiều so với chiều rộng
        rows, cols = 4, 2
    else:  # Ảnh gần vuông
        rows, cols = 3, 3
    
    text_areas = []
    cell_height = height // rows
    cell_width = width // cols
    
    # Tạo các vùng bounding box dựa trên lưới
    for row in range(rows):
        for col in range(cols):
            # Tính tọa độ của ô
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = (col + 1) * cell_width
            y2 = (row + 1) * cell_height
            
            # Tạo bounding box
            box = [
                [x1, y1],  # top-left
                [x2, y1],  # top-right
                [x2, y2],  # bottom-right
                [x1, y2]   # bottom-left
            ]
            
            # Tạo khu vực vùng văn bản
            region = image[y1:y2, x1:x2]
            
            # Kiểm tra nếu vùng quá trống (nghĩa là chỉ có một màu)
            if region.size > 0:
                std_dev = np.std(region)
                if std_dev < 10:  # Vùng quá đồng nhất, có thể không có văn bản
                    continue
            
            # Thêm vào danh sách với text rỗng và độ tin cậy thấp
            text_areas.append((box, f"Region_{row}_{col}", 0.5))
    
    print(f"Đã tạo {len(text_areas)} vùng với phương pháp dự phòng")
    return text_areas

def draw_text_areas(image, text_areas, output_path):
    """
    Vẽ các vùng văn bản lên ảnh.
    
    Args:
        image: Ảnh gốc
        text_areas: Danh sách các vùng văn bản
        output_path: Đường dẫn lưu ảnh kết quả
    """
    # Tạo bản sao của ảnh để vẽ lên
    result_image = image.copy()
    
    # Vẽ bounding box cho từng vùng văn bản
    for idx, (box, text, confidence) in enumerate(text_areas):
        # Chuyển box thành dạng points cho hàm vẽ
        points = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        
        # Vẽ bounding box
        cv2.polylines(result_image, [points], True, (0, 255, 0), 2)
        
        # Vẽ số thứ tự
        x, y = box[0]  # Góc trên bên trái
        cv2.putText(result_image, f"#{idx+1}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # In thông tin
        print(f"Vùng #{idx+1}:")
        print(f"  - Tọa độ: {box}")
        print(f"  - Văn bản: {text}")
        print(f"  - Độ tin cậy: {confidence:.4f}")
    
    # Lưu ảnh kết quả
    cv2.imwrite(output_path, result_image)
    print(f"Đã lưu ảnh kết quả vào: {output_path}")
    
    return result_image

def draw_with_paddleocr(image_path, text_areas, output_path):
    """
    Vẽ kết quả sử dụng hàm draw_ocr của PaddleOCR.
    
    Args:
        image_path: Đường dẫn đến ảnh gốc
        text_areas: Danh sách các vùng văn bản
        output_path: Đường dẫn lưu ảnh kết quả
    """
    try:
        from paddleocr import draw_ocr
        
        # Đọc ảnh
        image = cv2.imread(image_path)
        
        # Tìm font
        font_path = find_font()
        
        # Chuẩn bị dữ liệu
        boxes = [area[0] for area in text_areas]
        txts = [area[1] for area in text_areas]
        scores = [area[2] for area in text_areas]
        
        # Vẽ kết quả
        result_image = draw_ocr(image, boxes, txts, scores, font_path=font_path)
        
        # Lưu ảnh
        cv2.imwrite(output_path, result_image)
        print(f"Đã vẽ kết quả sử dụng PaddleOCR và lưu vào: {output_path}")
        
        return result_image
    
    except ImportError:
        print("Lỗi: Thiếu thư viện PaddleOCR để vẽ kết quả")
        return None
    except Exception as e:
        print(f"Lỗi khi vẽ kết quả: {e}")
        return None

def main():
    # Phân tích tham số
    args = parse_args()
    
    # Đọc ảnh
    image = read_image(args.image_path)
    if image is None:
        return False
    
    # Phát hiện vùng văn bản
    text_areas = detect_text_areas(args.image_path, args.lang, args.use_easyocr, args.use_fallback)
    
    if not text_areas:
        print("Không phát hiện được vùng văn bản nào")
        return False
    
    # Lưu thông tin text_areas vào file để sử dụng ở các bước sau
    text_areas_file = os.path.join(os.path.dirname(args.output_path), "text_areas.txt")
    with open(text_areas_file, 'w', encoding='utf-8') as f:
        for idx, (box, text, confidence) in enumerate(text_areas):
            box_str = ",".join([f"{p[0]},{p[1]}" for p in box])
            f.write(f"{idx+1}|{box_str}|{text}|{confidence}\n")
    
    print(f"Đã lưu thông tin vùng văn bản vào: {text_areas_file}")
    
    # Vẽ kết quả
    result_image = draw_text_areas(image, text_areas, args.output_path)
    
    # Vẽ kết quả sử dụng PaddleOCR nếu không sử dụng EasyOCR
    if not args.use_easyocr:
        paddle_output = args.output_path.replace('.jpg', '_paddle.jpg')
        draw_with_paddleocr(args.image_path, text_areas, paddle_output)
    
    # Hiển thị kết quả nếu yêu cầu
    if args.show and result_image is not None:
        cv2.imshow('Text Areas', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return True

if __name__ == "__main__":
    main() 