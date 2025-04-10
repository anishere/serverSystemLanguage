#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bước 7: Vẽ văn bản đã dịch vào vùng bounding box
"""

import os
import cv2
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Vẽ văn bản đã dịch vào vùng crop')
    parser.add_argument('--translated_file', required=True, help='File chứa văn bản đã dịch')
    parser.add_argument('--crop_dir', help='Thư mục chứa các ảnh đã cắt')
    parser.add_argument('--output_dir', default='translated_crops', help='Thư mục lưu các ảnh đã vẽ văn bản')
    parser.add_argument('--bg_color', default='white', help='Màu nền cho vùng văn bản (tên màu hoặc mã hex: white, black, #FFFFFF, ...)')
    parser.add_argument('--text_color', default='black', help='Màu chữ (tên màu hoặc mã hex: black, red, #000000, ...)')
    parser.add_argument('--font_size', type=int, default=0, help='Kích thước font (0: tự động)')
    parser.add_argument('--show', action='store_true', help='Hiển thị ảnh')
    return parser.parse_args()

def load_translated_texts(translated_file):
    """
    Đọc văn bản đã dịch từ file.
    
    Args:
        translated_file: Đường dẫn đến file chứa văn bản đã dịch
        
    Returns:
        List thông tin các ảnh và văn bản đã dịch
    """
    translated_texts = []
    
    try:
        with open(translated_file, 'r', encoding='utf-8') as f:
            # Bỏ qua dòng header
            header = f.readline()
            
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    index = int(parts[0])
                    path = parts[1]
                    extracted_text = parts[2]
                    translated_text = parts[3]
                    
                    translated_texts.append({
                        'index': index,
                        'path': path,
                        'extracted_text': extracted_text,
                        'translated_text': translated_text
                    })
        
        print(f"Đã đọc {len(translated_texts)} văn bản đã dịch từ {translated_file}")
    
    except Exception as e:
        print(f"Lỗi khi đọc file văn bản đã dịch: {e}")
    
    return translated_texts

def find_font():
    """
    Tìm font phù hợp để hiển thị tiếng Việt.
    
    Returns:
        str: Đường dẫn đến font
    """
    # Danh sách font ưu tiên
    font_candidates = [
        'Arial Unicode MS',
        'Arial',
        'DejaVu Sans',
        'Segoe UI',
        'Tahoma',
        'Times New Roman',
        'Calibri',
        'Microsoft Sans Serif'
    ]
    
    # Kiểm tra các font có sẵn trong hệ thống
    system_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    for font_name in font_candidates:
        for font_path in system_fonts:
            if font_name.lower() in os.path.basename(font_path).lower():
                print(f"Đã tìm thấy font: {font_path}")
                return font_path
    
    # Nếu không tìm thấy, kiểm tra các đường dẫn font cụ thể
    possible_font_paths = [
        './fonts/arial.ttf',
        'C:/Windows/Fonts/Arial.ttf',
        'C:/Windows/Fonts/calibri.ttf',
        'C:/Windows/Fonts/tahoma.ttf',
        'C:/Windows/Fonts/segoeui.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/Library/Fonts/Arial Unicode.ttf',
    ]
    
    for path in possible_font_paths:
        if os.path.exists(path):
            print(f"Đã tìm thấy font: {path}")
            return path
    
    # Fallback: Sử dụng font mặc định của matplotlib
    print("Không tìm thấy font tiếng Việt, sử dụng font mặc định")
    return fm.findfont(fm.FontProperties())

def get_text_dimensions(text, font_path, font_size):
    """
    Tính toán kích thước của đoạn văn bản với font và kích thước cho trước.
    
    Args:
        text: Văn bản cần tính toán
        font_path: Đường dẫn đến font
        font_size: Kích thước font
        
    Returns:
        tuple: (chiều rộng, chiều cao)
    """
    try:
        # Sử dụng Pillow để tính kích thước
        font = ImageFont.truetype(font_path, font_size)
        
        # Tạo ảnh tạm để đo kích thước
        img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(img)
        
        # Đo kích thước văn bản
        text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
        
        return text_width, text_height
    except Exception as e:
        print(f"Lỗi khi tính kích thước văn bản: {e}")
        # Ước tính thô: mỗi ký tự rộng khoảng font_size * 0.6, cao font_size * 1.2
        return len(text) * font_size * 0.6, font_size * 1.2

def find_optimal_font_size(text, max_width, max_height, font_path, min_size=10, max_size=100):
    """
    Tìm kích thước font tối ưu để vừa với kích thước cho trước.
    
    Args:
        text: Văn bản cần vẽ
        max_width: Chiều rộng tối đa
        max_height: Chiều cao tối đa
        font_path: Đường dẫn đến font
        min_size: Kích thước font tối thiểu
        max_size: Kích thước font tối đa
        
    Returns:
        int: Kích thước font tối ưu
    """
    # Nếu không có văn bản, trả về kích thước tối thiểu
    if not text.strip():
        return min_size
    
    # Tìm kiếm nhị phân để tìm kích thước tối ưu
    low, high = min_size, max_size
    optimal_size = min_size
    
    while low <= high:
        mid = (low + high) // 2
        text_width, text_height = get_text_dimensions(text, font_path, mid)
        
        if text_width <= max_width * 0.9 and text_height <= max_height * 0.9:
            # Kích thước vẫn còn vừa, thử tăng lên
            optimal_size = mid
            low = mid + 1
        else:
            # Kích thước quá lớn, giảm xuống
            high = mid - 1
    
    return optimal_size

def hex_to_rgb(hex_color):
    """
    Chuyển đổi màu dạng hex sang RGB.
    
    Args:
        hex_color: Mã màu hex (ví dụ: #FFFFFF)
        
    Returns:
        tuple: (R, G, B)
    """
    # Loại bỏ ký tự # nếu có
    hex_color = hex_color.lstrip('#')
    
    # Chuyển đổi sang RGB
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_color(color_str):
    """
    Chuyển đổi chuỗi màu thành tuple RGB.
    
    Args:
        color_str: Tên màu hoặc mã hex
        
    Returns:
        tuple: (R, G, B)
    """
    # Danh sách màu cơ bản
    color_map = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'gray': (128, 128, 128)
    }
    
    # Nếu là tên màu trong danh sách
    if color_str.lower() in color_map:
        return color_map[color_str.lower()]
    
    # Nếu là mã hex
    if color_str.startswith('#') and len(color_str) in (4, 7):
        try:
            return hex_to_rgb(color_str)
        except ValueError:
            pass
    
    # Mặc định là đen
    print(f"Không nhận dạng được màu '{color_str}', sử dụng màu đen")
    return (0, 0, 0)

def draw_text_on_image(image, text, font_path, font_size=20, bg_color='white', text_color='black'):
    """
    Vẽ văn bản lên ảnh với nền màu và căn giữa.
    
    Args:
        image: Ảnh gốc (numpy array)
        text: Văn bản cần vẽ
        font_path: Đường dẫn đến font
        font_size: Kích thước font
        bg_color: Màu nền
        text_color: Màu chữ
        
    Returns:
        numpy array: Ảnh đã vẽ văn bản
    """
    if not text.strip():
        return image.copy()  # Trả về bản sao của ảnh gốc nếu không có text
    
    # Chuyển đổi màu sang tuple RGB
    bg_color_rgb = get_color(bg_color)
    text_color_rgb = get_color(text_color)
    
    # Kích thước ảnh
    height, width = image.shape[:2]
    
    # Nếu font_size là 0, tự động tính kích thước font
    if font_size <= 0:
        font_size = find_optimal_font_size(text, width, height, font_path)
    
    # Tạo ảnh mới với cùng kích thước và màu nền
    result_image = np.ones((height, width, 3), dtype=np.uint8)
    result_image[:] = bg_color_rgb
    
    try:
        # Chuyển đổi sang ảnh Pillow
        pil_img = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Tạo font
        font = ImageFont.truetype(font_path, font_size)
        
        # Tính kích thước văn bản để căn giữa
        text_width, text_height = get_text_dimensions(text, font_path, font_size)
        
        # Tính toán vị trí để căn giữa
        x = max(0, (width - text_width) // 2)
        y = max(0, (height - text_height) // 2)
        
        # Vẽ văn bản
        draw.text((x, y), text, font=font, fill=text_color_rgb)
        
        # Chuyển đổi ngược lại sang OpenCV
        result_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        print(f"Lỗi khi vẽ văn bản: {e}")
        # Fallback: Sử dụng OpenCV để vẽ văn bản (không hỗ trợ Unicode đầy đủ)
        
        # Xác định vị trí văn bản để căn giữa
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size/30, 1)[0]
        text_x = max(0, (width - text_size[0]) // 2)
        text_y = max(0, (height + text_size[1]) // 2)
        
        # Vẽ văn bản
        cv2.putText(result_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_size/30, text_color_rgb, 1, cv2.LINE_AA)
    
    # Đảm bảo ảnh đầu ra có đúng kích thước
    if result_image.shape[:2] != (height, width):
        result_image = cv2.resize(result_image, (width, height))
    
    return result_image

def process_images(translated_texts, crop_dir, output_dir, bg_color, text_color, font_size):
    """
    Xử lý các ảnh để vẽ văn bản đã dịch.
    
    Args:
        translated_texts: Danh sách thông tin các văn bản đã dịch
        crop_dir: Thư mục chứa các ảnh đã cắt
        output_dir: Thư mục lưu các ảnh đã vẽ văn bản
        bg_color: Màu nền
        text_color: Màu chữ
        font_size: Kích thước font
        
    Returns:
        List thông tin các ảnh đã vẽ văn bản
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Tìm font phù hợp
    font_path = find_font()
    
    # Danh sách lưu thông tin các ảnh đã xử lý
    processed_images = []
    
    # Xử lý từng ảnh
    for item in translated_texts:
        print(f"\nXử lý ảnh #{item['index']}: {item['path']}")
        
        # Đường dẫn ảnh gốc
        crop_path = item['path']
        if not os.path.exists(crop_path) and crop_dir:
            # Thử tìm ảnh trong thư mục crop_dir
            basename = os.path.basename(crop_path)
            crop_path = os.path.join(crop_dir, basename)
        
        if not os.path.exists(crop_path):
            print(f"Không tìm thấy ảnh {crop_path}, bỏ qua")
            continue
        
        # Đọc ảnh
        crop_img = cv2.imread(crop_path)
        if crop_img is None:
            print(f"Không thể đọc ảnh {crop_path}, bỏ qua")
            continue
        
        # Vẽ văn bản đã dịch lên ảnh
        result_img = draw_text_on_image(
            crop_img, 
            item['translated_text'], 
            font_path,
            font_size,
            bg_color,
            text_color
        )
        
        # Tạo đường dẫn đầu ra
        output_path = os.path.join(output_dir, f"translated_{item['index']}.jpg")
        
        # Lưu ảnh
        cv2.imwrite(output_path, result_img)
        print(f"Đã lưu ảnh có văn bản đã dịch vào: {output_path}")
        
        # Cập nhật thông tin
        item['output_path'] = output_path
        processed_images.append(item)
    
    # Lưu thông tin các ảnh đã xử lý
    output_info_path = os.path.join(output_dir, "translated_info.txt")
    with open(output_info_path, 'w', encoding='utf-8') as f:
        f.write("INDEX|CROP_PATH|OUTPUT_PATH|EXTRACTED_TEXT|TRANSLATED_TEXT\n")
        for item in processed_images:
            f.write(f"{item['index']}|{item['path']}|{item['output_path']}|{item['extracted_text']}|{item['translated_text']}\n")
    
    print(f"Đã lưu thông tin các ảnh đã xử lý vào: {output_info_path}")
    
    return processed_images

def main():
    # Phân tích tham số
    args = parse_args()
    
    # Kiểm tra file văn bản đã dịch
    if not os.path.exists(args.translated_file):
        print(f"Lỗi: Không tìm thấy file {args.translated_file}")
        return False
    
    # Đọc văn bản đã dịch
    translated_texts = load_translated_texts(args.translated_file)
    
    if not translated_texts:
        print("Không có văn bản nào để xử lý")
        return False
    
    # Xử lý các ảnh
    crop_dir = args.crop_dir or os.path.dirname(args.translated_file)
    output_dir = os.path.join(os.path.dirname(args.translated_file), args.output_dir)
    
    processed_images = process_images(
        translated_texts,
        crop_dir,
        output_dir,
        args.bg_color,
        args.text_color,
        args.font_size
    )
    
    # Hiển thị kết quả nếu yêu cầu
    if args.show and processed_images:
        print("\nHiển thị kết quả (nhấn phím bất kỳ để chuyển ảnh, ESC để thoát)...")
        
        for item in processed_images:
            # Đọc ảnh gốc và ảnh đã xử lý
            crop_img = cv2.imread(item['path'])
            output_img = cv2.imread(item['output_path'])
            
            if crop_img is not None and output_img is not None:
                # Hiển thị ảnh gốc
                cv2.imshow(f"Crop #{item['index']} - Original", crop_img)
                
                # Hiển thị ảnh đã xử lý
                cv2.imshow(f"Crop #{item['index']} - Translated", output_img)
                
                # Đợi phím nhấn
                key = cv2.waitKey(0)
                if key == 27:  # ESC
                    break
                
                # Đóng cửa sổ
                cv2.destroyAllWindows()
    
    print(f"\nĐã xử lý {len(processed_images)} ảnh")
    
    return True

if __name__ == "__main__":
    main() 