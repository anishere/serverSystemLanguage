import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageEnhance
import logging
import time

# Thiết lập logging để theo dõi quá trình xử lý
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("ocr_log.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def preprocess_image(image_path, resize_factor=1.5):
    """
    Tiền xử lý ảnh để tăng độ chính xác OCR cho chữ viết tay châu Á.
    
    Args:
        image_path: Đường dẫn đến file ảnh
        resize_factor: Hệ số thay đổi kích thước ảnh
    """
    try:
        # Kiểm tra file tồn tại
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")
        
        # Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể tải ảnh: {image_path}")
        
        # Lưu ảnh gốc
        original = img.copy()
        
        # Thay đổi kích thước ảnh (giúp nhận dạng tốt hơn cho chữ nhỏ hoặc lớn)
        if resize_factor > 0:
            height, width = img.shape[:2]
            new_height, new_width = int(height * resize_factor), int(width * resize_factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Tạo bản sao để xử lý
        preprocessed = img.copy()
        
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
        
        # Áp dụng lọc song phương để giữ lại cạnh trong khi loại bỏ nhiễu
        # Đặc biệt quan trọng cho các ký tự châu Á phức tạp
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Áp dụng ngưỡng thích ứng để xử lý các điều kiện ánh sáng khác nhau
        # Rất hữu ích cho chữ viết tay có độ tương phản thấp
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Loại bỏ nhiễu bằng bộ lọc trung vị
        gray = cv2.medianBlur(gray, 3)
        
        # Giãn nở và xói mòn để tăng cường cạnh ký tự
        # Quan trọng cho các nét chữ mỏng trong chữ viết tay châu Á
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)
        gray = cv2.erode(gray, kernel, iterations=1)
        
        logger.info(f"Tiền xử lý ảnh hoàn tất: {image_path}")
        
        return gray, original
    except Exception as e:
        logger.error(f"Lỗi trong quá trình tiền xử lý ảnh: {e}")
        return None, None

def enhance_image_for_asian_text(img):
    """
    Áp dụng cải tiến cụ thể cho văn bản châu Á
    """
    try:
        # Chuyển đổi sang ảnh PIL để thực hiện các thao tác cải tiến
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Tăng cường độ tương phản (chữ châu Á thường cần độ tương phản cao)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.5)  # Tăng độ tương phản 50%
        
        # Tăng cường độ sắc nét (giúp ích cho các ký tự phức tạp)
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.5)  # Tăng độ sắc nét 50%
        
        # Chuyển lại sang định dạng OpenCV
        enhanced_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        return enhanced_img
    except Exception as e:
        logger.error(f"Lỗi trong quá trình cải tiến ảnh: {e}")
        return img

def recognize_text(preprocessed_img, original_img, languages=['en', 'ja', 'ko', 'zh', 'zh_tra'], gpu=True):
    """
    Nhận dạng văn bản trong ảnh đã tiền xử lý sử dụng EasyOCR.
    Hỗ trợ nhiều ngôn ngữ châu Á.
    """
    try:
        start_time = time.time()
        
        # Kiểm tra GPU
        try:
            logger.info(f"Khởi tạo EasyOCR với ngôn ngữ: {languages}, GPU: {gpu}")
            reader = easyocr.Reader(languages, gpu=gpu)
        except Exception as e:
            logger.warning(f"Khởi tạo GPU thất bại: {e}. Chuyển sang CPU.")
            reader = easyocr.Reader(languages, gpu=False)
        
        # Thử với ảnh đã tiền xử lý
        logger.info("Đang chạy OCR trên ảnh đã tiền xử lý")
        result = reader.readtext(
            preprocessed_img,
            detail=1,
            paragraph=True,               # Nhóm văn bản thành đoạn
            contrast_ths=0.1,             # Ngưỡng thấp hơn cho văn bản có độ tương phản thấp
            adjust_contrast=0.5,          # Điều chỉnh độ tương phản
            text_threshold=0.7,           # Ngưỡng cao hơn cho phát hiện văn bản
            width_ths=0.7,                # Ngưỡng cao hơn cho chiều rộng
            height_ths=0.7,               # Ngưỡng cao hơn cho chiều cao
            decoder='greedy',             # Sử dụng bộ giải mã tham lam cho tốc độ
            beamWidth=5,                  # Độ rộng chùm tìm kiếm tăng độ chính xác
            batch_size=4,                 # Kích thước batch phù hợp
            allowlist=None,               # Không giới hạn ký tự
            blocklist=None                # Không chặn ký tự
        )
        
        # Nếu ít kết quả, thử với ảnh được tăng cường đặc biệt cho văn bản châu Á
        if len(result) < 3:
            logger.info("Phát hiện ít kết quả, thử với ảnh được tăng cường cho văn bản châu Á")
            enhanced_img = enhance_image_for_asian_text(original_img)
            result_enhanced = reader.readtext(
                enhanced_img,
                detail=1,
                paragraph=True,
                contrast_ths=0.1,
                adjust_contrast=0.5,
                text_threshold=0.6,        # Ngưỡng thấp hơn để bắt được nhiều văn bản hơn
                width_ths=0.6,             # Ngưỡng thấp hơn cho chiều rộng
                height_ths=0.6,            # Ngưỡng thấp hơn cho chiều cao
                decoder='beamsearch',      # Sử dụng beamsearch cho độ chính xác cao hơn
                beamWidth=10,              # Tăng độ rộng chùm tìm kiếm
                paragraph_threshold=0.3,   # Ngưỡng đoạn văn thấp hơn
                mag_ratio=1.5              # Tỷ lệ phóng đại để xử lý văn bản nhỏ
            )
            
            # Sử dụng kết quả có nhiều phát hiện hơn
            if len(result_enhanced) > len(result):
                logger.info(f"Sử dụng kết quả từ ảnh tăng cường: {len(result_enhanced)} so với {len(result)}")
                result = result_enhanced
        
        elapsed_time = time.time() - start_time
        logger.info(f"OCR hoàn thành trong {elapsed_time:.2f} giây, phát hiện {len(result)} vùng văn bản")
        
        return result, original_img
    except Exception as e:
        logger.error(f"Lỗi trong quá trình nhận dạng văn bản: {e}")
        return None, original_img

def post_process_text(text_result, languages):
    """
    Hậu xử lý văn bản được nhận dạng để cải thiện độ chính xác.
    Áp dụng xử lý đặc thù cho từng ngôn ngữ.
    """
    processed_results = []
    
    for (bbox, text, score) in text_result:
        # Bỏ qua kết quả trống
        if not text.strip():
            continue
        
        # Xử lý cơ bản
        processed_text = text.strip()
        
        # Kiểm tra xem có ngôn ngữ châu Á trong các ngôn ngữ phát hiện không
        asian_languages = ['ja', 'ko', 'zh', 'zh_tra']
        has_asian = any(lang in languages for lang in asian_languages)
        
        if has_asian:
            # Loại bỏ khoảng trắng không cần thiết trong văn bản châu Á
            # Các ngôn ngữ châu Á thường không sử dụng khoảng trắng giữa các ký tự
            if any(ord(c) > 0x3000 for c in processed_text):  # Kiểm tra ký tự châu Á
                processed_text = ''.join(processed_text.split())
            
            # Có thể thêm xử lý cụ thể cho từng ngôn ngữ châu Á ở đây
            # Ví dụ: sửa các lỗi OCR phổ biến cho các ngôn ngữ cụ thể
        
        # Chỉ giữ lại kết quả có độ tin cậy đủ cao
        if score > 0.25:
            processed_results.append((bbox, processed_text, score))
    
    return processed_results

def visualize_results(img, text_result, threshold=0.25, save_path=None):
    """
    Hiển thị kết quả phát hiện văn bản trên ảnh.
    """
    result_img = img.copy()
    
    # Tạo lớp phủ bán trong suốt để hiển thị văn bản tốt hơn
    overlay = result_img.copy()
    
    # Dictionary để lưu văn bản theo vị trí để loại bỏ trùng lặp
    text_positions = {}
    
    for t in text_result:
        bbox, text, score = t
        
        if score > threshold:
            # Chuyển đổi bbox thành tọa độ số nguyên
            top_left = tuple(map(int, bbox[0]))
            top_right = tuple(map(int, bbox[1]))
            bottom_right = tuple(map(int, bbox[2]))
            bottom_left = tuple(map(int, bbox[3]))
            
            # Tính toán tọa độ đa giác
            polygon = np.array([top_left, top_right, bottom_right, bottom_left])
            
            # Tạo vùng nổi bật bán trong suốt cho vùng văn bản
            cv2.fillPoly(overlay, [polygon], (0, 255, 0, 128))
            
            # Vẽ đường viền đa giác
            cv2.polylines(result_img, [polygon], True, (0, 255, 0), 2)
            
            # Lưu văn bản theo vị trí để tránh trùng lặp
            position_key = f"{top_left[0]}_{top_left[1]}"
            if position_key not in text_positions or text_positions[position_key][1] < score:
                text_positions[position_key] = (text, score, top_left)
    
    # Trộn lớp phủ với ảnh gốc
    alpha = 0.3  # Hệ số trong suốt
    result_img = cv2.addWeighted(overlay, alpha, result_img, 1 - alpha, 0)
    
    # Thêm chú thích văn bản
    for (text, score, pos) in text_positions.values():
        # Thêm hình chữ nhật nền để hiển thị văn bản tốt hơn
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Hình chữ nhật nền (lớn hơn văn bản một chút)
        cv2.rectangle(
            result_img, 
            (pos[0], pos[1] - text_height - 5), 
            (pos[0] + text_width, pos[1]), 
            (255, 255, 255), 
            -1
        )
        
        # Thêm văn bản
        cv2.putText(
            result_img, 
            text, 
            (pos[0], pos[1] - 5), 
            font, 
            font_scale, 
            (0, 0, 255), 
            thickness
        )
        
        # In ra console
        logger.info(f"Văn bản: {text}, Độ tin cậy: {score:.2f}")
    
    # Lưu kết quả nếu cung cấp đường dẫn
    if save_path:
        try:
            cv2.imwrite(save_path, result_img)
            logger.info(f"Ảnh kết quả đã được lưu tại: {save_path}")
        except Exception as e:
            logger.error(f"Không thể lưu ảnh kết quả: {e}")
    
    # Hiển thị kết quả
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title("Kết quả phát hiện văn bản")
    plt.axis("off")
    plt.show()
    
    return result_img

def main(image_path, languages=['en', 'ja', 'ko', 'zh', 'zh_tra'], threshold=0.25, resize_factor=1.5, save_results=True):
    """
    Hàm chính để xử lý ảnh và nhận dạng văn bản.
    """
    try:
        logger.info(f"Bắt đầu quy trình OCR cho: {image_path}")
        logger.info(f"Ngôn ngữ: {languages}, Ngưỡng: {threshold}")
        
        # Bước 1: Tiền xử lý ảnh
        preprocessed_img, original_img = preprocess_image(image_path, resize_factor)
        if preprocessed_img is None or original_img is None:
            logger.error("Tiền xử lý thất bại, đang hủy")
            return None
        
        # Bước 2: Nhận dạng văn bản
        text_result, original_img = recognize_text(preprocessed_img, original_img, languages)
        if text_result is None:
            logger.error("Nhận dạng văn bản thất bại, đang hủy")
            return None
        
        # Bước 3: Hậu xử lý văn bản
        processed_results = post_process_text(text_result, languages)
        logger.info(f"Hậu xử lý hoàn tất: {len(processed_results)} vùng văn bản hợp lệ")
        
        # Bước 4: Hiển thị kết quả
        save_path = os.path.splitext(image_path)[0] + "_ocr_result.jpg" if save_results else None
        result_img = visualize_results(original_img, processed_results, threshold, save_path)
        
        # Trả về kết quả đã xử lý để sử dụng thêm nếu cần
        return processed_results
    except Exception as e:
        logger.error(f"Lỗi trong hàm main: {e}")
        return None

if __name__ == "__main__":
    # Đường dẫn ảnh
    image_path = "C:/Users/acer/Pictures/Screenshots/Screenshot 2025-03-09 111006.png"
    
    # Định nghĩa ngôn ngữ cần nhận dạng
    # en: Tiếng Anh, ja: Tiếng Nhật, ko: Tiếng Hàn, zh: Tiếng Trung giản thể, zh_tra: Tiếng Trung phồn thể
    languages = ['en', 'ja', 'ko', 'zh', 'zh_tra']
    
    # Đặt ngưỡng độ tin cậy
    threshold = 0.25
    
    # Đặt hệ số thay đổi kích thước (1.5 = 150% kích thước gốc, giúp cho văn bản nhỏ)
    resize_factor = 1.5
    
    # Chạy hàm main
    results = main(image_path, languages, threshold, resize_factor, save_results=True)
    
    # Bạn có thể xử lý thêm kết quả ở đây nếu cần
    if results:
        # Ví dụ: Xuất kết quả vào file văn bản
        with open("ocr_results.txt", "w", encoding="utf-8") as f:
            for _, text, score in results:
                f.write(f"{text} (Độ tin cậy: {score:.2f})\n")
        logger.info("Kết quả đã được xuất vào ocr_results.txt")