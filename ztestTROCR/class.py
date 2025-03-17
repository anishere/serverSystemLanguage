"""
TrOCR với khả năng phát hiện vị trí văn bản và hiển thị bounding box
"""

import sys
import os
import time
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Kiểm tra và cài đặt các thư viện cần thiết
def install_required_packages():
    try:
        import torch
        import transformers
        import easyocr
        import matplotlib
    except ImportError:
        print("Đang cài đặt các thư viện cần thiết...")
        import subprocess
        packages = ["torch", "torchvision", "transformers", "pillow", 
                   "opencv-python", "easyocr", "matplotlib"]
        for package in packages:
            print(f"Đang cài đặt {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("Đã cài đặt xong các thư viện cần thiết!")

# Khi hàm này được gọi, nó sẽ cài đặt các thư viện cần thiết nếu chưa có
install_required_packages()

# Bây giờ nhập các thư viện đã cài đặt
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr

class TrOCRWithBoundingBox:
    def __init__(self, model_name="microsoft/trocr-base-printed", 
                 detection_languages=['en'], gpu=None):
        """
        Khởi tạo các mô hình cần thiết
        
        Args:
            model_name (str): Tên mô hình TrOCR
            detection_languages (list): Danh sách ngôn ngữ cho phát hiện văn bản
            gpu (bool): Sử dụng GPU nếu True, ngược lại None để tự phát hiện
        """
        print(f"Đang khởi tạo phát hiện văn bản với EasyOCR ({detection_languages})...")
        self.reader = easyocr.Reader(detection_languages, gpu=gpu)
        
        print(f"Đang tải mô hình TrOCR {model_name}...")
        start_time = time.time()
        
        # Tải processor và model TrOCR
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # Di chuyển mô hình đến GPU nếu có
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        load_time = time.time() - start_time
        print(f"Đã tải mô hình thành công trong {load_time:.2f} giây. Sử dụng thiết bị: {self.device}")
    
    def detect_and_recognize(self, image_path, min_size=10, save_result=None):
        """
        Phát hiện văn bản và nhận dạng sử dụng TrOCR
        
        Args:
            image_path (str): Đường dẫn hình ảnh
            min_size (int): Kích thước tối thiểu của vùng văn bản (để loại bỏ nhiễu)
            save_result (str): Đường dẫn để lưu hình ảnh kết quả, None nếu không lưu
            
        Returns:
            tuple: (image_with_boxes, text_results)
                   image_with_boxes: Hình ảnh với bounding box
                   text_results: Danh sách kết quả (vị trí, văn bản)
        """
        # Đọc hình ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc hình ảnh từ {image_path}")
        
        # Chuyển đổi màu từ BGR (OpenCV) sang RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Phát hiện vùng văn bản với EasyOCR
        print("Đang phát hiện vùng văn bản...")
        detection_start = time.time()
        detection_results = self.reader.detect(image_rgb, 
                                            min_size=min_size,
                                            text_threshold=0.7,
                                            link_threshold=0.4,
                                            low_text=0.4)
        horizontal_list, free_list = detection_results
        detection_time = time.time() - detection_start
        print(f"Phát hiện xong {len(horizontal_list) + len(free_list)} vùng văn bản trong {detection_time:.2f} giây")
        
        # Kết hợp cả vùng ngang và vùng tự do
        combined_boxes = horizontal_list + free_list
        
        # Lưu kết quả
        image_with_boxes = image_rgb.copy()
        text_results = []
        
        print(f"Đang nhận dạng văn bản trong {len(combined_boxes)} vùng...")
        for i, box in enumerate(combined_boxes):
            # Cắt vùng văn bản từ hình ảnh gốc
            if len(box) == 4:  # horizontal box: x_min, x_max, y_min, y_max
                x_min, x_max, y_min, y_max = box
                box_points = np.array([[x_min, y_min], [x_max, y_min], 
                                      [x_max, y_max], [x_min, y_max]])
            else:  # free box với các điểm
                box_points = np.array(box)
                x_min, y_min = box_points.min(axis=0)
                x_max, y_max = box_points.max(axis=0)
            
            # Mở rộng khung một chút để đảm bảo văn bản không bị cắt
            padding = 5
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(image_rgb.shape[1], x_max + padding)
            y_max = min(image_rgb.shape[0], y_max + padding)
            
            # Cắt vùng văn bản
            roi = image_rgb[int(y_min):int(y_max), int(x_min):int(x_max)]
            
            if roi.size == 0:  # Kiểm tra nếu vùng cắt trống
                continue
                
            # Chuyển đổi sang định dạng PIL Image cho TrOCR
            pil_roi = Image.fromarray(roi)
            
            # Nhận dạng văn bản bằng TrOCR
            try:
                pixel_values = self.processor(images=pil_roi, return_tensors="pt").pixel_values.to(self.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(pixel_values)
                predicted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Loại bỏ các kết quả rỗng hoặc chỉ có khoảng trắng
                if predicted_text.strip():
                    # Lưu kết quả
                    text_results.append({
                        "id": i,
                        "box": box_points.tolist() if isinstance(box_points, np.ndarray) else box,
                        "text": predicted_text,
                        "position": {
                            "x_min": int(x_min),
                            "y_min": int(y_min),
                            "x_max": int(x_max),
                            "y_max": int(y_max),
                            "width": int(x_max - x_min),
                            "height": int(y_max - y_min)
                        }
                    })
                    
                    # Vẽ bounding box
                    cv2.polylines(image_with_boxes, [box_points.astype(np.int32)], 
                                True, (255, 0, 0), 2)
                    # Vẽ text ID
                    cv2.putText(image_with_boxes, str(i), 
                              (int(x_min), int(y_min) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            except Exception as e:
                print(f"Lỗi khi nhận dạng vùng {i}: {str(e)}")
        
        print(f"Đã nhận dạng được {len(text_results)} vùng văn bản có ý nghĩa")
        
        # Lưu hình ảnh kết quả nếu được yêu cầu
        if save_result:
            plt.figure(figsize=(12, 10))
            plt.imshow(image_with_boxes)
            plt.axis('off')
            plt.savefig(save_result, bbox_inches='tight')
            plt.close()
            print(f"Đã lưu hình ảnh kết quả tại: {save_result}")
        
        return image_with_boxes, text_results
    
    def display_results(self, image, text_results):
        """
        Hiển thị kết quả với matplotlib
        
        Args:
            image: Hình ảnh với bounding box
            text_results: Danh sách kết quả (vị trí, văn bản)
        """
        # Hiển thị hình ảnh với bounding box
        plt.figure(figsize=(12, 10))
        plt.imshow(image)
        
        # Hiển thị thông tin văn bản bên cạnh
        for result in text_results:
            box = result["box"]
            text = result["text"]
            position = result["position"]
            
            # Vẽ chú thích
            plt.annotate(f"ID: {result['id']}, Text: {text}", 
                      xy=(position["x_min"], position["y_min"]),
                      xytext=(position["x_min"], position["y_min"] - 15),
                      color='blue', fontsize=8)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def main():
    if len(sys.argv) < 2:
        print("Sử dụng: python trocr_boundingbox.py <đường_dẫn_hình_ảnh> [tên_mô_hình] [đường_dẫn_lưu_kết_quả]")
        print("\nCác mô hình có sẵn:")
        print("  - microsoft/trocr-base-printed: Cho văn bản in (mặc định)")
        print("  - microsoft/trocr-base-handwritten: Cho chữ viết tay")
        print("  - microsoft/trocr-large-printed: Phiên bản lớn hơn cho văn bản in")
        print("  - microsoft/trocr-large-handwritten: Phiên bản lớn hơn cho chữ viết tay")
        sys.exit(1)
    
    # Lấy đường dẫn hình ảnh và các tham số khác
    image_path = sys.argv[1]
    
    model_name = "microsoft/trocr-base-printed"  # Mặc định
    if len(sys.argv) >= 3:
        model_name = sys.argv[2]
    
    save_path = None
    if len(sys.argv) >= 4:
        save_path = sys.argv[3]
    
    # Khởi tạo mô hình
    ocr_detector = TrOCRWithBoundingBox(model_name=model_name)
    
    # Phát hiện và nhận dạng văn bản
    image_with_boxes, text_results = ocr_detector.detect_and_recognize(
        image_path, save_result=save_path)
    
    # Hiển thị kết quả
    ocr_detector.display_results(image_with_boxes, text_results)
    
    # In danh sách vị trí và văn bản
    print("\n" + "="*50)
    print("DANH SÁCH KẾT QUẢ NHẬN DẠNG:")
    print("="*50)
    for result in text_results:
        print(f"ID: {result['id']}")
        print(f"Vị trí: {result['position']}")
        print(f"Văn bản: {result['text']}")
        print("-"*30)

if __name__ == "__main__":
    main()