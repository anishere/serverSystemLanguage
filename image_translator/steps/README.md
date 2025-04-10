# Image Translator - Thư mục steps

Đây là thư mục chứa các module thực hiện từng bước xử lý trong quy trình dịch văn bản từ ảnh.

## Cấu trúc quy trình

1. `step1_read_image.py`: Đọc ảnh và hiển thị thông tin ảnh
2. `step2_text_detection.py`: Phát hiện vùng bounding box chứa văn bản (hỗ trợ EasyOCR và PaddleOCR)
   - **Lưu ý**: File này đã thay thế cho 2 file cũ là `step2_detect_text.py` và `step2_easyocr.py`. Nếu bạn vẫn thấy 2 file cũ này tồn tại, hãy xóa chúng.
3. `step3_crop_boxes.py`: Cắt vùng bounding box từ ảnh gốc
4. `step4_preprocess_image.py`: Tiền xử lý ảnh đã cắt (lọc nhiễu, tăng độ tương phản, làm sắc nét)
5. `step5_extract_text.py`: Trích xuất văn bản từ ảnh đã xử lý sử dụng OpenAI GPT-4o-mini
6. `step6_translate_text.py`: Dịch văn bản đã trích xuất sang tiếng Việt
7. `step7_draw_translated_text.py`: Vẽ văn bản đã dịch vào vùng crop
8. `step8_merge_image.py`: Chèn vùng đã xử lý trở lại ảnh gốc

Ngoài ra còn có:

- `detect_language.py`: Module phát hiện ngôn ngữ từ ảnh sử dụng GPT

## Tổng quan

Đây là một quy trình dịch văn bản từ ảnh hoàn chỉnh, bao gồm các bước từ phát hiện vùng văn bản, tiền xử lý ảnh, trích xuất và dịch văn bản, đến tổng hợp lại ảnh kết quả cuối cùng.

Bạn có thể chạy từng bước riêng lẻ hoặc sử dụng file `main.py` ở thư mục gốc để chạy toàn bộ quy trình.
