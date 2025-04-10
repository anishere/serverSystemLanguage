# Công cụ dịch văn bản từ ảnh

Công cụ dịch văn bản từ ảnh sử dụng PaddleOCR và OpenAI GPT-4o-mini để trích xuất và dịch văn bản từ ảnh.

## Quy trình xử lý

1. Đọc ảnh
2. Phát hiện vùng bounding box chứa văn bản (PaddleOCR)
3. Crop vùng bounding box
4. Tô trắng vùng chữ cũ trong ảnh gốc
5. Sử dụng GPT để trích xuất văn bản từ ảnh crop
6. Dịch văn bản trích xuất
7. Vẽ văn bản đã dịch vào vùng crop (căn giữa)
8. Chèn vùng đã xử lý lại vào ảnh gốc

## Yêu cầu

- Python 3.7+
- paddle==2.5.2
- paddleocr>=2.6.0.1
- opencv-python
- numpy
- requests
- Pillow
- python-dotenv
- matplotlib

## Cài đặt

1. Clone repository hoặc tải về:

```bash
git clone <repository_url>
cd image_translator
```

2. Cài đặt các thư viện cần thiết:

```bash
pip install paddle==2.5.2 paddleocr opencv-python numpy requests pillow python-dotenv matplotlib
```

3. Tạo file `.env` trong thư mục gốc và thêm API key của OpenAI:

```
OPENAI_API_KEY="your_openai_api_key"
```

## Cách sử dụng

### Chạy toàn bộ quy trình

```bash
python main.py --image_path "đường_dẫn_đến_ảnh" [--target_lang vi] [--show]
```

Các tham số:

- `--image_path`: (Bắt buộc) Đường dẫn đến ảnh cần dịch
- `--output_dir`: (Tùy chọn) Thư mục lưu kết quả (mặc định: thư mục cùng với ảnh)
- `--source_lang`: (Tùy chọn) Ngôn ngữ nguồn (mặc định: tự động phát hiện)
- `--target_lang`: (Tùy chọn) Ngôn ngữ đích (mặc định: tiếng Việt - 'vi')
- `--api_key`: (Tùy chọn) OpenAI API key (nếu không cung cấp, sẽ lấy từ biến môi trường)
- `--bg_color`: (Tùy chọn) Màu nền cho văn bản dịch (mặc định: white)
- `--text_color`: (Tùy chọn) Màu chữ cho văn bản dịch (mặc định: black)
- `--show`: (Tùy chọn) Hiển thị kết quả

### Chạy từng bước riêng lẻ

Bạn cũng có thể chạy từng bước trong quy trình một cách riêng lẻ để kiểm tra kết quả:

#### Bước 1: Đọc ảnh

```bash
python steps/step1_read_image.py --image_path "đường_dẫn_đến_ảnh" --show
```

#### Bước 2: Phát hiện vùng văn bản

```bash
python steps/step2_detect_text.py --image_path "đường_dẫn_đến_ảnh" --output_path "detected_boxes.jpg" --lang en --show
```

#### Bước 3: Cắt vùng bounding box và tô trắng

```bash
python steps/step3_crop_boxes.py --image_path "đường_dẫn_đến_ảnh" --output_dir "crops" --show
```

#### Bước 5: Trích xuất văn bản từ các ảnh đã cắt

```bash
python steps/step5_extract_text.py --crop_dir "crops" --output_file "extracted_texts.txt"
```

#### Bước 6: Dịch văn bản đã trích xuất

```bash
python steps/step6_translate_text.py --extracted_file "extracted_texts.txt" --output_file "translated_texts.txt" --target_lang vi
```

#### Bước 7: Vẽ văn bản đã dịch vào vùng crop

```bash
python steps/step7_draw_translated_text.py --translated_file "translated_texts.txt" --crop_dir "crops" --output_dir "translated_crops" --show
```

#### Bước 8: Chèn vùng đã xử lý vào ảnh gốc

```bash
python steps/step8_merge_image.py --image_path "đường_dẫn_đến_ảnh" --whitened_image "whitened_image.jpg" --translated_dir "translated_crops" --output_path "final_translated_image.jpg" --show
```

## Ví dụ sử dụng

```bash
python main.py --image_path "C:\Users\acer\Desktop\imagedemo\Screenshot 2025-04-10 145220.png" --target_lang vi --show
```

## Kết quả

Chương trình sẽ tạo một thư mục với cấu trúc như sau:

```
tên_ảnh_translated/
├── crops/                     # Thư mục chứa các vùng đã cắt
│   ├── crop_1.jpg
│   ├── crop_2.jpg
│   └── cropped_info.txt       # Thông tin các vùng đã cắt
├── translated_crops/          # Thư mục chứa các vùng đã dịch
│   ├── translated_1.jpg
│   ├── translated_2.jpg
│   └── translated_info.txt    # Thông tin các vùng đã dịch
├── whitened_image.jpg         # Ảnh gốc đã tô trắng các vùng văn bản
├── detected_boxes.jpg         # Ảnh hiển thị các vùng văn bản đã phát hiện
├── extracted_texts.txt        # Văn bản đã trích xuất
├── translated_texts.txt       # Văn bản đã dịch
├── final_translated_image.jpg # Ảnh kết quả cuối cùng
└── tên_ảnh_gốc.jpg            # Bản sao của ảnh gốc
```

## Lưu ý

- Chất lượng dịch phụ thuộc vào chất lượng ảnh đầu vào và khả năng trích xuất văn bản của PaddleOCR.
- API của OpenAI có giới hạn tốc độ gọi, nên chương trình có thể mất thời gian nếu có nhiều vùng văn bản.
- Để có kết quả tốt nhất, hãy sử dụng ảnh có độ phân giải cao và văn bản rõ ràng.
