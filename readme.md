# Format-Preserving DOCX Translator

Công cụ dịch tài liệu DOCX với việc bảo toàn định dạng, được tối ưu hóa bằng đa luồng và bộ nhớ đệm.

## Tính năng

- Dịch thuật tài liệu DOCX với việc giữ nguyên định dạng (font, styles, hình ảnh, bảng,...)
- Xử lý đa luồng để tăng hiệu suất
- Bộ nhớ đệm để tránh dịch lại các đoạn văn trùng lặp
- Hỗ trợ nhiều định dạng tài liệu DOCX khác nhau

## Yêu cầu

- Python 3.7+
- OpenAI API key

## Cài đặt

```bash
pip install requests tqdm
```

## Sử dụng

```bash
python maintest.py input.docx --target en
```

Tham số:

- `input.docx`: Đường dẫn tới file DOCX cần dịch
- `--target` hoặc `-t`: Ngôn ngữ đích (mặc định: en)
- `--output` hoặc `-o`: Đường dẫn tới file đầu ra (tùy chọn)
- `--api-key`: OpenAI API key (nếu không có, sẽ lấy từ biến môi trường OPENAI_API_KEY)
- `--model` hoặc `-m`: Mô hình OpenAI (mặc định: gpt-4o-mini)
- `--workers` hoặc `-w`: Số lượng luồng xử lý (mặc định: 4)
- `--paragraph-mode` hoặc `-p`: Kích hoạt chế độ dịch theo đoạn văn (thay vì từng phần tử)

## Cải tiến đặc biệt

Dự án bao gồm các cải tiến đặc biệt để xử lý các loại tài liệu DOCX khác nhau:

1. **Dịch theo đoạn văn**: Gộp các phần tử văn bản trong cùng một đoạn để dịch với ngữ cảnh đầy đủ
2. **Xử lý đặc biệt cho file chỉ có 1 XML**: Tự động phát hiện và áp dụng phương pháp đặc biệt cho các file DOCX đơn giản
3. **Bộ nhớ đệm thông minh**: Sử dụng cơ chế LRU để tối ưu hóa bộ nhớ đệm
4. **Bảo vệ thẻ đặc biệt**: Bảo vệ URLs, emails, và tên miền khỏi bị dịch

## Kiểm tra kết quả

Sử dụng script `check_docx.py` để kiểm tra nội dung của các file DOCX đã dịch:

```bash
python check_docx.py output.docx [<số_đoạn_tối_đa>] [<vị_trí_bắt_đầu>]
```

## Cấu trúc dự án

- `maintest.py`: Script chính để dịch file DOCX
- `docx_translator.py`: Module xử lý trực tiếp file DOCX
- `check_docx.py`: Công cụ kiểm tra nội dung file DOCX

## Ví dụ sử dụng

```bash
# Dịch file mau1.docx sang tiếng Anh
python maintest.py mau1.docx --target en

# Dịch với 8 luồng xử lý để tăng tốc độ
python maintest.py mau1.docx --target en --workers 8

# Dịch và chỉ định đường dẫn đầu ra
python maintest.py mau1.docx -o D:/LuanVan/output_file.docx --target en

# Kiểm tra nội dung file đã dịch
python check_docx.py mau1_en.docx 15
```

### ma hoa code

`pyarmor gen -O encode -i app`

### pyinstaller

`pyinstaller --onefile run.py`

taskkill /F /IM python.exe /T

tạo môi trường ảo
lần đầu python -m venv venv
.\venv\Scripts\activate

pip install whisper
pip install git+https://github.com/openai/whisper.git

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

nvcc --version
nvidia-smi

warning whisper
D:\LuanVan\api_base_public-main\venv\lib\site-packages\whisper\_\_init\_\_.py

checkpoint = torch.load(fp, map_location=device)

checkpoint = torch.load(fp, map_location=device, weights_only=True)

# Cấu hình CORS

app.add*middleware(
CORSMiddleware,
allow_origins=["*"], # Cho phép mọi nguồn (có thể thay "*" bằng một danh sách các domain cụ thể)
allow*credentials=True,
allow_methods=["*"], # Cho phép tất cả các phương thức HTTP (GET, POST, ...)
allow_headers=["*"], # Cho phép tất cả các header
)

pip install pytesseract

pip install easyocr opencv-python pillow

pip install torch torchvision segment-anything

tenserflow
keras

pip install python-docx langdetect requests

pip install PyMuPDF python-docx comtypes

pip install docx2pdf fitz
pip install pdf2docx python-docx weasyprint mammoth

pip install PyMuPDF

python maintest.py sample4.docx --target en

python maintest.py sample4.docx --target en --workers 8

# Cài đặt PaddlePaddle cho CUDA 11.8 (sẽ hoạt động trên driver CUDA 12.8)

pip install paddlepaddle-gpu==2.5.2 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

# Cài đặt lại các gói phụ thuộc cần thiết

pip install opencv-python imgaug pyclipper lmdb tqdm numpy visualdl

# Cài đặt PaddleOCR phiên bản ổn định

pip install paddleocr==2.7.0.3

# Cài đặt Albumentations phiên bản cụ thể (tránh lỗi)

pip install albumentations==1.3.1

pip install -r requirements.txt

đảm bảo Gộp các phần tử <w:t> trong cùng một đoạn (<w:p>) thành một đơn vị dịch để ko mất ngữ cảnh
Đa luồng giữa các file XML khi xử lý file DOCX phức tạp
Đa luồng giữa các đoạn văn trong file document.xml
