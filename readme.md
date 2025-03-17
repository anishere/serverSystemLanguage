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
