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
