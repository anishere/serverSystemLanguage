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

# Cấu hình CORS

app.add_middleware(
CORSMiddleware,
allow_origins=["*"], # Cho phép mọi nguồn (có thể thay "_" bằng một danh sách các domain cụ thể)
allow_credentials=True,
allow_methods=["_"], # Cho phép tất cả các phương thức HTTP (GET, POST, ...)
allow_headers=["*"], # Cho phép tất cả các header
)
