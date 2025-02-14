import uvicorn
# import subprocess

# Chạy ứng dụng FastAPI
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
    # uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
    # subprocess.run(["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "2"])
