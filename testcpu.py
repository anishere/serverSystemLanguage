import os
import psutil

def check_system_resources():
    # Kiểm tra số lượng CPU logic
    cpu_count = os.cpu_count()
    
    # Kiểm tra RAM khả dụng (GB)
    available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    
    print(f"CPU Cores: {cpu_count}")
    print(f"Available Memory: {available_memory:.2f} GB")
    
    # Đề xuất số lượng workers dựa trên CPU
    suggested_workers = min(cpu_count * 2, 30)
    
    return suggested_workers

# Chạy kiểm tra
suggested_workers = check_system_resources()
print(f"Suggested max workers: {suggested_workers}")