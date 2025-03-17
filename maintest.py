#!/usr/bin/env python3
"""
Format-Preserving Translator (Optimized Test)
- Chạy thử nghiệm translator với đa luồng và tối ưu hóa hiệu suất
"""

import os
import time
import argparse
from pathlib import Path
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import module translator đã tối ưu
from docx_translator import DocxTranslator

def setup_openai_api(api_key, target_lang="en", model="gpt-4o-mini", temperature=0.3, request_timeout=60):
    """
    Thiết lập hàm gọi API OpenAI để dịch văn bản
    
    Returns:
        callable: Hàm dịch văn bản sử dụng OpenAI API
    """
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    api_url = "https://api.openai.com/v1/chat/completions"
    
    def translate_text(text):
        """
        Dịch văn bản sử dụng OpenAI API
        """
        if not text or not text.strip():
            return text
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # Sử dụng prompt chuyên nghiệp
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional translator. Translate the given text "
                        f"to {target_lang} accurately and naturally. "
                        "Maintain the original style, formatting, and tone. "
                        "Return ONLY the translated text without explanations or notes."
                    )
                },
                {
                    "role": "user",
                    "content": f"Translate this text to {target_lang}:\n\n{text}"
                }
            ]
            
            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            
            response = session.post(
                api_url,
                headers=headers,
                data=json.dumps(data),
                timeout=request_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                translated_text = result["choices"][0]["message"]["content"].strip()
                
                # Kiểm tra nếu văn bản trả về là thông báo lỗi
                error_messages = [
                    "It seems that there is no text provided",
                    "Please provide the text you would like",
                    "There is no text to translate",
                    "No content to translate"
                ]
                
                if any(error_msg in translated_text for error_msg in error_messages):
                    return text
                
                return translated_text
            else:
                print(f"Lỗi API: {response.status_code}")
                if response.status_code == 429:
                    print("Rate limit exceeded. Đợi và thử lại...")
                    time.sleep(20)  # Đợi 20 giây trước khi thử lại
                    return translate_text(text)  # Thử lại
                
                return text
                
        except requests.exceptions.Timeout:
            print(f"Timeout khi gọi API OpenAI sau {request_timeout}s")
            return text
        except requests.exceptions.RequestException as e:
            print(f"Lỗi kết nối khi gọi API OpenAI: {e}")
            return text
        except Exception as e:
            print(f"Lỗi không xác định khi gọi API OpenAI: {e}")
            return text
    
    return translate_text

def main():
    """Hàm main để xử lý các tham số dòng lệnh và thực hiện dịch file."""
    parser = argparse.ArgumentParser(
        description="Dịch DOCX với OpenAI API - phiên bản tối ưu đa luồng",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_file", help="Đường dẫn đến file DOCX cần dịch")
    parser.add_argument("--output", "-o", help="Đường dẫn đến file output (tùy chọn)")
    parser.add_argument("--target", "-t", default="en", help="Mã ngôn ngữ đích (mặc định: en)")
    parser.add_argument("--api-key", help="OpenAI API key (nếu không cung cấp, sẽ tìm từ biến môi trường OPENAI_API_KEY)")
    parser.add_argument("--model", "-m", default="gpt-4o-mini", help="Mô hình OpenAI (mặc định: gpt-4o-mini)")
    parser.add_argument("--temperature", "-temp", type=float, default=0.3, help="Độ sáng tạo của mô hình (0.0-1.0)")
    parser.add_argument("--timeout", type=int, default=60, help="Thời gian chờ tối đa cho mỗi API request (giây)")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Số lượng worker cho đa luồng")
    
    args = parser.parse_args()
    
    try:
        # Lấy API key từ tham số hoặc biến môi trường
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key không được cung cấp. Vui lòng cung cấp qua tham số --api-key hoặc "
                "biến môi trường OPENAI_API_KEY"
            )
        
        # Thiết lập hàm dịch sử dụng OpenAI API
        translate_func = setup_openai_api(
            api_key=api_key,
            target_lang=args.target,
            model=args.model,
            temperature=args.temperature,
            request_timeout=args.timeout
        )
        
        # Xác định output_path nếu không được cung cấp
        input_path = Path(args.input_file)
        if not args.output:
            output_path = input_path.with_stem(f"{input_path.stem}_{args.target}")
        else:
            output_path = Path(args.output)
        
        # Đảm bảo thư mục đích tồn tại
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo DocxTranslator với cấu hình đa luồng
        translator = DocxTranslator(
            translate_func=translate_func,
            max_workers=args.workers
        )
        
        print(f"Khởi tạo Translator với {args.workers} luồng")
        
        # Đo thời gian thực hiện
        start_time = time.time()
        
        # Dịch file
        output_file = translator.translate_docx_complete(args.input_file, output_path)
        
        # Hiển thị thông tin kết quả
        elapsed_time = time.time() - start_time
        if output_file:
            print(f"Đã hoàn thành trong {elapsed_time:.2f} giây!")
            print(f"File kết quả: {output_file}")
        else:
            print(f"Quá trình dịch không thành công sau {elapsed_time:.2f} giây.")
    
    except Exception as e:
        print(f"Lỗi: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)