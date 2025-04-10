#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nhận dạng ngôn ngữ từ ảnh sử dụng OpenAI GPT-4o-mini
"""

import os
import argparse
import base64
import requests
from dotenv import load_dotenv

# Tải biến môi trường
load_dotenv()

def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Nhận dạng ngôn ngữ từ ảnh')
    parser.add_argument('--image_path', required=True, help='Đường dẫn đến ảnh đầu vào')
    parser.add_argument('--api_key', help='OpenAI API key (nếu không cung cấp, sẽ lấy từ biến môi trường)')
    return parser.parse_args()

def detect_language_from_image(image_path, api_key):
    """
    Nhận dạng ngôn ngữ từ ảnh sử dụng GPT-4o-mini.
    
    Args:
        image_path: Đường dẫn đến ảnh
        api_key: OpenAI API key
        
    Returns:
        tuple: (language_code, language_name) - mã ngôn ngữ ISO 639-1 và tên đầy đủ
    """
    print(f"Đang nhận dạng ngôn ngữ từ ảnh: {image_path}")
    
    try:
        # Mã hóa ảnh thành Base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Chuẩn bị payload
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = """
        Identify the primary language used in this image.
        
        Requirements:
        1. Look at all text elements in the image
        2. Determine the main language used
        3. Return ONLY the language code (2-letter ISO 639-1 code) and full language name in JSON format
        4. Use the format: {"code": "xx", "name": "Language Name"}
        5. For example: {"code": "en", "name": "English"} or {"code": "vi", "name": "Vietnamese"}
        
        Common language codes:
        - English: en
        - Vietnamese: vi
        - Chinese: zh
        - Japanese: ja
        - Korean: ko
        - French: fr
        - German: de
        - Spanish: es
        - Russian: ru
        
        Do not include any explanation or additional text, ONLY return the JSON object.
        """
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "max_tokens": 150
        }
        
        # Gọi API
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        result = response.json()
        
        # Lấy phản hồi JSON
        response_content = result['choices'][0]['message']['content']
        
        # Parse JSON
        import json
        language_info = json.loads(response_content)
        
        # Lấy mã ngôn ngữ và tên đầy đủ
        language_code = language_info.get('code', 'en')
        language_name = language_info.get('name', 'English')
        
        print(f"Ngôn ngữ được nhận dạng: {language_name} ({language_code})")
        
        return language_code, language_name
    
    except Exception as e:
        print(f"Lỗi khi nhận dạng ngôn ngữ: {e}")
        # Mặc định là tiếng Anh nếu có lỗi
        return "en", "English"

def main():
    # Phân tích tham số
    args = parse_args()
    
    # Lấy API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Lỗi: Không tìm thấy OpenAI API key")
        return False
    
    # Kiểm tra file ảnh đầu vào
    if not os.path.exists(args.image_path):
        print(f"Lỗi: Không tìm thấy ảnh {args.image_path}")
        return False
    
    # Nhận dạng ngôn ngữ
    language_code, language_name = detect_language_from_image(args.image_path, api_key)
    
    print(f"Kết quả nhận dạng ngôn ngữ: {language_name} ({language_code})")
    
    return language_code, language_name

if __name__ == "__main__":
    main() 