#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bước 6: Dịch văn bản đã trích xuất sang tiếng Việt sử dụng OpenAI GPT-4o-mini
"""

import os
import argparse
import requests
import json
import time
from dotenv import load_dotenv

# Tải biến môi trường
load_dotenv()

def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Dịch văn bản đã trích xuất')
    parser.add_argument('--extracted_file', required=True, help='File chứa văn bản đã trích xuất')
    parser.add_argument('--output_file', default='translated_texts.txt', help='File lưu văn bản đã dịch')
    parser.add_argument('--target_lang', default='vi', help='Ngôn ngữ đích để dịch (mặc định: tiếng Việt)')
    parser.add_argument('--api_key', help='OpenAI API key (nếu không cung cấp, sẽ lấy từ biến môi trường)')
    return parser.parse_args()

def load_extracted_texts(extracted_file):
    """
    Đọc văn bản đã trích xuất từ file.
    
    Args:
        extracted_file: Đường dẫn đến file chứa văn bản đã trích xuất
        
    Returns:
        List thông tin các ảnh và văn bản đã trích xuất
    """
    extracted_texts = []
    
    try:
        with open(extracted_file, 'r', encoding='utf-8') as f:
            # Bỏ qua dòng header
            header = f.readline()
            
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    index = int(parts[0])
                    path = parts[1]
                    original_text = parts[2]
                    extracted_text = parts[3]
                    
                    extracted_texts.append({
                        'index': index,
                        'path': path,
                        'original_text': original_text,
                        'extracted_text': extracted_text
                    })
        
        print(f"Đã đọc {len(extracted_texts)} văn bản đã trích xuất từ {extracted_file}")
    
    except Exception as e:
        print(f"Lỗi khi đọc file văn bản đã trích xuất: {e}")
    
    return extracted_texts

def translate_text(text, target_lang, api_key):
    """
    Dịch văn bản sang ngôn ngữ đích sử dụng GPT-4o-mini.
    
    Args:
        text: Văn bản cần dịch
        target_lang: Ngôn ngữ đích
        api_key: OpenAI API key
        
    Returns:
        str: Văn bản đã dịch
    """
    if not text.strip():
        return ""
    
    print(f"Dịch văn bản: {text}")
    
    try:
        # Xác định ngôn ngữ đích với tên đầy đủ
        language_map = {
            'vi': 'Vietnamese',
            'en': 'English',
            'fr': 'French',
            'de': 'German',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'es': 'Spanish',
            'ru': 'Russian',
            'it': 'Italian',
            'pt': 'Portuguese',
            'nl': 'Dutch',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'th': 'Thai',
            'id': 'Indonesian',
            'ms': 'Malay'
        }
        
        # Nếu đã truyền tên đầy đủ, sử dụng trực tiếp, nếu không thì tra trong bảng
        if target_lang in language_map.values():
            language_name = target_lang
        else:
            language_name = language_map.get(target_lang.lower(), 'English')
            # Nếu không tìm thấy trong map, giả định target_lang là tên đầy đủ
            if target_lang not in language_map:
                language_name = target_lang
        
        # Chuẩn bị payload
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Sử dụng prompt rõ ràng hơn với ngôn ngữ đầy đủ
        prompt = f"""
        Translate the following text to {language_name}.
        
        Guidelines:
        - Translate ALL text accurately
        - Maintain original formatting and structure
        - Preserve numbers, dates, and special characters
        - For single words or short phrases, provide direct translation only
        - Do not add explanations or comments
        - Return ONLY the translated text
        """
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        # Gọi API
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        result = response.json()
        
        # Lấy văn bản đã dịch
        translated_text = result['choices'][0]['message']['content'].strip()
        
        print(f"Văn bản đã dịch: {translated_text}")
        
        return translated_text
    
    except Exception as e:
        print(f"Lỗi khi dịch văn bản: {e}")
        return text  # Trả về văn bản gốc nếu có lỗi

def process_translations(extracted_texts, target_lang, api_key, output_file):
    """
    Xử lý dịch văn bản cho tất cả các mục.
    
    Args:
        extracted_texts: Danh sách thông tin các văn bản đã trích xuất
        target_lang: Ngôn ngữ đích
        api_key: OpenAI API key
        output_file: Đường dẫn file đầu ra
    
    Returns:
        List thông tin các văn bản đã dịch
    """
    translated_texts = []
    
    # Mở file để lưu kết quả
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("INDEX|PATH|EXTRACTED_TEXT|TRANSLATED_TEXT\n")
        
        # Xử lý từng mục
        for item in extracted_texts:
            print(f"\nXử lý văn bản #{item['index']}: {item['path']}")
            
            # Dịch văn bản
            translated_text = translate_text(item['extracted_text'], target_lang, api_key)
            
            # Thêm văn bản đã dịch vào thông tin
            item['translated_text'] = translated_text
            
            # Ghi kết quả vào file
            f.write(f"{item['index']}|{item['path']}|{item['extracted_text']}|{translated_text}\n")
            
            # Thêm vào danh sách đã xử lý
            translated_texts.append(item)
            
            # Chờ một chút để tránh giới hạn API
            time.sleep(0.5)
    
    print(f"Đã lưu văn bản đã dịch vào: {output_file}")
    
    return translated_texts

def main():
    # Phân tích tham số
    args = parse_args()
    
    # Lấy API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Lỗi: Không tìm thấy OpenAI API key")
        return False
    
    # Kiểm tra file văn bản đã trích xuất
    if not os.path.exists(args.extracted_file):
        print(f"Lỗi: Không tìm thấy file {args.extracted_file}")
        return False
    
    # Đọc văn bản đã trích xuất
    extracted_texts = load_extracted_texts(args.extracted_file)
    
    if not extracted_texts:
        print("Không có văn bản nào để dịch")
        return False
    
    # Xử lý dịch văn bản
    output_file = os.path.join(os.path.dirname(args.extracted_file), args.output_file)
    translated_texts = process_translations(extracted_texts, args.target_lang, api_key, output_file)
    
    print(f"\nĐã dịch {len(translated_texts)} văn bản")
    
    return True

if __name__ == "__main__":
    main() 