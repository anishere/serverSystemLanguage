import cv2
import numpy as np
import os
import json
import time
import torch
import base64
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
import argparse
import matplotlib.pyplot as plt
import easyocr
from sklearn.cluster import DBSCAN

# ===== CONFIGURATION =====
OPENAI_API_KEY = "your_openai_api_key_here"
TARGET_LANGUAGE = "Vietnamese"
BG_OPACITY = 0.9
TEXT_COLOR = (0, 0, 255)  # Blue
BG_COLOR = (255, 255, 255)  # White
OUTLINE_COLOR = (200, 200, 255)  # Light blue
MIN_REGION_HEIGHT = 30
LINE_MERGE_THRESHOLD = 25
MIN_CONFIDENCE = 0.1

class OptimalImageTranslator:
    def __init__(self, api_key, target_lang="Vietnamese"):
        """Initialize the translator with API key and target language."""
        self.api_key = api_key
        self.target_lang = target_lang
        self.client = OpenAI(api_key=api_key)
        
        # Detect available GPU
        self.use_gpu = torch.cuda.is_available()
        print(f"GPU available: {self.use_gpu}")
        
    def translate_image(self, image_path, output_path=None):
        """Main method to translate an image."""
        print(f"\n===== TRANSLATING IMAGE: {image_path} =====")
        start_time = time.time()
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None
            
        # 1. Analyze image with GPT to identify text and language
        analysis = self._analyze_with_gpt(image_path)
        source_language = analysis["source_language"]["name"]
        extracted_text = analysis["text"]
        language_codes = [lang["code"] for lang in analysis["languages"]]
        
        # 2. Translate the text - ONCE ONLY
        translated_text = self._translate_with_gpt(extracted_text, source_language, self.target_lang)
        
        # 3. Process image to find text regions
        line_regions, debug_img = self._detect_text_regions(image, language_codes)
        
        # 4. Create the translated overlay
        result_image = self._create_overlay(image, line_regions, translated_text)
        
        # Display results
        plt.figure(figsize=(18, 10))
        
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Detected Text Regions")
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title(f"Translated to {self.target_lang}")
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Save results if output_path is provided
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"Saved result to: {output_path}")
            
            # Save debug image
            debug_path = os.path.splitext(output_path)[0] + "_debug.jpg"
            cv2.imwrite(debug_path, debug_img)
            
            # Save analysis info
            info_path = os.path.splitext(output_path)[0] + "_info.json"
            analysis_info = {
                "source_language": source_language,
                "target_language": self.target_lang,
                "extracted_text": extracted_text,
                "translated_text": translated_text,
                "processing_time": time.time() - start_time
            }
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_info, f, ensure_ascii=False, indent=2)
        
        print(f"Translation completed in {time.time() - start_time:.2f} seconds")
        return result_image
        
    def _analyze_with_gpt(self, image_path):
        """Analyze image with GPT-4o-mini to identify text and language."""
        print("\n===== ANALYZING IMAGE WITH GPT-4o-mini =====")
        
        # Read the image as bytes
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        # Convert to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a specialized image analysis assistant.
                        Analyze the image and identify:
                        1. What language(s) the text is written in
                        2. The full text content
                        
                        Return your analysis in JSON format:
                        {
                            "languages": [
                                {"name": "language name", "code": "ISO code"}
                            ],
                            "text": "extracted text content",
                            "source_language": {"name": "main language", "code": "ISO code"}
                        }
                        """
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this image and extract the text."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            # Get response content
            content = response.choices[0].message.content
            print("GPT-4o-mini response:", content)
            
            # Parse JSON from response
            try:
                import re
                json_match = re.search(r'```json(.*?)```', content, re.DOTALL)
                
                if json_match:
                    json_content = json_match.group(1).strip()
                else:
                    json_content = content
                    
                # Clean JSON
                json_content = re.sub(r'```json|```', '', json_content)
                
                analysis = json.loads(json_content)
                return analysis
            except Exception as e:
                print(f"Error parsing JSON: {str(e)}")
                # Fallback to heuristic analysis
                if "languages" in content.lower() and "text" in content.lower():
                    langs = []
                    if "english" in content.lower():
                        langs.append({"name": "English", "code": "en"})
                    if "vietnamese" in content.lower():
                        langs.append({"name": "Vietnamese", "code": "vi"})
                    if "japanese" in content.lower():
                        langs.append({"name": "Japanese", "code": "ja"})
                    if "chinese" in content.lower():
                        langs.append({"name": "Chinese", "code": "zh"})
                    
                    return {
                        "languages": langs if langs else [{"name": "English", "code": "en"}],
                        "text": content,
                        "source_language": langs[0] if langs else {"name": "English", "code": "en"}
                    }
                else:
                    return {
                        "languages": [{"name": "English", "code": "en"}],
                        "text": content,
                        "source_language": {"name": "English", "code": "en"}
                    }
                
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            return {
                "languages": [{"name": "English", "code": "en"}],
                "text": "Could not analyze image",
                "source_language": {"name": "English", "code": "en"}
            }
            
    def _translate_with_gpt(self, text, source_lang, target_lang):
        """Translate text with GPT-4o-mini."""
        print(f"\n===== TRANSLATING FROM {source_lang} TO {target_lang} =====")
        
        # Skip translation if source and target are the same
        if source_lang.lower() == target_lang.lower():
            print("Source and target languages are the same - no translation needed")
            return text
            
        try:
            # Call OpenAI API for translation
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a professional translator.
                        Translate the text from {source_lang} to {target_lang}.
                        Return only the translated text, without explanations."""
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                max_tokens=1000
            )
            
            translation = response.choices[0].message.content
            print(f"Original: {text}")
            print(f"Translation: {translation}")
            
            return translation
            
        except Exception as e:
            print(f"Error during translation: {str(e)}")
            return text
            
    def _detect_text_regions(self, image, language_codes):
        """Detect text regions using advanced techniques."""
        print("\n===== DETECTING TEXT REGIONS =====")
        
        # Ensure we have at least one language code
        if not language_codes:
            language_codes = ["en"]
            
        # Remove duplicates
        language_codes = list(set(language_codes))
        print(f"Using languages: {language_codes}")
        
        # Initialize EasyOCR reader
        reader = easyocr.Reader(language_codes, gpu=self.use_gpu)
        
        # Enhance image for handwriting detection
        enhanced_image = self._enhance_image(image)
        
        # Try different detection methods
        ocr_results = []
        
        # First try paragraph mode
        try:
            print("Trying paragraph detection...")
            paragraph_results = reader.readtext(image, paragraph=True)
            if paragraph_results:
                ocr_results = paragraph_results
                print(f"Found {len(ocr_results)} regions with paragraph mode")
        except Exception as e:
            print(f"Error in paragraph detection: {str(e)}")
            
        # If no results, try normal mode
        if not ocr_results:
            try:
                print("Trying standard detection...")
                standard_results = reader.readtext(image)
                if standard_results:
                    ocr_results = standard_results
                    print(f"Found {len(ocr_results)} regions with standard mode")
            except Exception as e:
                print(f"Error in standard detection: {str(e)}")
                
        # If still no results, try enhanced image
        if not ocr_results:
            try:
                print("Trying enhanced image...")
                enhanced_results = reader.readtext(enhanced_image)
                if enhanced_results:
                    ocr_results = enhanced_results
                    print(f"Found {len(ocr_results)} regions with enhanced image")
            except Exception as e:
                print(f"Error in enhanced image detection: {str(e)}")
                
        # Filter out low-confidence results
        filtered_results = []
        for result in ocr_results:
            try:
                if len(result) >= 3 and isinstance(result[2], (int, float)) and result[2] >= MIN_CONFIDENCE:
                    filtered_results.append(result)
            except:
                continue
                
        # If no valid results, create one covering most of the image
        if not filtered_results:
            print("No valid regions found. Creating default region.")
            h, w = image.shape[:2]
            margin = int(min(w, h) * 0.1)
            fallback_bbox = [
                [margin, margin],
                [w-margin, margin],
                [w-margin, h-margin],
                [margin, h-margin]
            ]
            filtered_results = [(fallback_bbox, "", 0.5)]
            
        # Group regions into lines using DBSCAN clustering
        line_regions = self._group_regions_by_cluster(filtered_results)
        
        # Create debug image
        debug_img = image.copy()
        
        # Draw detected lines
        for i, region in enumerate(line_regions):
            points = np.array(region["points"], dtype=np.int32)
            cv2.polylines(debug_img, [points], True, (0, 255, 0), 2)
            cv2.putText(debug_img, f"Line {i+1}", 
                       (int(points[0][0]), int(points[0][1])-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                       
        return line_regions, debug_img
        
    def _enhance_image(self, image):
        """Enhance image for better text detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for handwriting
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
                                      
        # Apply morphology to connect text
        kernel = np.ones((1, 1), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to color
        enhanced = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
        
        return enhanced
        
    def _group_regions_by_cluster(self, regions):
        """Group regions into lines using DBSCAN clustering."""
        if not regions:
            return []
            
        # Extract y-coordinates
        y_coords = []
        for region in regions:
            bbox = region[0]
            # Calculate center y
            y_center = sum(point[1] for point in bbox) / len(bbox)
            y_coords.append([y_center])
            
        # Apply DBSCAN clustering on y-coordinates
        y_coords = np.array(y_coords)
        clustering = DBSCAN(eps=LINE_MERGE_THRESHOLD, min_samples=1).fit(y_coords)
        
        # Group regions by cluster
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(regions[i])
            
        # Create line regions
        line_regions = []
        for cluster_id, cluster_regions in clusters.items():
            # Sort regions in cluster from left to right
            cluster_regions.sort(key=lambda r: min(point[0] for point in r[0]))
            
            # Create a unified region for this line
            line_region = self._create_unified_region(cluster_regions)
            line_regions.append(line_region)
            
        # Sort lines from top to bottom
        line_regions.sort(key=lambda r: r["y_min"])
        
        return line_regions
        
    def _create_unified_region(self, regions):
        """Create a unified region from multiple regions in a line."""
        if not regions:
            return None
            
        # Extract all text
        texts = []
        confidences = []
        all_points = []
        
        for bbox, text, confidence in regions:
            if text:
                texts.append(text)
                confidences.append(confidence)
            all_points.extend(bbox)
            
        # Calculate bounding box that contains all points
        x_min = min(point[0] for point in all_points)
        y_min = min(point[1] for point in all_points)
        x_max = max(point[0] for point in all_points)
        y_max = max(point[1] for point in all_points)
        
        # Ensure minimum height
        height = y_max - y_min
        if height < MIN_REGION_HEIGHT:
            center_y = (y_min + y_max) / 2
            y_min = center_y - MIN_REGION_HEIGHT / 2
            y_max = center_y + MIN_REGION_HEIGHT / 2
            
        # Create rectangle points
        points = [
            [x_min, y_min],  # Top-left
            [x_max, y_min],  # Top-right
            [x_max, y_max],  # Bottom-right
            [x_min, y_max]   # Bottom-left
        ]
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Create unified region
        return {
            "points": points,
            "text": " ".join(texts),
            "confidence": avg_confidence,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "width": x_max - x_min,
            "height": y_max - y_min
        }
        
    def _create_overlay(self, image, line_regions, translated_text):
        """Create an overlay with the translated text."""
        print("\n===== CREATING TRANSLATION OVERLAY =====")
        
        # Create a copy of the original image
        result_image = image.copy()
        
        # Convert to PIL for better font handling
        pil_image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image, 'RGBA')
        
        # Try to load a font
        font = None
        try:
            font_paths = [
                "arial.ttf",
                "seguisym.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/AppleSDGothicNeo.ttc"
            ]
            
            for path in font_paths:
                try:
                    font = ImageFont.truetype(path, 24)
                    print(f"Loaded font: {path}")
                    break
                except:
                    continue
        except:
            pass
            
        if font is None:
            font = ImageFont.load_default()
            print("Using default font")
            
        # Distribute translated text among regions
        text_segments = self._distribute_text(translated_text, len(line_regions))
        
        # Draw text for each region
        for i, (region, text_segment) in enumerate(zip(line_regions, text_segments)):
            # Skip empty text
            if not text_segment.strip():
                continue
                
            # Extract dimensions
            x_min = region["x_min"]
            y_min = region["y_min"]
            x_max = region["x_max"]
            y_max = region["y_max"]
            width = region["width"]
            height = region["height"]
            
            # Calculate font size based on height
            font_size = int(height * 0.6)
            font_size = max(12, min(font_size, 32))
            
            try:
                scaled_font = ImageFont.truetype(font.path, font_size)
            except:
                scaled_font = font
                
            # Draw background
            draw.rectangle(
                [(x_min, y_min), (x_max, y_max)],
                fill=(*BG_COLOR, int(BG_OPACITY * 255)),
                outline=(*OUTLINE_COLOR, 255)
            )
            
            # Wrap text to fit width
            wrapped_text = self._wrap_text(text_segment, scaled_font, width - 10)
            
            # Draw each line
            y_pos = y_min + 5
            for line in wrapped_text.split('\n'):
                # Draw shadow
                draw.text(
                    (x_min + 6, y_pos + 1),
                    line,
                    fill=(100, 100, 200, 200),
                    font=scaled_font
                )
                
                # Draw main text
                draw.text(
                    (x_min + 5, y_pos),
                    line,
                    fill=(*TEXT_COLOR, 255),
                    font=scaled_font
                )
                
                # Move to next line
                if hasattr(scaled_font, 'getbbox'):
                    text_bbox = scaled_font.getbbox(line)
                    line_height = text_bbox[3] - text_bbox[1]
                else:
                    _, line_height = scaled_font.getsize(line)
                    
                y_pos += line_height + 2
                
        # Convert back to OpenCV format
        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return result_image
        
    def _distribute_text(self, text, num_regions):
        """Distribute translated text among regions."""
        if num_regions <= 1:
            return [text]
            
        # First check if text has natural line breaks
        if '\n' in text:
            lines = text.split('\n')
            
            # If we have exactly the right number, use them
            if len(lines) == num_regions:
                return lines
                
            # If too many lines, combine some
            if len(lines) > num_regions:
                result = []
                lines_per_region = len(lines) // num_regions
                
                for i in range(0, num_regions - 1):
                    start = i * lines_per_region
                    end = (i + 1) * lines_per_region
                    result.append('\n'.join(lines[start:end]))
                    
                # Add remaining lines to last region
                result.append('\n'.join(lines[(num_regions-1)*lines_per_region:]))
                return result
                
            # If too few lines, add empty ones
            while len(lines) < num_regions:
                lines.append("")
            return lines
            
        # Try split by sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # If we have exactly the right number, use them
        if len(sentences) == num_regions:
            return sentences
            
        # If too many sentences, combine some
        if len(sentences) > num_regions:
            result = []
            sentences_per_region = len(sentences) // num_regions
            
            for i in range(0, num_regions - 1):
                start = i * sentences_per_region
                end = (i + 1) * sentences_per_region
                result.append(' '.join(sentences[start:end]))
                
            # Add remaining sentences to last region
            result.append(' '.join(sentences[(num_regions-1)*sentences_per_region:]))
            return result
            
        # If too few sentences, try comma splitting
        if len(sentences) < num_regions:
            parts = []
            for sentence in sentences:
                parts.extend(re.split(r'(?<=,)\s+', sentence))
                
            # If still not enough, split by words
            if len(parts) < num_regions:
                words = text.split()
                result = []
                words_per_region = max(1, len(words) // num_regions)
                
                for i in range(0, num_regions):
                    start = i * words_per_region
                    end = min((i + 1) * words_per_region, len(words))
                    if start < len(words):
                        result.append(' '.join(words[start:end]))
                    else:
                        result.append("")
                        
                return result
                
            # Use comma-split parts
            result = []
            parts_per_region = max(1, len(parts) // num_regions)
            
            for i in range(0, num_regions):
                start = i * parts_per_region
                end = min((i + 1) * parts_per_region, len(parts))
                if start < len(parts):
                    result.append(' '.join(parts[start:end]))
                else:
                    result.append("")
                    
            return result
            
        # Add empty strings if needed
        result = sentences.copy()
        while len(result) < num_regions:
            result.append("")
            
        return result
        
    def _wrap_text(self, text, font, max_width):
        """Wrap text to fit within specified width."""
        words = text.split()
        if not words:
            return ""
            
        lines = []
        current_line = []
        
        for word in words:
            # Try adding word to current line
            test_line = " ".join(current_line + [word])
            
            # Check if line gets too wide
            if hasattr(font, 'getbbox'):
                text_bbox = font.getbbox(test_line)
                line_width = text_bbox[2] - text_bbox[0]
            else:
                line_width, _ = font.getsize(test_line)
                
            if line_width <= max_width or not current_line:
                # Word fits, add it
                current_line.append(word)
            else:
                # Word doesn't fit, start new line
                lines.append(" ".join(current_line))
                current_line = [word]
                
        # Add final line
        if current_line:
            lines.append(" ".join(current_line))
            
        return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description='Optimal Image Translator')
    parser.add_argument('--image', required=True, help='Path to the image to translate')
    parser.add_argument('--output', help='Path to save the result image (optional)')
    parser.add_argument('--target', default=TARGET_LANGUAGE, help=f'Target language (default: {TARGET_LANGUAGE})')
    parser.add_argument('--api-key', default=OPENAI_API_KEY, help='OpenAI API key')
    
    args = parser.parse_args()
    
    # Create default output path if not specified
    if not args.output:
        base_name = os.path.basename(args.image)
        name_without_ext = os.path.splitext(base_name)[0]
        args.output = f"{name_without_ext}_translated.jpg"
    
    # Create and run translator
    translator = OptimalImageTranslator(args.api_key, args.target)
    translator.translate_image(args.image, args.output)

if __name__ == "__main__":
    main()