from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from app.models.analyzeAI import AnalyzeRequest, AnalyzeResponse, LanguageFragment
from app.config import settings
import openai
from app.security.security import get_api_key
import json
import re
import asyncio
from typing import List, Dict, Any, Tuple, Set
import time
import hashlib
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multilingual_analyzer")

# Cấu hình client OpenAI
client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

# Bỏ việc sử dụng Redis và NLTK, dùng các phương pháp đơn giản hơn
in_memory_cache = {}

# Kiểm tra nếu có thể sử dụng fasttext (tùy chọn)
USE_FASTTEXT = False
try:
    import fasttext
    # Để an toàn, chỉ khi nào load model thành công mới bật flag
    try:
        # Đặt điều kiện hoặc biến môi trường để bỏ qua nếu không cần
        if hasattr(settings, 'USE_FASTTEXT') and settings.USE_FASTTEXT:
            FASTTEXT_MODEL = fasttext.load_model('lid.176.bin')
            USE_FASTTEXT = True
    except:
        logger.info("FastText model not available, using basic language detection")
except ImportError:
    logger.info("FastText package not installed, using basic language detection")

# Kiểm tra nếu có thể sử dụng langdetect (phương pháp dự phòng)
USE_LANGDETECT = False
try:
    from langdetect import detect, detect_langs
    from langdetect.lang_detect_exception import LangDetectException
    USE_LANGDETECT = True
except ImportError:
    logger.info("Langdetect not available, using simple language detection heuristics")

# Tạo instance router
router = APIRouter(prefix="/analyzeAI", tags=["analyzeAI"])

# Khai báo các biến cấu hình
MAX_CHUNK_SIZE = 300  # Độ dài tối đa của một chunk
MAX_CONCURRENT_REQUESTS = 5  # Số lượng request đồng thời tối đa
CACHE_TIMEOUT = 3600  # Thời gian cache kết quả (giây)
TEMPERATURE = 0.2  # Nhiệt độ cho API GPT (giảm để có kết quả chính xác hơn)
MODEL = "gpt-4o-mini"  # Model GPT sử dụng

class LanguageConfidence:
    """Class để lưu trữ ngôn ngữ và độ tin cậy."""
    def __init__(self, language: str, code: str, confidence: float):
        self.language = language
        self.code = code
        self.confidence = confidence

def get_cache_key(text: str) -> str:
    """Tạo khóa cache từ văn bản."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def set_cache(key: str, value: Any, timeout: int = CACHE_TIMEOUT):
    """Lưu giá trị vào cache."""
    in_memory_cache[key] = (time.time() + timeout, value)

def get_cache(key: str) -> Any:
    """Lấy giá trị từ cache."""
    if key in in_memory_cache:
        expiry, value = in_memory_cache[key]
        if time.time() < expiry:
            return value
        else:
            del in_memory_cache[key]
    return None

def clean_cache():
    """Xóa các bản ghi cache cũ."""
    current_time = time.time()
    keys_to_remove = []
    
    for key, (expiry, _) in in_memory_cache.items():
        if current_time > expiry:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del in_memory_cache[key]

def simple_tokenize(text: str) -> List[str]:
    """
    Phương pháp đơn giản để tách văn bản thành các từ mà không cần NLTK.
    """
    # Tách theo khoảng trắng nhưng giữ lại dấu câu
    pattern = r'(\w+|\s+|[^\w\s])'
    return [token for token in re.findall(pattern, text) if token.strip()]

def simple_sent_tokenize(text: str) -> List[str]:
    """
    Phương pháp đơn giản để tách văn bản thành các câu mà không cần NLTK.
    """
    # Tách theo các dấu câu phổ biến nhưng giữ nguyên dấu
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\:|\;)\s'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]

def get_language_name(code: str) -> str:
    """Chuyển đổi mã ngôn ngữ sang tên đầy đủ."""
    language_map = {
        'af': 'Afrikaans', 'ar': 'Arabic', 'bg': 'Bulgarian', 'bn': 'Bengali', 
        'ca': 'Catalan', 'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 
        'de': 'German', 'el': 'Greek', 'en': 'English', 'es': 'Spanish', 
        'et': 'Estonian', 'fa': 'Persian', 'fi': 'Finnish', 'fr': 'French', 
        'gu': 'Gujarati', 'he': 'Hebrew', 'hi': 'Hindi', 'hr': 'Croatian', 
        'hu': 'Hungarian', 'id': 'Indonesian', 'it': 'Italian', 'ja': 'Japanese', 
        'kn': 'Kannada', 'ko': 'Korean', 'lt': 'Lithuanian', 'lv': 'Latvian', 
        'mk': 'Macedonian', 'ml': 'Malayalam', 'mr': 'Marathi', 'my': 'Burmese', 
        'ne': 'Nepali', 'nl': 'Dutch', 'no': 'Norwegian', 'pa': 'Punjabi', 
        'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 
        'sk': 'Slovak', 'sl': 'Slovenian', 'so': 'Somali', 'sq': 'Albanian', 
        'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu', 
        'th': 'Thai', 'tl': 'Tagalog', 'tr': 'Turkish', 'uk': 'Ukrainian', 
        'ur': 'Urdu', 'vi': 'Vietnamese', 'zh': 'Chinese', 'zh-cn': 'Chinese',
        'zh-tw': 'Chinese'
    }
    
    return language_map.get(code, f"Unknown ({code})")

def detect_language(text: str) -> List[LanguageConfidence]:
    """
    Phát hiện ngôn ngữ sử dụng phương pháp có sẵn (fasttext, langdetect hoặc heuristic).
    """
    if not text.strip():
        return []
        
    # Thử với FastText nếu có
    if USE_FASTTEXT:
        try:
            # FastText trả về dạng ('__label__en', 0.9876)
            predictions = FASTTEXT_MODEL.predict(text, k=3)  # Top 3 dự đoán
            languages = []
            
            for i in range(len(predictions[0])):
                lang_code = predictions[0][i].replace('__label__', '')
                confidence = float(predictions[1][i])
                
                # Ánh xạ mã ngôn ngữ sang tên đầy đủ
                lang_name = get_language_name(lang_code)
                languages.append(LanguageConfidence(
                    language=lang_name,
                    code=lang_code,
                    confidence=confidence
                ))
            
            return languages
        except Exception as e:
            logger.warning(f"FastText error: {e}, falling back to langdetect")
    
    # Thử với langdetect nếu có
    if USE_LANGDETECT:
        try:
            # Phát hiện các ngôn ngữ có thể có với độ tin cậy
            languages = detect_langs(text)
            return [
                LanguageConfidence(
                    language=get_language_name(lang.lang),
                    code=lang.lang,
                    confidence=lang.prob
                )
                for lang in languages
            ]
        except Exception as e:
            logger.warning(f"Langdetect error: {e}, falling back to heuristics")
    
    # Phương pháp dự phòng cuối cùng: heuristic đơn giản
    # Kiểm tra các đặc điểm cơ bản của từng ngôn ngữ
    
    # Danh sách một số ngôn ngữ phổ biến và các ký tự đặc trưng
    # (Đây chỉ là phương pháp đơn giản, không chính xác như các thư viện)
    language_patterns = [
        (r'[\u0041-\u007A]', 'en', 'English'),  # Latin alphabet (mặc định tiếng Anh)
        (r'[\u00C0-\u00FF]', 'fr', 'French'),   # Latin extended (giả định là tiếng Pháp)
        (r'[\u0400-\u04FF]', 'ru', 'Russian'),  # Cyrillic
        (r'[\u0600-\u06FF]', 'ar', 'Arabic'),   # Arabic
        (r'[\u0900-\u097F]', 'hi', 'Hindi'),    # Devanagari
        (r'[\u3040-\u309F]', 'ja', 'Japanese'), # Hiragana
        (r'[\u30A0-\u30FF]', 'ja', 'Japanese'), # Katakana
        (r'[\u4E00-\u9FFF]', 'zh', 'Chinese'),  # Han/Kanji
        (r'[\uAC00-\uD7A3]', 'ko', 'Korean'),   # Hangul
        (r'[\u0E00-\u0E7F]', 'th', 'Thai'),     # Thai
        (r'[\u0370-\u03FF]', 'el', 'Greek'),    # Greek
    ]
    
    # Đếm số lượng ký tự khớp với từng mẫu
    counts = {}
    total_chars = len(re.findall(r'\S', text))  # Đếm ký tự không phải khoảng trắng
    
    if total_chars == 0:
        return []
    
    for pattern, code, name in language_patterns:
        match_count = len(re.findall(pattern, text))
        if match_count > 0:
            confidence = match_count / total_chars
            counts[code] = (name, confidence)
    
    # Sắp xếp theo độ tin cậy giảm dần
    languages = []
    for code, (name, confidence) in sorted(counts.items(), key=lambda x: x[1][1], reverse=True):
        languages.append(LanguageConfidence(language=name, code=code, confidence=confidence))
    
    # Nếu không phát hiện được ngôn ngữ, giả định là tiếng Anh
    if not languages:
        languages.append(LanguageConfidence(language="English", code="en", confidence=0.5))
    
    return languages

def split_text_smart(text: str) -> List[str]:
    """
    Tách văn bản thành các đoạn nhỏ một cách thông minh dựa trên cấu trúc văn bản.
    """
    # Bước 1: Tách theo dấu câu kết thúc (.?!:;)
    sentences = simple_sent_tokenize(text)
    
    # Bước 2: Xử lý các câu dài
    result_chunks = []
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        if len(sentence) <= MAX_CHUNK_SIZE:
            result_chunks.append(sentence.strip())
        else:
            # Tách câu dài theo dấu phẩy
            comma_chunks = re.split(r'(?<=,)\s', sentence)
            
            current_chunk = ""
            for chunk in comma_chunks:
                if len(current_chunk) + len(chunk) <= MAX_CHUNK_SIZE:
                    if current_chunk:
                        current_chunk += ", " + chunk
                    else:
                        current_chunk = chunk
                else:
                    if current_chunk:
                        result_chunks.append(current_chunk.strip())
                    
                    # Xử lý các chunk quá dài (vượt quá MAX_CHUNK_SIZE)
                    if len(chunk) > MAX_CHUNK_SIZE:
                        # Tách theo dấu cách cho những chunk quá dài
                        words = chunk.split()
                        sub_chunk = ""
                        for word in words:
                            if len(sub_chunk) + len(word) + 1 <= MAX_CHUNK_SIZE:
                                if sub_chunk:
                                    sub_chunk += " " + word
                                else:
                                    sub_chunk = word
                            else:
                                result_chunks.append(sub_chunk.strip())
                                sub_chunk = word
                        
                        if sub_chunk:
                            current_chunk = sub_chunk
                        else:
                            current_chunk = ""
                    else:
                        current_chunk = chunk
            
            if current_chunk:
                result_chunks.append(current_chunk.strip())
    
    return result_chunks

def get_enhanced_prompt() -> str:
    """
    Tạo prompt nâng cao tập trung vào phát hiện thay đổi ngôn ngữ ở cấp độ từ.
    """
    return (
        "You are an expert multilingual text analyzer focused on detecting language switches at the word level. "
        "Your task is to analyze the given text and identify EVERY language change, even for a SINGLE WORD or CHARACTER. "
        "Pay special attention to isolated foreign words and short phrases that might appear in a different language.\n\n"

        "###**Key Requirements:**\n"
        "- **Word-level detection**: Identify even a single word or character in a different language\n"
        "- **No minimum length**: There is no minimum length for a language segment - even a single character matters\n"
        "- **Preserve exact text**: Keep the original text intact, including punctuation\n"
        "- **Be precise with boundaries**: Determine exactly where one language ends and another begins\n"
        "- **Handle mixed words**: Identify when part of a word contains a different language (e.g., loan words)\n\n"

        "###**Language Code Mapping (BCP-47 Format):**\n"
        "Afrikaans => af, Arabic => ar, Bengali => bn, Bulgarian => bg, Burmese => my, Cantonese => zh-yue, "
        "Chinese => zh, Czech => cs, Danish => da, Dutch => nl, English => en, Estonian => et, Filipino => tl, "
        "Finnish => fi, French => fr, German => de, Greek => el, Gujarati => gu, Hebrew => he, Hindi => hi, "
        "Hungarian => hu, Icelandic => is, Indonesian => id, Italian => it, Japanese => ja, Javanese => jv, "
        "Kannada => kn, Khmer => km, Korean => ko, Lao => lo, Latvian => lv, Lithuanian => lt, Malayalam => ml, "
        "Marathi => mr, Mongolian => mn, Nepali => ne, Norwegian => no, Polish => pl, Portuguese => pt, "
        "Punjabi => pa, Romanian => ro, Russian => ru, Sinhala => si, Slovak => sk, Slovenian => sl, "
        "Spanish => es, Sundanese => su, Swahili => sw, Swedish => sv, Tamil => ta, Telugu => te, Thai => th, "
        "Turkish => tr, Ukrainian => uk, Urdu => ur, Uzbek => uz, Vietnamese => vi, Zulu => zu.\n\n"

        "###**Expected JSON Structure:**\n"
        "{\n"
        "  \"language\": [\n"
        "    {\n"
        "      \"name\": \"<Language name>\",\n"
        "      \"code\": \"<BCP-47 code>\",\n"
        "      \"text\": \"<Extracted text belonging to this language>\"\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n\n"

        "###**Examples of Single-Word Language Changes:**\n"
        "Input: \"I love eating phở in Vietnam\"\n"
        "Output: [{\"name\": \"English\", \"code\": \"en\", \"text\": \"I love eating \"}, "
        "{\"name\": \"Vietnamese\", \"code\": \"vi\", \"text\": \"phở\"}, "
        "{\"name\": \"English\", \"code\": \"en\", \"text\": \" in Vietnam\"}]\n\n"
        
        "Input: \"Xin chào, my name is David\"\n"
        "Output: [{\"name\": \"Vietnamese\", \"code\": \"vi\", \"text\": \"Xin chào\"}, "
        "{\"name\": \"English\", \"code\": \"en\", \"text\": \", my name is David\"}]\n\n"
    )

def analyze_word_by_word(text: str) -> List[Dict[str, Any]]:
    """
    Phân tích văn bản từng từ một cho các đoạn có sự thay đổi ngôn ngữ đột ngột.
    """
    words = simple_tokenize(text)
    results = []
    
    current_language = None
    current_text = ""
    
    for word in words:
        # Bỏ qua khoảng trắng
        if not word.strip():
            # Thêm khoảng trắng vào đoạn hiện tại
            if current_text:
                current_text += word
            continue
        
        # Phát hiện ngôn ngữ cho từ hiện tại
        lang_result = detect_language(word)
        
        if not lang_result:
            # Không thể xác định ngôn ngữ, thêm vào đoạn hiện tại
            current_text += word
            continue
        
        detected_language = lang_result[0]
        
        # Nếu đây là từ đầu tiên hoặc ngôn ngữ thay đổi
        if current_language is None or detected_language.code != current_language["code"]:
            # Lưu đoạn hiện tại nếu có
            if current_text:
                results.append({
                    "name": current_language["name"],
                    "code": current_language["code"],
                    "text": current_text.strip()
                })
            
            # Bắt đầu đoạn mới
            current_language = {
                "name": detected_language.language,
                "code": detected_language.code
            }
            current_text = word
        else:
            # Tiếp tục với đoạn hiện tại
            current_text += word
    
    # Thêm đoạn cuối cùng
    if current_text and current_language:
        results.append({
            "name": current_language["name"],
            "code": current_language["code"],
            "text": current_text.strip()
        })
    
    return results

def get_language_transitions(text: str) -> List[int]:
    """
    Phát hiện các vị trí có khả năng chuyển đổi ngôn ngữ cao.
    Trả về danh sách các chỉ số trong văn bản nơi có thể xảy ra chuyển đổi.
    """
    transitions = []
    words = simple_tokenize(text)
    
    current_position = 0
    last_language = None
    
    for word in words:
        # Bỏ qua khoảng trắng và dấu câu
        if not word.strip() or not re.search(r'\w', word):
            current_position += len(word)
            continue
            
        # Phát hiện ngôn ngữ của từ hiện tại
        lang_result = detect_language(word)
        
        if lang_result:
            current_language = lang_result[0].code
            
            # Kiểm tra xem có sự chuyển đổi ngôn ngữ không
            if last_language is not None and current_language != last_language:
                transitions.append(current_position)
                
            last_language = current_language
        
        current_position += len(word)
    
    return transitions

def adaptive_chunking(text: str) -> List[Dict[str, Any]]:
    """
    Phân đoạn thích ứng dựa trên độ phức tạp ngôn ngữ của văn bản.
    """
    # Phân tích theo câu thông thường
    sentences = simple_sent_tokenize(text)
    chunks = []
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= MAX_CHUNK_SIZE:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return [{"text": chunk, "requires_gpt": True} for chunk in chunks]

async def analyze_chunk_with_gpt(chunk: str) -> List[Dict[str, str]]:
    """
    Sử dụng GPT để phân tích một đoạn văn bản.
    """
    # Tạo hash cho chunk để lưu cache
    cache_key = get_cache_key(chunk)
    
    # Kiểm tra cache
    cached_result = get_cache(cache_key)
    if cached_result:
        logger.info(f"Cache hit for chunk: {chunk[:30]}...")
        return cached_result
    
    # Cấu hình prompt nâng cao
    messages = [
        {
            "role": "system",
            "content": get_enhanced_prompt()
        },
        {
            "role": "user",
            "content": f"Analyze this text: {chunk}"
        }
    ]
    
    try:
        # Gọi API GPT để phân tích
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE
        )

        # Lấy kết quả trả về từ GPT
        result = response.choices[0].message.content

        # Parse kết quả trả về dưới dạng JSON
        try:
            chunk_result = json.loads(result)
            fragments = chunk_result.get("language", [])
        except json.JSONDecodeError:
            # Thử lọc ra phần JSON từ kết quả
            try:
                json_text = re.search(r'(\{.*\})', result, re.DOTALL)
                if json_text:
                    chunk_result = json.loads(json_text.group(1))
                    fragments = chunk_result.get("language", [])
                else:
                    fragments = [{"name": "Unknown", "code": "und", "text": chunk}]
            except (json.JSONDecodeError, AttributeError):
                fragments = [{"name": "Unknown", "code": "und", "text": chunk}]
        
        # Lưu vào cache
        set_cache(cache_key, fragments)
        
        return fragments
    except Exception as e:
        logger.error(f"Error analyzing chunk with GPT: {str(e)}")
        return [{"name": "Error", "code": "err", "text": chunk}]

async def analyze_chunks_parallel(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Phân tích nhiều đoạn văn bản đồng thời.
    """
    # Sử dụng semaphore để giới hạn số lượng request đồng thời
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def analyze_with_limit(chunk):
        async with semaphore:
            return await analyze_chunk_with_gpt(chunk["text"])
    
    # Tạo các task phân tích
    tasks = [analyze_with_limit(chunk) for chunk in chunks]
    
    # Thực hiện các task đồng thời
    results = await asyncio.gather(*tasks)
    
    # Gộp tất cả kết quả
    all_fragments = []
    for fragments in results:
        all_fragments.extend(fragments)
    
    return all_fragments

def merge_language_fragments(fragments: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Gộp các đoạn văn bản cùng ngôn ngữ liền kề.
    """
    if not fragments:
        return []
        
    merged = []
    current = fragments[0].copy()
    
    for fragment in fragments[1:]:
        # Kiểm tra nếu ngôn ngữ hiện tại giống với ngôn ngữ trước đó
        if fragment["code"] == current["code"]:
            # Gộp văn bản
            current["text"] += fragment["text"]
        else:
            # Thêm đoạn hiện tại vào kết quả và bắt đầu đoạn mới
            merged.append(current)
            current = fragment.copy()
    
    # Thêm đoạn cuối cùng
    merged.append(current)
    
    return merged

async def analyze_multilingual_text(text: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Phân tích văn bản đa ngôn ngữ với chiến lược phân tích thích ứng.
    """
    try:
        # Phân đoạn thích ứng dựa trên độ phức tạp ngôn ngữ
        chunks = adaptive_chunking(text)
        logger.info(f"Adaptive chunking created {len(chunks)} chunks")
        
        # Phân tích các chunk đồng thời
        all_fragments = await analyze_chunks_parallel(chunks)
        
        # Gộp các đoạn cùng ngôn ngữ liền kề
        merged_fragments = merge_language_fragments(all_fragments)
        
        return {"language": merged_fragments}
    except Exception as e:
        logger.error(f"Error during multilingual text analysis: {str(e)}")
        raise RuntimeError(f"Error during multilingual text analysis: {e}")

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks, api_key: str = get_api_key):
    """
    API phân tích văn bản đa ngôn ngữ cải tiến.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        # Gọi hàm phân tích văn bản
        analysis_result = await analyze_multilingual_text(request.text)
        
        # Lên lịch làm sạch cache
        background_tasks.add_task(clean_cache)
        
        # Chuyển đổi kết quả phân tích thành dạng trả về của API
        language_fragments = [
            LanguageFragment(name=lang["name"], code=lang["code"], text=lang["text"])
            for lang in analysis_result.get("language", [])
        ]
        
        # Trả kết quả phân tích
        return AnalyzeResponse(language=language_fragments)

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

# Thêm các endpoint cho phân tích theo từng từ

@router.post("/analyze/word-level")
async def analyze_word_level(request: AnalyzeRequest, api_key: str = get_api_key):
    """
    API phân tích ở mức độ từng từ, đặc biệt hữu ích cho văn bản có thay đổi ngôn ngữ đột ngột.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        # Phân tích từng từ
        fragments = analyze_word_by_word(request.text)
        
        # Chuyển đổi kết quả thành dạng trả về của API
        language_fragments = [
            LanguageFragment(name=lang["name"], code=lang["code"], text=lang["text"])
            for lang in fragments
        ]
        
        return AnalyzeResponse(language=language_fragments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))