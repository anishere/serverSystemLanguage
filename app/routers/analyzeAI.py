from fastapi import APIRouter, HTTPException
from app.models.analyzeAI import AnalyzeRequest, AnalyzeResponse, LanguageFragment
from app.config import settings
import openai  # Thư viện OpenAI để gọi API
from app.security.security import get_api_key
import json

# Cấu hình client OpenAI
client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

# Tạo instance router
router = APIRouter(prefix="/analyzeAI", tags=["analyzeAI"])

def analyze_multilingual_text_with_gpt(text: str) -> dict:
    """
    Phân tích văn bản đa ngôn ngữ và trả về các đoạn văn bản theo ngôn ngữ.
    """
    try:
        # Cấu hình prompt yêu cầu GPT phân tích văn bản đa ngôn ngữ
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a highly precise multilingual text analyzer. "
                    "Analyze the input text and identify every language it contains. "
                    "For each language, extract all the text fragments belonging to that language. "
                    "Use the following list to map each language to its BCP-47 code: \n\n"

                    # == DANH SÁCH NGÔN NGỮ & CODE (phiên bản rút gọn, 1 code/ngôn ngữ) ==
                    "Afrikaans => af"
                    "Arabic => ar"
                    "Bengali => bn"
                    "Bulgarian => bg"
                    "Burmese => my"
                    "Cantonese => zh-yue"
                    "Chinese => zh"
                    "Czech => cs"
                    "Danish => da"
                    "Dutch => nl"
                    "English => en"
                    "Estonian => et"
                    "Filipino => tl"
                    "Finnish => fi"
                    "French => fr"
                    "German => de"
                    "Greek => el"
                    "Gujarati => gu"
                    "Hebrew => he"
                    "Hindi => hi"
                    "Hungarian => hu"
                    "Icelandic => is"
                    "Indonesian => id"
                    "Italian => it"
                    "Japanese => ja"
                    "Javanese => jv"
                    "Kannada => kn"
                    "Khmer => km"
                    "Korean => ko"
                    "Lao => lo"
                    "Latvian => lv"
                    "Lithuanian => lt"
                    "Malayalam => ml"
                    "Marathi => mr"
                    "Mongolian => mn"
                    "Nepali => ne"
                    "Norwegian => no"
                    "Polish => pl"
                    "Portuguese => pt"
                    "Punjabi => pa"
                    "Romanian => ro"
                    "Russian => ru"
                    "Sinhala => si"
                    "Slovak => sk"
                    "Slovenian => sl"
                    "Spanish => es"
                    "Sundanese => su"
                    "Swahili => sw"
                    "Swedish => sv"
                    "Tamil => ta"
                    "Telugu => te"
                    "Thai => th"
                    "Turkish => tr"
                    "Ukrainian => uk"
                    "Urdu => ur"
                    "Uzbek => uz"
                    "Vietnamese => vi"
                    "Zulu => zu"

                    # == HƯỚNG DẪN CẤU TRÚC TRẢ VỀ ==
                    "Return the result as a JSON object with the structure:\n"
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

                    "Ensure no text fragments are missed, and correctly associate each fragment with its language. "
                    "Handle multilingual cases where languages are mixed within the same sentence. "
                    "Example input: 'hello my name is ana, 我十八岁 and étudier en Australie'. "
                    "Example output:\n"
                    "{\n"
                    "  \"language\": [\n"
                    "    {\n"
                    "      \"name\": \"English\",\n"
                    "      \"code\": \"en-US\",\n"
                    "      \"text\": \"hello my name is ana and\"\n"
                    "    },\n"
                    "    {\n"
                    "      \"name\": \"Chinese\",\n"
                    "      \"code\": \"zh-CN\",\n"
                    "      \"text\": \"我十八岁\"\n"
                    "    },\n"
                    "    {\n"
                    "      \"name\": \"French\",\n"
                    "      \"code\": \"fr-FR\",\n"
                    "      \"text\": \"étudier en Australie\"\n"
                    "    }\n"
                    "  ]\n"
                    "}\n\n"

                    "Only output the final JSON. No explanations. No extra commentary."
                )
            },
            {
                "role": "user",
                "content": f"Analyze this text: {text}"
            }
        ]

        # Gọi API GPT để phân tích
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Hoặc model bạn đang sử dụng
            messages=messages,
            temperature=0.5
        )

        # Lấy kết quả trả về từ GPT
        result = response.choices[0].message.content

        # Parse kết quả trả về dưới dạng JSON
        return json.loads(result)

    except Exception as e:
        raise RuntimeError(f"Error during multilingual text analysis: {e}")

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, api_key: str = get_api_key):
    """
    API phân tích văn bản đa ngôn ngữ, trả về các đoạn văn bản phân theo ngôn ngữ.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        # Gọi hàm phân tích văn bản
        analysis_result = analyze_multilingual_text_with_gpt(request.text)
        
        # Chuyển đổi kết quả phân tích thành dạng trả về của API
        language_fragments = [
            LanguageFragment(name=lang["name"], code=lang["code"], text=lang["text"])
            for lang in analysis_result.get("language", [])
        ]
        
        # Trả kết quả phân tích
        return AnalyzeResponse(language=language_fragments)

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
