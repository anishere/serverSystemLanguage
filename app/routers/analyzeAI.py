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
                    "You are an expert multilingual text analyzer with a focus on accurately detecting language switches. "
                    "Your task is to carefully analyze the given text and **precisely detect every language switch**, even if a change happens **mid-sentence, within a single word, or in mixed phrases**.\n\n"

                    "###**Key Segmentation Rules:**\n"
                    "- **DO NOT** group text of the same language together if a **new phrase or idea starts**.\n"
                    "- **Immediately split text when a new language appears**, even for **one word or part of a word**.\n"
                    "- **Each JSON object must contain text in only ONE language.**\n"
                    "- **Preserve the original order** of the text without modifying the structure.\n"
                    "- **Ensure language detection is highly accurate**. Double-check and prevent misclassifications (e.g., Russian being labeled as French).\n"
                    "- **Return the output strictly in JSON format without any additional text or explanations.**\n\n"

                    "###**Advanced Language Detection Considerations:**\n"
                    "- **Context matters**: For proper identification, use contextual clues. For example, if a word is common in one language but has a different meaning or usage in another, consider the broader sentence context.\n"
                    "- **Mixed phrases and transliterations**: Ensure that transliterations or mixed-use phrases (e.g., 'Konnichiwa' in a Japanese context) are correctly identified based on both context and language characteristics.\n"
                    "- **Ensure differentiation** for languages with similar words or common phrases (e.g., ‘siempre’ in Spanish vs. Vietnamese context).\n"
                    "- **Special characters**: Use the presence of specific characters (e.g., Chinese characters or Arabic script) to help detect language switches more effectively.\n\n"

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

                    "###**Examples:**\n"
                    "**Example Input:**\n"
                    "'Class mình à, aujourd’hui, j’ai một chút feedback для вас, d’accord?'\n\n"

                    "**Expected JSON Output:**\n"
                    "{\n"
                    "  \"language\": [\n"
                    "    {\"name\": \"English\", \"code\": \"en\", \"text\": \"Class\"},\n"
                    "    {\"name\": \"Vietnamese\", \"code\": \"vi\", \"text\": \"mình à,\"},\n"
                    "    {\"name\": \"French\", \"code\": \"fr\", \"text\": \"aujourd’hui,\"},\n"
                    "    {\"name\": \"French\", \"code\": \"fr\", \"text\": \"j’ai\"},\n"
                    "    {\"name\": \"Vietnamese\", \"code\": \"vi\", \"text\": \"một chút\"},\n"
                    "    {\"name\": \"English\", \"code\": \"en\", \"text\": \"feedback\"},\n"
                    "    {\"name\": \"Russian\", \"code\": \"ru\", \"text\": \"для вас,\"},\n"
                    "    {\"name\": \"French\", \"code\": \"fr\", \"text\": \"d’accord?\"}\n"
                    "  ]\n"
                    "}\n\n"

                    "###**Fixing Misclassifications:**\n"
                    "- **Double-check language classification** to prevent incorrect labeling (e.g., Russian words incorrectly labeled as French).\n"
                    "- **Use advanced contextual understanding** to differentiate similar words across languages (e.g., ‘siempre’ in Spanish vs. Vietnamese context).\n"
                    "- **DO NOT infer missing words or change the text structure.** Keep everything as it appears in the original input.\n\n"

                    "###**Additional Notes:**\n"
                    "- **DO NOT** include additional explanations or formatting beyond the required JSON.\n"
                    "- **DO NOT** infer additional meaning beyond what is explicitly provided in the text.\n"
                    "- **Ensure each language fragment is correctly identified and separated.**\n\n"

                    "###**Important:**\n"
                    "- **Contextual clues** can help improve language detection (e.g., the combination of a name like 'Jean' and a word like 'Bonjour' suggests French).\n"
                    "- Be aware of common transliterations or slang that may affect detection (e.g., 'Konnichiwa' can refer to both Japanese and informal Japanese terms).\n"
                    "- **Special characters** in the text can help indicate language switches (e.g., Chinese characters or Arabic script)."
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
