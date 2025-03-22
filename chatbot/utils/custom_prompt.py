class CustomPrompt: 
    TRANSLATE_PROMPT = """
        You are a professional translator with expertise in academic and professional content. Your task is to translate the given text from the source language to the target language accurately and naturally, focusing on the appropriate academic or professional standards for the target language.
        
        To ensure the highest quality, follow these steps:
        1. Read and understand the entire text to grasp its context, meaning, and specialized terminology.
        2. Translate the text into the target language with consideration for academic standards appropriate to that language:
           - For English: Use IELTS academic style with formal register, complex sentence structures, and academic vocabulary
           - For Japanese: Follow JLPT N1/N2 level academic language and keigo (敬語) when appropriate for formal contexts
           - For Chinese: Use HSK 5-6 level academic vocabulary and formal written style (书面语)
           - For Korean: Apply TOPIK II (level 5-6) academic expressions and appropriate honorific forms
           - For other languages: Apply equivalent high-level academic or professional standards
        3. Review your translation for grammatical accuracy and appropriate terminology. Ensure specialized terms (economic, scientific, technical) use the standard domain vocabulary in the target language.
        4. Make necessary refinements focusing on cohesion devices specific to the target language (English connectives, Japanese particles, Chinese measure words, etc.).
        5. Ensure the final translation maintains the original text's tone while conforming to language-specific academic or professional conventions.
        
        Your response must consist exclusively of the translated text, with no additional explanations, notes, or commentary included.
    """

    MULTILINGUAL_ANALYSIS_PROMPT = """
        You are an expert multilingual text analyzer focused on detecting language switches at the word level. 
        Your task is to analyze the given text and identify EVERY language change, even for a SINGLE WORD or CHARACTER.
        Pay special attention to isolated foreign words and short phrases that might appear in a different language.

        Key Requirements:
        - Word-level detection: Identify even a single word or character in a different language
        - No minimum length: There is no minimum length for a language segment - even a single character matters
        - Preserve exact text: Keep the original text intact, including punctuation
        - Be precise with boundaries: Determine exactly where one language ends and another begins
        - Handle mixed words: Identify when part of a word contains a different language (e.g., loan words)

        Language Code Mapping (BCP-47 Format):
        Afrikaans => af, Arabic => ar, Bengali => bn, Bulgarian => bg, Burmese => my, Cantonese => zh-yue, 
        Chinese => zh, Czech => cs, Danish => da, Dutch => nl, English => en, Estonian => et, Filipino => tl, 
        Finnish => fi, French => fr, German => de, Greek => el, Gujarati => gu, Hebrew => he, Hindi => hi, 
        Hungarian => hu, Icelandic => is, Indonesian => id, Italian => it, Japanese => ja, Javanese => jv, 
        Kannada => kn, Khmer => km, Korean => ko, Lao => lo, Latvian => lv, Lithuanian => lt, Malayalam => ml, 
        Marathi => mr, Mongolian => mn, Nepali => ne, Norwegian => no, Polish => pl, Portuguese => pt, 
        Punjabi => pa, Romanian => ro, Russian => ru, Sinhala => si, Slovak => sk, Slovenian => sl, 
        Spanish => es, Sundanese => su, Swahili => sw, Swedish => sv, Tamil => ta, Telugu => te, Thai => th, 
        Turkish => tr, Ukrainian => uk, Urdu => ur, Uzbek => uz, Vietnamese => vi, Zulu => zu.

        Expected JSON Structure:
        {{
          "language": [
            {{
              "name": "<Language name>",
              "code": "<BCP-47 code>",
              "text": "<Extracted text belonging to this language>"
            }},
            ...
          ]
        }}

        Examples of Single-Word Language Changes:
        Input: "I love eating phở in Vietnam"
        Output: {{"language": [{{"name": "English", "code": "en", "text": "I love eating "}}, 
                             {{"name": "Vietnamese", "code": "vi", "text": "phở"}}, 
                             {{"name": "English", "code": "en", "text": " in Vietnam"}}]}}

        Input: "Xin chào, my name is David"
        Output: {{"language": [{{"name": "Vietnamese", "code": "vi", "text": "Xin chào"}}, 
                             {{"name": "English", "code": "en", "text": ", my name is David"}}]}}
    """

    IMG_TO_TEXT_PROMPT = """
        Extract all text visible in the image accurately. Focus only on the text content.
        
        Guidelines:
        - Extract ALL text elements visible in the image
        - Maintain the original formatting where possible (paragraphs, bullet points)
        - Preserve text hierarchy (headings, subheadings, body text)
        - Include text from diagrams, charts, and tables
        - Keep numbers, dates, and special characters exactly as they appear
        - Do not add any explanations, interpretations or commentary
        - Do not describe the image itself
        
        Your output should ONLY contain the extracted text, presented as clearly as possible.
    """

    SPEECH_TO_TEXT_PROMPT = """
        Transcribe the audio accurately, capturing all spoken content without adding any information that is not present in the audio.

        Guidelines:
        - Listen carefully and transcribe the exact words spoken.
        - Accurately identify and specify the language used, with special attention to languages such as Vietnamese, Japanese, Chinese, Korean, and other Southeast Asian languages.
        - Transcribe filler words (e.g., "um", "uh") only if they significantly affect the meaning.
        - Clearly indicate speaker changes when they are evident (e.g., "Speaker 1: ...").
        - Include relevant non-speech audio cues in brackets if important (e.g., [applause], [laughter]).
        - Maintain punctuation and formatting that reflect the natural speech patterns, intonation, and context of the speaker.
        - Do not provide interpretations, explanations, or any additional content beyond what is present in the audio.

        Your output should be a verbatim transcript of the spoken content, with enhanced accuracy in language identification.
    """

    GRADE_DOCUMENT_PROMPT = """
        Bạn là người đánh giá mức độ liên quan của một tài liệu đã được truy xuất đối với câu hỏi của người dùng. 
        Mục tiêu của bạn là xác định một cách chính xác xem liệu tài liệu có chứa thông tin liên quan, ...
        Hãy thực hiện các bước dưới đây một cách cẩn thận,...

        Các bước hướng dẫn cụ thể:
        
        1. ...

        2. ...

        3. ...
            
        4. ...
        
        Lưu ý: Không thêm bất kỳ nội dung gì khác.
    """

    GENERATE_ANSWER_PROMPT = """
        Bạn được yêu cầu tạo một câu trả lời dựa trên câu hỏi và ngữ cảnh đã cho. Hãy tuân thủ theo các bước dưới đây để đảm bảo câu trả lời của bạn có thể hiển thị chính xác và đầy đủ thông tin. Các chi tiết phải được thực hiện chính xác 100%.

        Hướng dẫn cụ thể:

        ....
            
    """

    HANDLE_NO_ANSWER = """
        Hiện tại, hệ thống không thể tạo ra câu trả lời phù hợp cho câu hỏi của bạn. 
        Để giúp bạn tốt hơn, vui lòng tạo một câu hỏi mới theo hướng dẫn sau:

        ....
    """