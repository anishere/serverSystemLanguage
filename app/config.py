# File cấu hình chung cho ứng dụng

import os
from dotenv import load_dotenv

# Load các biến môi trường từ file .env
load_dotenv()


class Settings:
    # SETTING
    DIR_ROOT = os.path.dirname(os.path.abspath(".env"))
    # API KEY
    API_KEY = os.environ["API_KEY"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    # chat bot
    KEY_API_GPT = os.environ["OPENAI_API_KEY"]
    
    NUM_DOC = os.environ["NUM_DOC"]
    
    LLM_NAME = os.environ["NUM_DOC"]
    
    OPENAI_LLM = os.environ["OPENAI_LLM"]

settings = Settings()
