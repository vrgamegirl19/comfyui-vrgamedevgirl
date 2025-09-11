import os


def get_open_ai_api_key() -> str:
    return os.environ.get("OPEN_AI_API_KEY", None)
