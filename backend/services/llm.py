import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

def get_chat(provider: str | None = None, model: str | None = None, temperature: float = 0) -> BaseChatModel:
    provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
    if provider == "openai":
        return ChatOpenAI(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)
    if provider == "anthropic":
        return ChatAnthropic(model=model or os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet"), temperature=temperature)
    # default -> gemini
    return ChatGoogleGenerativeAI(model=model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), temperature=temperature)