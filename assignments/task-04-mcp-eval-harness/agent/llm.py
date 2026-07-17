import os
from dotenv import load_dotenv

load_dotenv()


def get_llm(temperature: float = 0.0):
    """Returns a chat model based on the first available provider API key.

    Checks providers in order of tool-calling reliability: Anthropic, OpenAI,
    Google, then Groq as the free-tier fallback. This satisfies the
    multi-provider pattern from Tasks 01/02 (no per-file duplication) while
    keeping the project runnable on a Groq-only key.

    NOTE on the Groq branch: llama-3.3-70b-versatile has a well-documented,
    ongoing reliability issue where it sometimes emits malformed pseudo-XML
    function calls instead of a proper structured tool call, causing
    `tool_use_failed` 400 errors from Groq's API. openai/gpt-oss-20b is a
    model Groq hosts specifically for reliable structured tool calling, and
    is used here instead for that reason.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=temperature)

    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)

    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

    raise ValueError(
        "No supported LLM provider API key found. Set one of: "
        "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY."
    )