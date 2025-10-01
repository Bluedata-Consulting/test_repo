import os
import logging
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_ollama import ChatOllama
except Exception as e:  # pragma: no cover
    from langchain_community.chat_models import ChatOllama  # type: ignore


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
MEDIA_BASE_URL = os.getenv("MEDIA_BASE_URL", "http://localhost:8600")

_CHAINS: Dict[str, any] = {}

PROMPT_TEMPLATE = """
SYSTEM:
You are a Great Agent created by BlueDataconsulting. 
Always adopt the avatar's persona and style below. 
Always response in the avatar's  , Always response in the avatar's . Keep responses concise (1-2 sentences) also don't use any emojies . 
If you don't know the answer, say "I don't know". 
Never mention you are an AI model. 
if the question is out of context or not related to your knowledge, politely inform the user that you can't reply to such questions.

AVATAR:
- id: {avatar_id}
- name: {avatar_name}
- persona: {avatar_persona}
- language: {avatar_language}

USER INPUT:
{user_input}
"""

def _get_chain(avatar: Dict):
    global _CHAINS
    avatar_id = avatar.get("id")
    if avatar_id and avatar_id in _CHAINS:
        return _CHAINS[avatar_id]

    chat_llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        temperature=0.4,
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    parser = StrOutputParser()
    chain = prompt | chat_llm | parser
    if avatar_id:
        _CHAINS[avatar_id] = chain
    return chain


def get_llm_response(user_text: str, avatar: Dict) -> str:
    try:
        chain = _get_chain(avatar)
        return chain.invoke({
            "avatar_id": avatar.get("id"),
            "avatar_name": avatar.get("name"),
            "avatar_persona": avatar.get("persona"),
            "avatar_language": avatar.get("language"),
            "avatar_image": avatar.get("image"),
            "avatar_video": avatar.get("ideal_video"),
            "avatar_voice": avatar.get("voice"),
            "avatar_media_type": avatar.get("media_type"),
            "media_base_url": MEDIA_BASE_URL,
            "user_input": user_text,
        })
    except Exception as e:
        logging.error(f"LLM error: {e}")
        return ""


