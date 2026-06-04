"""TTS — edge-tts로 mp3 바이트 생성 (웹 재생용).

기존 voice/speaker.py의 VOICE_MAP을 재활용하되, 파일 재생(mpg123) 대신
브라우저가 재생할 수 있도록 mp3 바이트를 반환한다.
"""
import edge_tts

VOICE_MAP = {
    "ko": "ko-KR-InJoonNeural",
    "en": "en-US-GuyNeural",
    "ja": "ja-JP-KeitaNeural",
    "zh": "zh-CN-YunfengNeural",
    "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-HenriNeural",
    "de": "de-DE-ConradNeural",
    "default": "ko-KR-InJoonNeural",
}


async def synthesize(text: str, voice: str = None, lang: str = "ko") -> bytes:
    """텍스트 → mp3 바이트."""
    text = (text or "").strip()
    if not text:
        return b""
    v = voice or VOICE_MAP.get(lang, VOICE_MAP["default"])
    communicate = edge_tts.Communicate(text, voice=v)
    chunks = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.extend(chunk["data"])
    return bytes(chunks)
