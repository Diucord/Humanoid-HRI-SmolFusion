import os
import edge_tts

# 언어별 보이스 설정
VOICE_MAP = {
    "ko": "ko-KR-InJoonNeural",    
    "en": "en-US-GuyNeural",        # 부드러운 중저음 남성
    "ja": "ja-JP-KeitaNeural",      # 자연스러운 남성 톤
    "zh": "zh-CN-YunfengNeural",    # 또렷한 중국어 남성
    "es": "es-ES-AlvaroNeural",     # 스페인어 남성
    "fr": "fr-FR-HenriNeural",      # 프랑스어 남성
    "de": "de-DE-ConradNeural",    # 독일어 남성
    "default": "en-US-GuyNeural"
}

# 언어 기반 TTS 음성 선택
def get_voice(lang_code="ko"):
    return VOICE_MAP.get(lang_code, VOICE_MAP["default"])

# 음성 출력 함수
async def speak(text: str, lang: str = "ko"):
    voice = get_voice(lang)
    tts = edge_tts.Communicate(text, voice)
    await tts.save("output.mp3")
    os.system("mpg123 output.mp3 > /dev/null 2>&1")
    os.remove("output.mp3")
