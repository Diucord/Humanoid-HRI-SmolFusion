# speaker.py
import os, edge_tts

# Voice settings by language
VOICE_MAP = {
    "ko": "ko-KR-InJoonNeural",     # Korean (male voice)
    "en": "en-US-GuyNeural",        # Smooth mid-low male tone
    "ja": "ja-JP-KeitaNeural",      # Natural male tone in Japanese
    "zh": "zh-CN-YunfengNeural",    # Clear Mandarin Chinese male voice
    "es": "es-ES-AlvaroNeural",     # Spanish male voice
    "fr": "fr-FR-HenriNeural",      # French male voice
    "de": "de-DE-ConradNeural",     # German male voice
    "default": "en-US-GuyNeural"
}

def get_voice(lang_code: str = "ko") -> str:
    """
    Select a TTS voice based on language code.
    
    Args:
        lang_code (str): Language code (e.g., "en", "ko")
    
    Returns:
        str: Corresponding voice ID
    """
    return VOICE_MAP.get(lang_code, VOICE_MAP["default"])

async def speak(text: str, lang: str = "ko"):
    """
    Generate speech from text and play it using mpg123.
    
    Args:
        text (str): The text to convert to speech
        lang (str): Language code (default = "ko")
    """
    voice = get_voice(lang)
    tts = edge_tts.Communicate(text, voice)

    # Save TTS to temporary file
    await tts.save("output.mp3")
    
    # Play audio (suppress terminal output)
    os.system("mpg123 output.mp3 > /dev/null 2>&1")
    
    # Remove file after playback
    os.remove("output.mp3")