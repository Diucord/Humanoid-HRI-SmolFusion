# voice/speaker.py
import os, shutil, edge_tts, asyncio, subprocess, tempfile
from pathlib import Path

# ---- asyncio.to_thread compatibility (Python 3.8+) ----
try:
    _to_thread = asyncio.to_thread  # Python 3.9+
except AttributeError:
    async def _to_thread(func, *args, **kwargs):  # Python 3.8 fallback
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

# ===== Config =====
VOICE_MAP = {
    "ko": "ko-KR-InJoonNeural",
    "en": "en-US-GuyNeural",
    "ja": "ja-JP-KeitaNeural",
    "zh": "zh-CN-YunfengNeural",
    "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-HenriNeural",
    "de": "de-DE-ConradNeural",
    "default": "en-US-GuyNeural",
}
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE")      # force voice if set
TTS_TIMEOUT = int(os.getenv("TTS_TIMEOUT", "15")) # synthesis timeout (sec)
PLAY_TIMEOUT = int(os.getenv("PLAY_TIMEOUT", "15"))# playback timeout (sec)

_speak_lock = asyncio.Lock()  # prevent overlapping playback

def _get_voice(lang: str) -> str:
    if EDGE_TTS_VOICE:
        return EDGE_TTS_VOICE
    return VOICE_MAP.get(lang, VOICE_MAP["default"])

def _pick_player() -> list:
    """
    Prefer mpg123, fallback to ffplay. Return [] if neither is available.
    """
    if shutil.which("mpg123"):
        return ["mpg123", "-q"]
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
    return []

async def _synthesize_to_mp3(text: str, voice: str, out_path: Path):
    tts = edge_tts.Communicate(text, voice=voice)
    await asyncio.wait_for(tts.save(str(out_path)), timeout=TTS_TIMEOUT)

async def speak(text: str, lang: str = "en"):
    """
    Synthesize speech with edge-tts → Play via mpg123/ffplay.
    Temporary files auto-cleaned. Compatible with Python 3.8+.
    """
    text = (text or "").strip()
    if not text:
        return

    voice = _get_voice(lang)
    player = _pick_player()
    if not player:
        print("[SPEAK] no audio player found (install mpg123 or ffplay)")
        return

    async with _speak_lock:
        fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        mp3_path = Path(tmp_path)

        try:
            # 1) Synthesize
            await _synthesize_to_mp3(text, voice, mp3_path)
            if mp3_path.stat().st_size < 128:
                raise RuntimeError("TTS output too small")

            # 2) Play (blocking subprocess, offloaded to thread)
            def _play():
                return subprocess.run(
                    player + [str(mp3_path)],
                    check=False,
                    timeout=PLAY_TIMEOUT
                ).returncode
            rc = await _to_thread(_play)

        except asyncio.TimeoutError:
            print("[SPEAK] timeout (synthesis or playback)")
        except Exception as e:
            print(f"[SPEAK] error: {e}")
        finally:
            try:
                mp3_path.unlink(missing_ok=True)
            except Exception:
                pass
