# voice/speaker.py
import os
import asyncio
import subprocess
import tempfile
from pathlib import Path
import shutil
import edge_tts

# ---- asyncio.to_thread 호환 (Py3.8 지원) ----
try:
    _to_thread = asyncio.to_thread  # Python 3.9+
except AttributeError:
    async def _to_thread(func, *args, **kwargs):  # Python 3.8 fallback
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
# ---------------------------------------------

# ===== 설정 =====
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
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE")             # 있으면 이 보이스를 강제 사용
TTS_TIMEOUT = int(os.getenv("TTS_TIMEOUT", "20"))         # 합성 타임아웃(초)
PLAY_TIMEOUT = int(os.getenv("PLAY_TIMEOUT", "20"))       # 재생 타임아웃(초)

_speak_lock = asyncio.Lock()  # 동시 재생 겹침 방지 락

def _get_voice(lang: str) -> str:
    if EDGE_TTS_VOICE:
        return EDGE_TTS_VOICE
    return VOICE_MAP.get(lang, VOICE_MAP["default"])

def _pick_player() -> list:
    """mpg123 우선, 없으면 ffplay 폴백. 둘 다 없으면 빈 리스트."""
    if shutil.which("mpg123"):
        return ["mpg123", "-q"]
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
    return []

async def _synthesize_to_mp3(text: str, voice: str, out_path: Path):
    tts = edge_tts.Communicate(text, voice=voice)
    await asyncio.wait_for(tts.save(str(out_path)), timeout=TTS_TIMEOUT)

async def speak(text: str, lang: str = "ko"):
    """edge-tts로 합성 → mpg123/ffplay로 재생 (파일 자동 정리, Py3.8~ 호환)"""
    text = (text or "").strip()
    if not text:
        return

    voice = _get_voice(lang)
    player = _pick_player()
    if not player:
        return

    async with _speak_lock:
        fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        mp3_path = Path(tmp_path)

        try:
            # 1) 합성
            await _synthesize_to_mp3(text, voice, mp3_path)
            size = mp3_path.stat().st_size
            if size < 128:
                raise RuntimeError("TTS output too small")

            # 2) 재생 (서브프로세스, 이벤트루프 비블로킹)
            def _play():
                return subprocess.run(
                    player + [str(mp3_path)],
                    check=False,
                    timeout=PLAY_TIMEOUT
                ).returncode
            rc = await _to_thread(_play)

        except asyncio.TimeoutError:
            print("[SPEAK] timeout (synth or play)")
        except Exception as e:
            print(f"[SPEAK] error: {e}")
        finally:
            try:
                mp3_path.unlink(missing_ok=True)
            except Exception:
                pass
