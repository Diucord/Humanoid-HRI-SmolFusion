# test.py

# ===== Py3.8 호환: asyncio.to_thread shim =====
import asyncio, functools, sys, cv2, os, tempfile, time, ctypes, json, uuid, threading
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if not hasattr(asyncio, "to_thread"):
    async def _to_thread(func, /, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
    asyncio.to_thread = _to_thread

import numpy as np
from memory import memory
from scipy.spatial.distance import cosine
from voice.speaker import speak
from dialogue.robot_qa import answer_about_robot
from dialogue.general_chat import general_chat
from vlm.analyze_person import (
    analyze_person,
    translate_en_to_ko,
    is_appearance_related,
)

# ===== API timing utility =====
from performance_monitor import record_api_call

# ===== Settings =====
ROBOT_MODEL = "dummy_model"
face_similarity_threshold = 0.6
analysis_interval = 3  # seconds
fallback_keywords = ["I don't know", "Not sure", "Sorry", "Cannot tell", "Uncertain"]

# ===== Performance tuning parameters =====
MOTION_DIFF_THRESH = 3.5   # average frame difference (0–255) threshold
STILL_FRAME_MIN    = 0.6   # skip analysis if static for this duration (s)
RUN_CUDA_WARMUP    = True  # ultra-light CUDA context warmup (run once at startup)

# ===== State =====
SPOKEN_ONCE = False
last_person_info = None

# ===== Preloaded responses =====
with open("config/dummy_general_responses.json", encoding="utf-8") as f:
    RESPONSES = json.load(f)

# ===== Face similarity =====
def is_same_person(new_vec, old_vec, threshold=face_similarity_threshold):
    if new_vec is None or old_vec is None:
        return False
    sim = 1 - cosine(np.array(new_vec), np.array(old_vec))
    print(f"[Face similarity]: {sim:.3f}")
    return sim > threshold

# ===== Ultra-light CUDA warmup (context only) =====
def _ultra_light_cuda_warmup():
    """
    Initialize CUDA context only, without heavy inference.
    1) If torch.cuda available → lazy_init + synchronize
    2) If failed → try libcudart.cudaFree(0)
    """
    ok = False
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda._lazy_init()
            torch.cuda.synchronize()
            ok = True
    except Exception:
        pass

    if not ok:
        try:
            for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11"):
                try:
                    cudart = ctypes.CDLL(name)
                    if hasattr(cudart, "cudaFree"):
                        cudart.cudaFree.restype = ctypes.c_int
                        cudart.cudaFree.argtypes = [ctypes.c_void_p]
                        cudart.cudaFree(0)  # trigger context creation
                        ok = True
                        break
                except Exception:
                    continue
        except Exception:
            pass

    print(f"[CUDA warmup]: {'OK' if ok else 'Skipped/Failed'}")
    return ok

# ===== analyze_person safe wrapper =====
_analyze_lock = None  # created at runtime

async def safe_analyze(img_path: str):
    """All analyze_person calls go through here (prevents re-entry/conflicts)."""
    global _analyze_lock
    if _analyze_lock is None:
        _analyze_lock = asyncio.Lock()
    async with _analyze_lock:
        return await asyncio.to_thread(analyze_person, img_path)

# ===== Camera handler =====
class CameraHandler:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Cannot open camera.")
        self.window_name = "ROBOT VIEW"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 480, 320)

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        path = os.path.join(tempfile.gettempdir(), f"frame_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(path, frame)
        return frame, path

    def show(self, frame):
        cv2.imshow(self.window_name, frame)

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

# ===== Print conversation log =====
def print_chat_log(session_id):
    history = memory.get(session_id)
    print(f"\n[Conversation Log - Session: {session_id}]")
    for turn in history:
        role = "👤 User" if turn.get("role") == "user" else "🤖 Robot"
        content = turn.get("content", "[❗No content]")
        print(f"{role}: {content}")
    print("")

# ===== Handle user input =====
async def handle_user_input(user_input: str, image_path: str = None, lang: str = "ko", session_id="default"):
    try:
        # For appearance-related questions → run image analysis
        if is_appearance_related(user_input) and image_path:
            print("[Appearance-related question detected → using analyze_person]")
            t0 = time.perf_counter()
            try:
                person_info = await safe_analyze(image_path)
                record_api_call("/analyze", time.perf_counter() - t0, 200)
            except Exception:
                record_api_call("/analyze", time.perf_counter() - t0, 500)
                raise

            comment_en = person_info.get("appearance_comment", "")
            translated = translate_en_to_ko(comment_en)
            print(f"[Appearance response (ko)]: {translated}")
            await speak(translated, lang=lang)

            memory.append(session_id, "user", user_input)
            memory.append(session_id, "assistant", translated)
            print_chat_log(session_id)
            return

        # General chat
        t0 = time.perf_counter()
        try:
            response_obj = general_chat(user_input, model_type=ROBOT_MODEL, session_id=session_id)
            record_api_call("/chat", time.perf_counter() - t0, 200)
        except Exception:
            record_api_call("/chat", time.perf_counter() - t0, 500)
            raise

        response = response_obj.get("text") if isinstance(response_obj, dict) else str(response_obj)
        print(f"[Robot response]: {response}")
        await speak(response, lang=lang)

        memory.append(session_id, "user", user_input)
        memory.append(session_id, "assistant", response)
        print_chat_log(session_id)

    except Exception as e:
        print(f"[❌ Input handling error]: {e}")
        await speak("Sorry, please say that again.", lang=lang)

# ===== Main loop =====
async def main_loop():
    global SPOKEN_ONCE, last_person_info
    print("[🔁 Robot standing by...]")
    camera = None
    session_id = "default"
    dialogue_count = 0

    prev_gray = None
    still_since = 0.0

    def text_input_thread(result_holder):
        result = input("[User question]: ").strip()
        result_holder.append(result)

    try:
        camera = CameraHandler()

        # Run ultra-light CUDA warmup once at startup
        if RUN_CUDA_WARMUP:
            _ultra_light_cuda_warmup()

        last_analysis_time = 0

        while True:
            frame, image_path = camera.read_frame()
            if frame is None:
                await asyncio.sleep(1)
                continue

            camera.show(frame)
            key = cv2.waitKey(1)

            # --- Motion score ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_score = 999.0
            if prev_gray is not None:
                motion_score = cv2.absdiff(gray, prev_gray).mean()
            prev_gray = gray

            now = time.time()
            if motion_score < MOTION_DIFF_THRESH:
                if still_since == 0.0:
                    still_since = now
            else:
                still_since = 0.0

            if now - last_analysis_time > analysis_interval:
                # Skip analysis if scene has been static long enough
                if still_since and (now - still_since) >= STILL_FRAME_MIN:
                    last_analysis_time = now
                else:
                    t0 = time.perf_counter()
                    try:
                        person_info = await safe_analyze(image_path)
                        record_api_call("/analyze", time.perf_counter() - t0, 200)

                        has_person = person_info.get("has_person", False)
                        if not has_person:
                            print("[No person detected]")
                            SPOKEN_ONCE = False
                            last_person_info = None
                            session_id = "default"
                            last_analysis_time = now
                            continue

                        # New person detected
                        if last_person_info is None or not is_same_person(
                            person_info.get("face_vector"), last_person_info.get("face_vector")
                        ):
                            greet_map = RESPONSES["greetings"]["ko"]
                            age_group = person_info.get("age_group", "young adult")
                            gender = person_info.get("gender", "unknown")

                            msg = greet_map.get(age_group, RESPONSES["default"])
                            if isinstance(msg, dict):
                                msg = msg.get(gender, msg.get("unknown", RESPONSES["default"]))

                            print(f"[Greeting]: {msg}")
                            await speak(msg, lang="ko")

                            last_person_info = person_info
                            session_id = f"person_{uuid.uuid4().hex[:8]}"
                            memory.reset(session_id)
                            print(f"[Session reset]: {session_id}")
                            SPOKEN_ONCE = True

                    except Exception as e:
                        record_api_call("/analyze", time.perf_counter() - t0, 500)
                        print(f"[⚠️ Analysis error]: {e}")
                    finally:
                        last_analysis_time = now

            if SPOKEN_ONCE:
                result_holder = []
                input_thread = threading.Thread(target=text_input_thread, args=(result_holder,))
                input_thread.start()

                while input_thread.is_alive():
                    frame, _ = camera.read_frame()
                    if frame is not None:
                        camera.show(frame)
                    if cv2.waitKey(1) == ord("q"):
                        print("[Quit key pressed]")
                        input_thread.join()
                        raise KeyboardInterrupt

                user_input = result_holder[0] if result_holder else ""
                if user_input:
                    await handle_user_input(user_input, image_path, session_id=session_id)
                    dialogue_count += 1
                    if dialogue_count >= 3:
                        print("[✅ 3 dialogues completed → exiting]")
                        break

            if key == ord("q"):
                print("[Quit key pressed]")
                break

    except Exception as e:
        print(f"[❌ Main loop error]: {e}")
    finally:
        if camera:
            camera.cleanup()
        print("[System shutdown]")

# ===== Entry point =====
if __name__ == "__main__":
    asyncio.run(main_loop())