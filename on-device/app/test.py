# test.py
import asyncio, cv2, tempfile, time, json, uuid, threading, os, sys, ctypes, contextlib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from memory import memory
import speech_recognition as sr
from voice.speaker import speak
from scipy.spatial.distance import cosine
from dialogue.general_chat import general_chat
from vlm.analyze_person import (
    analyze_person, is_appearance_related, translate_en_to_ko,
    describe_appearance, get_face_embedding, is_same_person
)
from insightface.app import FaceAnalysis

@contextlib.contextmanager
def suppress_stdout():
    """Suppress stdout temporarily (to silence InsightFace logs)"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

with suppress_stdout():
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(1000, 480))

# ===== Configuration =====
ROBOT_MODEL = "dummy-model"
analysis_interval = 3  # seconds
fallback_keywords = ["I don’t know", "Not sure", "Sorry", "Unknown", "Uncertain"]
SPOKEN_ONCE = False
last_person_info = None
person_present = False
session_id_holder = {"id": "default"}
stop_flag = {"stop": False}
mic = sr.Microphone() 

with open("config/dummy_general_responses.json", encoding="utf-8") as f:
    RESPONSES = json.load(f)

# ===== Shared frame buffer =====
shared_frame = {"frame": None, "path": None}
frame_lock = threading.Lock()

# ===== Camera Handler =====
class CameraHandler:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Could not open camera.")
        self.window_name = "ROBOT VIEW"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)

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

# ===== Print Chat Log =====
def print_chat_log(session_id):
    history = memory.get(session_id)
    print(f"\n[ Chat Log - Session: {session_id} ]")
    for turn in history:
        role = "👤 User" if turn.get("role") == "user" else "🤖 Robot"
        content = turn.get("content", "[❗No Content]")
        print(f"{role}: {content}")
    print("")

# ===== Handle User Input =====
async def handle_user_input(user_input: str, image_path: str = None, lang: str = "ko", session_id="default"):
    try:
        # Process general chat
        response_obj = general_chat(user_input, model_type="dummy_model", session_id=session_id)
        response = response_obj.get("text") if isinstance(response_obj, dict) else str(response)
        
        print(f"[ Robot Response ]: {response}")
        await speak(response, lang=lang)

        memory.append(session_id, "user", user_input)
        memory.append(session_id, "assistant", response)
        print_chat_log(session_id)

    except Exception as e:
        print(f"[❌ Input Handling Exception]: {e}")
        await speak("Sorry, could you please repeat that?", lang=lang)

# ===== Microphone Input Thread =====
def listen_in_thread(result_holder):
    print("Please speak...")
    recognizer = sr.Recognizer()
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio, language="ko-KR")
            print(f"[ Recognized Text ]: {text}")
            result_holder.append(text)

    except Exception as e:
        print(f"[ Microphone Error ]: {e}")
        result_holder.append("")

# ===== Camera Watch Thread =====
def camera_watch_thread(camera, stop_flag):
    global SPOKEN_ONCE, last_person_info, person_present, session_id_holder

    while not stop_flag["stop"]:
        with frame_lock:
            frame = shared_frame["frame"]
            image_path = shared_frame["path"]

        if frame is None:
            time.sleep(0.5)
            continue

        faces = face_app.get(frame)
        if not faces:
            if person_present:
                print("[ No person detected → switching to idle mode... ]")
                SPOKEN_ONCE = False
                last_person_info = None
                session_id_holder["id"] = "default"
            person_present = False
        else:
            person_present = True
            # New person check
            new_vec = get_face_embedding(image_path)
            if new_vec is not None:
                if last_person_info is None:
                    last_person_info = {"face_vector": new_vec}
                else:
                    if not is_same_person(new_vec, last_person_info.get("face_vector"), log=False):
                        print("[ New person detected → session reset ]")
                        last_person_info = {"face_vector": new_vec}
                        session_id_holder["id"] = f"person_{uuid.uuid4().hex[:8]}"
                        memory.reset(session_id_holder["id"])
                        SPOKEN_ONCE = False  # Greeting handled in main loop
        time.sleep(0.5)  # Prevent excessive loop frequency

# ===== Main Loop =====
async def main_loop():
    global SPOKEN_ONCE, last_person_info, person_present, session_id_holder
    print("[ 🔁 Robot waiting... ]")
    camera = None

    try:
        camera = CameraHandler()

        # Start camera watch thread
        threading.Thread(target=camera_watch_thread, args=(camera, stop_flag), daemon=True).start()

        while True:
            frame, image_path = camera.read_frame()
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            with frame_lock:
                shared_frame["frame"] = frame
                shared_frame["path"] = image_path

            camera.show(frame)
            key = cv2.waitKey(1)

            # Greeting when a person first appears
            if person_present and not SPOKEN_ONCE:
                person_info = analyze_person(image_path)
                if person_info.get("has_person", False):
                    greet_map = RESPONSES["greetings"]["ko"]
                    age_group = person_info.get("age_group", "young adult")
                    gender = person_info.get("gender", "unknown")

                    msg = greet_map.get(age_group, RESPONSES["default"])
                    if isinstance(msg, dict):
                        msg = msg.get(gender, msg.get("unknown", RESPONSES["default"]))

                    print(f"[ Greeting ]: {msg}")
                    await speak(msg, lang="ko")

                    last_person_info = person_info
                    session_id_holder["id"] = f"person_{uuid.uuid4().hex[:8]}"
                    memory.reset(session_id_holder["id"])
                    SPOKEN_ONCE = True

            # Conversation loop
            if SPOKEN_ONCE:
                result_holder = []
                input_thread = threading.Thread(target=listen_in_thread, args=(result_holder,), daemon=True)
                input_thread.start()

                while input_thread.is_alive():
                    frame, _ = camera.read_frame()
                    if frame is not None:
                        camera.show(frame)
                    if not person_present:  # Updated by camera_watch_thread
                        try:
                            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                                ctypes.c_long(input_thread.ident), ctypes.py_object(SystemExit)
                            )
                        except Exception:
                            pass
                        SPOKEN_ONCE = False
                        break
                    if cv2.waitKey(1) == ord("q"):
                        print("[ Quit key pressed ]")
                        input_thread.join()
                        raise KeyboardInterrupt

                if not SPOKEN_ONCE:  # Conversation ends if no person present
                    continue

                user_input = result_holder[0] if result_holder else ""
                if user_input:
                    await handle_user_input(user_input, image_path, session_id=session_id_holder["id"])

            if key == ord("q"):
                print("[ Quit key pressed ]")
                break

    except Exception as e:
        print(f"[❌ Main Loop Exception]: {e}")
    finally:
        stop_flag["stop"] = True
        if camera:
            camera.cleanup()
        print("[ ROBOT system shutting down. ]")

# ===== Entry Point =====
if __name__ == "__main__":
    asyncio.run(main_loop())