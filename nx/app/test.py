import asyncio
import cv2
import os
import tempfile
import time
import numpy as np
import json
import uuid
import threading

from voice.speaker import speak
from dialogue.robot_qa import answer_about_robot
from dialogue.general_chat import general_chat
from vlm.analyze_person import (
    analyze_person, 
    translate_en_to_ko, 
    is_appearance_related
)
from scipy.spatial.distance import cosine
from memory import memory
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ===== 설정 =====
ROBOT_MODEL = "igris-C"
face_similarity_threshold = 0.6
analysis_interval = 3  # 초
fallback_keywords = ["모르겠어요", "잘 모르겠어요", "죄송합니다", "알 수 없어요", "확실하지 않아요"]


# ===== 상태 =====
SPOKEN_ONCE = False
last_person_info = None


# ===== 응답 로딩 =====
with open("config/general_responses.json", encoding="utf-8") as f:
    RESPONSES = json.load(f)


# ===== 얼굴 유사도 =====
def is_same_person(new_vec, old_vec, threshold=face_similarity_threshold):
    if new_vec is None or old_vec is None:
        return False
    sim = 1 - cosine(np.array(new_vec), np.array(old_vec))
    print(f"[얼굴 유사도]: {sim:.3f}")
    return sim > threshold


# ===== 카메라 핸들러 =====
class CameraHandler:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("❌ 카메라를 열 수 없습니다.")
        self.window_name = "로봇 시야"
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


# ===== 대화 로그 프린트 =====
def print_chat_log(session_id):
    history = memory.get(session_id)
    print(f"\n[대화 로그 - 세션: {session_id}]")
    for turn in history:
        role = "👤 사용자" if turn.get("role") == "user" else "🤖 로봇"
        content = turn.get("content", "[❗내용 없음]")
        print(f"{role}: {content}")
    print("")


# ===== 입력 처리 =====
async def handle_user_input(user_input: str, image_path: str = None, lang: str = "ko", session_id="default"):
    try:
        # 외모 관련 질문일 경우에만 이미지 분석 수행
        if is_appearance_related(user_input) and image_path:
            print("[외모 관련 질문 감지됨 → analyze_person 사용]")
            person_info = analyze_person(image_path)
            comment_en = person_info.get("appearance_comment", "")
            translated = translate_en_to_ko(comment_en)
            print(f"[외모 응답 (ko)]: {translated}")
            await speak(translated, lang=lang)

            memory.append(session_id, "user", user_input)
            memory.append(session_id, "assistant", translated)
            print_chat_log(session_id)
            return

        # 일반 질문 처리 (train.json 기반 유사 응답)
        response_obj = general_chat(user_input, model_type=ROBOT_MODEL, session_id=session_id)
        response = response_obj.get("text") if isinstance(response_obj, dict) else str(response)

        print(f"[로봇 응답]: {response}")
        await speak(response, lang=lang)

        memory.append(session_id, "user", user_input)
        memory.append(session_id, "assistant", response)
        print_chat_log(session_id)

    except Exception as e:
        print(f"[❌ 입력 처리 예외]: {e}")
        await speak("죄송합니다. 다시 말씀해 주세요.", lang=lang)


# ===== 메인 루프 =====
async def main_loop():
    global SPOKEN_ONCE, last_person_info
    print("[🔁 로봇 대기 중...]")
    camera = None
    session_id = "default"
    dialogue_count = 0  # 대화 횟수 카운트 추가

    def text_input_thread(result_holder):
        result = input("[사용자 질문]: ").strip()
        result_holder.append(result)

    try:
        camera = CameraHandler()
        last_analysis_time = 0

        while True:
            frame, image_path = camera.read_frame()
            if frame is None:
                await asyncio.sleep(1)
                continue

            camera.show(frame)
            key = cv2.waitKey(1)

            now = time.time()
            if now - last_analysis_time > analysis_interval:
                try:
                    person_info = analyze_person(image_path)
                    has_person = person_info.get("has_person", False)

                    if not has_person:
                        print("[사람 없음]")
                        SPOKEN_ONCE = False
                        last_person_info = None
                        session_id = "default"
                        last_analysis_time = now
                        continue

                    if last_person_info is None or not is_same_person(
                        person_info.get("face_vector"), last_person_info.get("face_vector")
                    ):
                        greet_map = RESPONSES["greetings"]["ko"]
                        age_group = person_info.get("age_group", "young adult")
                        gender = person_info.get("gender", "unknown")

                        msg = greet_map.get(age_group, RESPONSES["default"])
                        if isinstance(msg, dict):
                            msg = msg.get(gender, msg.get("unknown", RESPONSES["default"]))

                        print(f"[인사]: {msg}")
                        await speak(msg, lang="ko")

                        last_person_info = person_info
                        session_id = f"person_{uuid.uuid4().hex[:8]}"
                        memory.reset(session_id)
                        print(f"[세션 초기화]: {session_id}")
                        SPOKEN_ONCE = True

                    last_analysis_time = now

                except Exception as e:
                    print(f"[⚠️ 분석 오류]: {e}")

            if SPOKEN_ONCE:
                result_holder = []
                input_thread = threading.Thread(target=text_input_thread, args=(result_holder,))
                input_thread.start()

                while input_thread.is_alive():
                    frame, _ = camera.read_frame()
                    if frame is not None:
                        camera.show(frame)
                    if cv2.waitKey(1) == ord("q"):
                        print("[종료 키 입력됨]")
                        input_thread.join()
                        raise KeyboardInterrupt

                user_input = result_holder[0] if result_holder else ""

                if user_input:
                    await handle_user_input(user_input, image_path, session_id=session_id)
                    dialogue_count += 1
                    if dialogue_count >= 3:
                        print("[✅ 대화 3회 완료 → 종료]")
                        break

            if key == ord("q"):
                print("[종료 키 입력됨]")
                break

    except Exception as e:
        print(f"[❌ 메인 루프 예외]: {e}")
    finally:
        if camera:
            camera.cleanup()
        print("[시스템 종료]")

# ===== 실행 =====
if __name__ == "__main__":
    asyncio.run(main_loop())
