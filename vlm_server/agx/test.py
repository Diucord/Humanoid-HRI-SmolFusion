# test.py

# ===== Py3.8 호환: asyncio.to_thread shim =====
import asyncio, functools
if not hasattr(asyncio, "to_thread"):
    async def _to_thread(func, /, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
    asyncio.to_thread = _to_thread

import cv2, os, tempfile, time, ctypes, json, uuid, threading
import numpy as np
from scipy.spatial.distance import cosine

from voice.speaker import speak
from dialogue.robot_qa import answer_about_robot
from dialogue.general_chat import general_chat
from vlm.analyze_person import (
    analyze_person,
    translate_en_to_ko,
    is_appearance_related,
)
from memory import memory

# API 타이밍 기록 유틸
from performance_monitor import record_api_call

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ===== 설정 =====
ROBOT_MODEL = "igris-C"
face_similarity_threshold = 0.6
analysis_interval = 3  # 초
fallback_keywords = ["모르겠어요", "잘 모르겠어요", "죄송합니다", "알 수 없어요", "확실하지 않아요"]

# ===== 성능 튜닝 파라미터 =====
MOTION_DIFF_THRESH = 3.5   # 프레임 차 평균(0~255) 기준
STILL_FRAME_MIN    = 0.6   # 정적 지속 시간(초) 이상이면 분석 스킵
RUN_CUDA_WARMUP    = True  # 초경량 CUDA 컨텍스트 워밍업(시작 시 1회)

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

# ===== 초경량 CUDA 워밍업 (컨텍스트만 생성) =====
def _ultra_light_cuda_warmup():
    """
    분석 호출 없이 CUDA 컨텍스트만 초기화.
    1) torch.cuda 가능하면 lazy_init + synchronize
    2) 실패 시 libcudart.cudaFree(0) 시도
    """
    ok = False
    try:
        import torch
        if torch.cuda.is_available():
            # 아주 가벼운 컨텍스트 초기화
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
                        cudart.cudaFree(0)  # 컨텍스트 생성 트리거
                        ok = True
                        break
                except Exception:
                    continue
        except Exception:
            pass

    print(f"[CUDA 워밍업]: {'OK' if ok else '건너뜀/실패'}")
    return ok

# ===== analyze_person 직렬화 래퍼 =====
_analyze_lock = None  # 런타임에 생성

async def safe_analyze(img_path: str):
    """모든 analyze_person 호출은 이 함수로만 (비재진입/충돌 방지)."""
    global _analyze_lock
    if _analyze_lock is None:
        _analyze_lock = asyncio.Lock()
    async with _analyze_lock:
        return await asyncio.to_thread(analyze_person, img_path)

# ===== 카메라 핸들러 =====
class CameraHandler:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("❌ 카메라를 열 수 없습니다.")
        self.window_name = "로봇 시야"
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
        # 외모 질문에만 이미지 분석
        if is_appearance_related(user_input) and image_path:
            print("[외모 관련 질문 감지됨 → analyze_person 사용]")
            t0 = time.perf_counter()
            try:
                person_info = await safe_analyze(image_path)
                record_api_call("/analyze", time.perf_counter() - t0, 200)
            except Exception:
                record_api_call("/analyze", time.perf_counter() - t0, 500)
                raise

            comment_en = person_info.get("appearance_comment", "")
            translated = translate_en_to_ko(comment_en)
            print(f"[외모 응답 (ko)]: {translated}")
            await speak(translated, lang=lang)

            memory.append(session_id, "user", user_input)
            memory.append(session_id, "assistant", translated)
            print_chat_log(session_id)
            return

        # 일반 질문 처리
        t0 = time.perf_counter()
        try:
            response_obj = general_chat(user_input, model_type=ROBOT_MODEL, session_id=session_id)
            record_api_call("/chat", time.perf_counter() - t0, 200)
        except Exception:
            record_api_call("/chat", time.perf_counter() - t0, 500)
            raise

        response = response_obj.get("text") if isinstance(response_obj, dict) else str(response_obj)
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
    dialogue_count = 0

    prev_gray = None
    still_since = 0.0

    def text_input_thread(result_holder):
        result = input("[사용자 질문]: ").strip()
        result_holder.append(result)

    try:
        camera = CameraHandler()

        # ⬇시작 시 단 1회: 초경량 CUDA 컨텍스트 워밍업
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

            # --- 모션 스코어 ---
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
                # 정적이면 이번 라운드 분석 스킵 (지속 STILL_FRAME_MIN 이상)
                if still_since and (now - still_since) >= STILL_FRAME_MIN:
                    last_analysis_time = now
                else:
                    t0 = time.perf_counter()
                    try:
                        person_info = await safe_analyze(image_path)
                        record_api_call("/analyze", time.perf_counter() - t0, 200)

                        has_person = person_info.get("has_person", False)
                        if not has_person:
                            print("[사람 없음]")
                            SPOKEN_ONCE = False
                            last_person_info = None
                            session_id = "default"
                            last_analysis_time = now
                            continue

                        # 새 사람 또는 다른 사람
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

                    except Exception as e:
                        record_api_call("/analyze", time.perf_counter() - t0, 500)
                        print(f"[⚠️ 분석 오류]: {e}")
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
