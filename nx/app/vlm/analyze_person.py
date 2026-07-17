import re
import json
import face_recognition  
import numpy as np
from vlm.api_client import run_smolvlm
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 외모 관련 질문 예시
APPEARANCE_QUESTIONS = [
    "오늘 내 패션 어때", "나 어때 보여", "헤어스타일 어때", "옷 괜찮아?", "내 인상 어때?", "지금 모습 어때?",
    "오늘 괜찮아 보여?", "나 잘생겼어?", "예뻐?", "오늘 어때 보여?", "옷차림 어때?", "얼굴 어때?", "내 분위기 어때?"
]
SMILE_QUESTIONS = ["나 웃고 있어?", "지금 웃는 거야?", "내 표정 어때?", "표정 괜찮아?", "인상 좋아 보여?"]
AGE_QUESTIONS = ["몇 살 같아?", "나이 많아 보여?", "나 어려 보여?", "젊어 보이나요?", "늙어 보여?"]

def is_appearance_related(user_input: str, threshold=0.5):
    vectorizer = TfidfVectorizer().fit_transform(APPEARANCE_QUESTIONS + [user_input])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity([vectors[-1]], vectors[:-1])
    return cosine_sim.max() > threshold


def translate_en_to_ko(text):
    translator = Translator()
    try:
        return translator.translate(text, src="en", dest="ko").text
    except Exception as e:
        print(f"[❌ 번역 오류]: {e}")
        return "죄송합니다. 다시 말씀해 주세요."
    

# === VLM 응답 텍스트 정제 ===
def clean_response(text: str) -> str:
    text = text.strip()
    text = re.sub(r'[^\w\s-]', '', text)  # 특수문자 제거
    return text.lower().strip()


# === Assistant 응답 부분만 추출 ===
def extract_assistant_answer(text: str) -> str:
    match = re.search(r"assistant\s*:\s*(.+)", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


# === 단일 구조 프롬프트 맵 로딩 ===
with open("config/prompt_map.json", encoding="utf-8") as f:
    PROMPT_MAP = json.load(f)


# === 얼굴 임베딩 벡터 추출 ===
def get_face_embedding(image_path: str):
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        return encodings[0] if encodings else None
    except Exception as e:
        print(f"[임베딩 추출 오류]: {e}")
        return None


# === 응답 → 연령대 매핑 ===
def map_age_group(age_text: str) -> str:
    exact_map = {
        "child": "child",
        "teenager": "teenager",
        "young adult": "young adult",
        "middle aged": "middle aged",
        "elderly": "elderly"
    }
    return exact_map.get(age_text, "unknown")


# === 응답 → 성별 매핑 ===
def map_gender(g_text: str) -> str:
    return g_text if g_text in ["male", "female"] else "unknown"


# === 응답 → 웃음 여부 매핑 ===
def map_smile_en_to_bool(smile_text: str) -> bool:
    return smile_text == "yes"


# ===== 외모 묘사 함수 추가 =====
def describe_appearance(image_path: str) -> str:
    try:
        prompt = PROMPT_MAP.get("describe_appearance")
        if not prompt:
            return "죄송해요, 외모 묘사를 위한 프롬프트가 설정되어 있지 않아요."

        vlm_result = run_smolvlm(image_path, prompt)
        en_response = extract_assistant_answer(vlm_result.get("result", ""))

        # 번역
        ko_response = translate_en_to_ko(en_response)
        return ko_response

    except Exception as e:
        print(f"[외모 분석 오류]: {e}")
        return "죄송해요, 외모를 분석하는 데 문제가 발생했어요."
    

# === 메인 분석 함수 ===
def analyze_person(image_path: str) -> dict:
    result = {}
    prompts = PROMPT_MAP  # 단일 구조 사용

    # 1. 얼굴 추출
    face_vector = get_face_embedding(image_path)
    if face_vector is None:
        print("[얼굴 임베딩 실패 → 사람 없음 처리]")
        return {"has_person": False}

    # 2. 연령대 판단
    try:
        age_resp = run_smolvlm(image_path, prompts["estimate_age_group"])
        age_text_raw = age_resp.get("result", "")
        age_answer = extract_assistant_answer(age_text_raw)
        age_text = clean_response(age_answer)

        if age_text in ["no", "none", "not found"]:
            print("[사람 없음 판단됨]")
            return {"has_person": False}

        result["age_group"] = map_age_group(age_text)
        print(f"[연령대 판단]: {result['age_group']}")
    except Exception as e:
        print(f"[VLM 오류 - 연령대]: {e}")
        return {"has_person": False}

    result["has_person"] = True

    # 3. 성별 판단
    try:
        gender_resp = run_smolvlm(image_path, prompts["check_gender"])
        g_text_raw = gender_resp.get("result", "")
        g_answer = extract_assistant_answer(g_text_raw)
        g_text = clean_response(g_answer)

        result["gender"] = map_gender(g_text)
        print(f"[성별 판단]: {result['gender']}")
    except Exception as e:
        print(f"[VLM 오류 - 성별]: {e}")
        result["gender"] = "unknown"

    # 4. 웃고 있는지 판단
    try:
        smile_resp = run_smolvlm(image_path, prompts["is_smiling"])
        s_text_raw = smile_resp.get("result", "")
        s_answer = extract_assistant_answer(s_text_raw)
        smile_text = clean_response(s_answer)

        result["is_smiling"] = map_smile_en_to_bool(smile_text)
        print(f"[웃고 있음 여부]: {'yes' if result['is_smiling'] else 'no'}")
    except Exception as e:
        print(f"[VLM 오류 - 웃음]: {e}")
        result["is_smiling"] = False

    # 5. 얼굴 벡터 저장
    result["face_vector"] = face_vector.tolist()

    # # 6. 외모 묘사 생성
    # try:
    #     result["appearance_comment"] = describe_appearance(image_path)
    #     print(f"[외모 묘사]: {result['appearance_comment']}")
    # except Exception as e:
    #     print(f"[VLM 오류 - 외모]: {e}")
    #     result["appearance_comment"] = "unknown"

    return result
