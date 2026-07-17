# analyze_person.py
import re, json, face_recognition, cv2, os, sys, contextlib, logging
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"  
logging.getLogger("insightface").setLevel(logging.ERROR)
import numpy as np
from vlm.api_client import run_vlm
from scipy.spatial.distance import cosine
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

@contextlib.contextmanager
def suppress_stdout():
    """
    Context manager to suppress console output temporarily
    (useful to silence face analysis initialization logs)
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

with suppress_stdout():
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(640, 480))

# === Example questions related to appearance (dummy samples) ===
APPEARANCE_QUESTIONS = [
    "How do I look today?", "What do you think about my style?", "Is my outfit okay?", 
    "Do I look good?", "How is my hairstyle?", "Do I look presentable?"
]

SMILE_QUESTIONS = [
    "Am I smiling?", "How is my expression?", "Do I look happy?", 
    "Do I look friendly?", "Is my face okay?"
]

AGE_QUESTIONS = [
    "How old do I look?", "Do I look young?", "Do I look older?", 
    "Do I look my age?", "What age do you think I look like?"
]

def is_appearance_related(user_input: str, threshold=0.5):
    """Check if user input is related to appearance questions using TF-IDF similarity"""
    vectorizer = TfidfVectorizer().fit_transform(APPEARANCE_QUESTIONS + [user_input])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity([vectors[-1]], vectors[:-1])
    return cosine_sim.max() > threshold

def translate_en_to_ko(text):
    """Translate English → Korean using Google Translate"""
    translator = Translator()
    try:
        return translator.translate(text, src="en", dest="ko").text
    except Exception as e:
        print(f"[❌ Translation Error]: {e}")
        return "Sorry, could you say that again?"
    
# === Clean VLM model responses ===
def clean_response(text: str) -> str:
    text = re.sub(r"assistant\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().lower()

# === Extract only assistant response ===
def extract_assistant_answer(text: str) -> str:
    # Remove tokens and clean line by line
    clean_text = re.sub(r"<\|im_start\|>|<\|im_end\|>", "", text)
    clean_text = clean_text.replace("assistant", "")
    return clean_text.strip()

# === Load prompt map (single structured JSON) ===
with open("config/prompt_map.json", encoding="utf-8") as f:
    PROMPT_MAP = json.load(f)

# === Extract face embedding vector ===
def get_face_embedding(image_path: str):
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            return np.array(encodings[0], dtype=np.float32)  
        else:
            return None 
    except Exception as e:
        print(f"[Embedding Extraction Error]: {e}")
        return None

# ===== Face similarity check =====
def is_same_person(new_vec, old_vec, threshold=0.3, log=True):
    if new_vec is None or old_vec is None:
        return False
    sim = 1 - cosine(np.array(new_vec), np.array(old_vec))
    if log:
        print(f"[ Face Similarity ]: {sim:.3f}")
    return sim > threshold

# === Map response → age group ===
def map_age_group(age_val) -> str:
    try:
        # Convert string to integer
        age_num = int(float(age_val))
    except (ValueError, TypeError):
        print(f"[Conversion Error] age_val={age_val}")
        return "unknown"

    if age_num < 13:
        return "child"
    elif age_num < 20:
        return "teenager"
    elif age_num < 35:
        return "young adult"
    elif age_num < 55:
        return "middle aged"
    else:
        return "elderly"

# === Map response → smile detection ===
def map_smile_en_to_bool(smile_text: str) -> bool:
    text = smile_text.lower().strip()
    positive_keywords = ["yes", "y", "맞아요", "응", "웃", "smile", "happy"]
    return any(word in text for word in positive_keywords)

# ===== Appearance description function =====
def describe_appearance(image_path: str) -> dict:
    """
    Function to generate appearance analysis focusing on clothing description + compliment
    """
    try:
        # Prompt for clothing-centered description
        prompt = "Describe the person's clothing style in detail and say something nice about it."
        vlm_result = run_vlm(image_path, prompts[prompt])
        en_response = extract_assistant_answer(vlm_result["result"])
        ko_response = translate_en_to_ko(en_response or "")

        if not ko_response or not ko_response.strip():
            return {"appearance_comment": "You look comfortable and stylish!"}

        return {"appearance_comment": ko_response}

    except Exception as e:
        print(f"[Appearance Analysis Error]: {e}")
        return {"appearance_comment": "Sorry, there was a problem analyzing the outfit."}

# === Main analysis function ===
def analyze_person(image_path: str, session_id: str = "default") -> dict:
    result = {}
    prompts = PROMPT_MAP
    est_age = "unknown"
    mapped_age = "unknown"
    gender = "unknown"
    is_smiling = False

    # 1. Face detection
    try:
        img = cv2.imread(image_path)
        faces = face_app.get(img)
        if not faces:
            return {"has_person": False}
        face_count = len(faces)
    except Exception as e:
        print(f"[ Face Detection Error ]: {e}")
        return {"has_person": False}

    # 2. Estimate age group
    try:
        est_age = faces[0].age
        mapped_age = map_age_group(est_age)
    except Exception:
        mapped_age = "unknown"
    result["age_group"] = mapped_age
    result["has_person"] = True

    # 3. Gender prediction
    try:
        gender_val = faces[0].sex  
        gender = "male" if gender_val == "M" else "female"
        result["gender"] = gender
    except Exception:
        result["gender"] = "unknown"

    # 4. Smiling check
    try:
        smile_resp = run_vlm(image_path, prompts["is_smiling"])
        s_text = clean_response(extract_assistant_answer(smile_resp.get("result", "")))
        is_smiling = map_smile_en_to_bool(s_text)
        result["is_smiling"] = is_smiling
    except:
        result["is_smiling"] = False

    print(f"[ Persons: {face_count} | Age: {est_age} ({mapped_age}) | Gender: {gender} | Smiling: {'yes' if is_smiling else 'no'} ]")

    # 5. Add face embedding
    face_vector = get_face_embedding(image_path)
    result["face_vector"] = face_vector.tolist() if face_vector is not None else []

    return result