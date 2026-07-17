"""외모 관련 질문 판별 (기존 nx/app/vlm/analyze_person.py 로직 복원).

"나 어때 보여?", "내 표정 어때?" 같은 질문이면 카메라 프레임을 VLM으로
분석해서 답하도록 chat.py에서 분기한다.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_APPEARANCE_QUESTIONS = [
    "오늘 내 패션 어때", "나 어때 보여", "헤어스타일 어때", "옷 괜찮아?",
    "내 인상 어때?", "지금 모습 어때?", "오늘 괜찮아 보여?", "나 잘생겼어?",
    "예뻐?", "오늘 어때 보여?", "옷차림 어때?", "얼굴 어때?", "내 분위기 어때?",
    "나 웃고 있어?", "지금 웃는 거야?", "내 표정 어때?", "표정 괜찮아?",
    "몇 살 같아?", "나이 많아 보여?", "나 어려 보여?",
]


def is_appearance_related(user_input: str, threshold: float = 0.5) -> bool:
    try:
        vectorizer = TfidfVectorizer().fit_transform(_APPEARANCE_QUESTIONS + [user_input])
        vectors = vectorizer.toarray()
        sim = cosine_similarity([vectors[-1]], vectors[:-1])
        return sim.max() > threshold
    except Exception:
        return False
