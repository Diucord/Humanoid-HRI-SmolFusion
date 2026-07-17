"""FastAPI 메인 앱.

엔드포인트:
  GET  /personas            페르소나 목록
  POST /vision/analyze      카메라 프레임 → 얼굴매칭 + 나이/성별/표정/장면
  POST /chat                페르소나 + RAG + Vision 컨텍스트 → 응답
  POST /rag/upload          문서 업로드 → 페르소나 지식베이스
  POST /rag/clear           지식베이스 초기화
  POST /tts                 텍스트 → mp3
  POST /session/reset       세션 초기화
"""
import io
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from core import settings
from core.memory import memory
from dialogue.personas import list_personas, create_persona, delete_persona
from dialogue.chat import respond
from dialogue.tts import synthesize
from dialogue.greeting import make_greeting
from dialogue.appearance import is_appearance_related
from vision.analyze import analyze_person, decode_image, is_same_person
from vision.vlm import analyzer as vlm_analyzer
from rag.store import get_store

app = FastAPI(title="Hera — Human-robot Engagement Responsive AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    # localhost(dev) + Vercel(*.vercel.app) + Cloudflare(*.trycloudflare.com) 허용
    allow_origin_regex=(
        r"http://(localhost|127\.0\.0\.1):\d+"
        r"|https://([a-z0-9-]+\.)*vercel\.app"
        r"|https://([a-z0-9-]+\.)*trycloudflare\.com"
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션별 마지막 얼굴 벡터 (동일인 판단용)
_last_face = {}
# 세션별 직전 사람 존재 여부 (얼굴 임베딩 없을 때 등장 판단용)
_last_present = {}
# 세션별 최근 카메라 프레임 (외모 질문 응답용 PIL 이미지)
_last_frame = {}
# 세션별 직전 (나이대, 성별) — 얼굴 임베딩 없을 때 사람 변화 감지용
_last_demo = {}


@app.get("/personas")
def get_personas():
    return {"personas": list_personas()}


class CreatePersonaRequest(BaseModel):
    name: str
    system_prompt: str = ""
    traits: dict | None = None
    voice: str = "ko-KR-InJoonNeural"
    language: str = "ko"


@app.post("/personas")
def create_persona_endpoint(req: CreatePersonaRequest):
    p = create_persona(
        name=req.name,
        system_prompt=req.system_prompt,
        traits=req.traits,
        voice=req.voice,
        language=req.language,
    )
    return {"ok": True, "persona": p}


@app.delete("/personas/{persona_id}")
def delete_persona_endpoint(persona_id: str):
    ok = delete_persona(persona_id)
    return {"ok": ok}


@app.post("/vision/analyze")
async def vision_analyze(
    session_id: str = Form("default"),
    manual: str = Form("false"),  # 사용자가 직접 올린 이미지면 항상 새 사람으로
    image: UploadFile = File(...),
):
    is_manual = manual.lower() == "true"
    data = await image.read()
    img = decode_image(data)
    _last_frame[session_id] = img  # 외모 질문 응답용 보관
    result = analyze_person(img)

    # 동일인 판단
    #  - 얼굴 임베딩이 있으면: 얼굴 매칭으로 새 사람 판단 (정확)
    #  - 임베딩이 없지만 사람이 있으면: '사람 없음→있음' 전환을 새 사람으로 간주
    is_new = False
    has_person = result.get("has_person", False)
    face_vec = result.get("face_vector")

    demo = (result.get("age_group", "unknown"), result.get("gender", "unknown"))

    # 사용자가 직접 올린 이미지(테스트)면 사람 있을 때 항상 새 사람으로 인사
    if is_manual and has_person:
        is_new = True
        if face_vec is not None:
            _last_face[session_id] = face_vec
        _last_present[session_id] = True
        _last_demo[session_id] = demo
    elif has_person and face_vec is not None:
        prev = _last_face.get(session_id)
        if prev is None or not is_same_person(face_vec, prev):
            is_new = True
        _last_face[session_id] = face_vec
        _last_present[session_id] = True
        _last_demo[session_id] = demo
    elif has_person:  # 사람은 있는데 얼굴 임베딩 실패
        prev_demo = _last_demo.get(session_id)
        if not _last_present.get(session_id, False):
            is_new = True  # 직전에 사람이 없었으면 새로 등장
        elif prev_demo is not None and demo != prev_demo and demo != ("unknown", "unknown"):
            is_new = True  # 나이대/성별이 바뀌면 다른 사람으로 간주
        _last_present[session_id] = True
        _last_demo[session_id] = demo
    else:  # 사람 없음
        _last_face.pop(session_id, None)
        _last_present[session_id] = False
        _last_demo.pop(session_id, None)

    # face_vector는 프론트로 안 보냄 (용량/프라이버시)
    result.pop("face_vector", None)
    result["is_new_person"] = is_new

    # 새 사람 감지 시 연령/성별 기반 인사말 생성
    result["greeting"] = ""
    if is_new and result.get("has_person"):
        result["greeting"] = make_greeting(
            age_group=result.get("age_group", "young adult"),
            gender=result.get("gender", "unknown"),
        )
    return result


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    persona_id: str = "igris"
    custom_prompt: str = ""
    vision_context: str = ""
    use_rag: bool = True
    traits: dict | None = None


@app.post("/chat")
def chat(req: ChatRequest):
    # 외모 관련 질문이면(igris) 최근 카메라 프레임을 VLM으로 묘사해 답변
    if req.persona_id == "igris" and is_appearance_related(req.message):
        frame = _last_frame.get(req.session_id)
        if frame is not None:
            comment = vlm_analyzer.describe_appearance(frame)
            memory.append(req.session_id, "user", req.message)
            memory.append(req.session_id, "assistant", comment)
            return {"source": "appearance", "text": comment}

    return respond(
        user_msg=req.message,
        session_id=req.session_id,
        persona_id=req.persona_id,
        custom_prompt=req.custom_prompt,
        vision_context=req.vision_context,
        use_rag=req.use_rag,
        traits=req.traits,
    )


@app.post("/rag/upload")
async def rag_upload(
    persona_id: str = Form("igris"),
    files: List[UploadFile] = File(...),
):
    store = get_store(persona_id)
    total_chunks = 0
    names = []
    for f in files:
        data = await f.read()
        total_chunks += store.add_file(f.filename, data)
        names.append(f.filename)
    return {"ok": True, "files": names, "chunks_added": total_chunks, "total_chunks": store.count()}


class RagClearRequest(BaseModel):
    persona_id: str = "igris"


@app.post("/rag/clear")
def rag_clear(req: RagClearRequest):
    get_store(req.persona_id).clear()
    return {"ok": True}


class TTSRequest(BaseModel):
    text: str
    lang: str = "ko"
    voice: Optional[str] = None


@app.post("/tts")
async def tts(req: TTSRequest):
    audio = await synthesize(req.text, voice=req.voice, lang=req.lang)
    return Response(content=audio, media_type="audio/mpeg")


class SessionResetRequest(BaseModel):
    session_id: str = "default"


@app.post("/session/reset")
def session_reset(req: SessionResetRequest):
    memory.reset(req.session_id)
    _last_face.pop(req.session_id, None)
    _last_present.pop(req.session_id, None)
    _last_frame.pop(req.session_id, None)
    _last_demo.pop(req.session_id, None)
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok", "device": settings.resolve_device()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
