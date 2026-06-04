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
from dialogue.personas import list_personas
from dialogue.chat import respond
from dialogue.tts import synthesize
from vision.analyze import analyze_person, decode_image, is_same_person
from rag.store import get_store

app = FastAPI(title="Humanoid HRI SmolFusion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션별 마지막 얼굴 벡터 (동일인 판단용)
_last_face = {}


@app.get("/personas")
def get_personas():
    return {"personas": list_personas()}


@app.post("/vision/analyze")
async def vision_analyze(
    session_id: str = Form("default"),
    image: UploadFile = File(...),
):
    data = await image.read()
    img = decode_image(data)
    result = analyze_person(img)

    # 동일인 판단
    is_new = False
    if result.get("has_person") and result.get("face_vector") is not None:
        prev = _last_face.get(session_id)
        if prev is None or not is_same_person(result["face_vector"], prev):
            is_new = True
        _last_face[session_id] = result["face_vector"]
    elif not result.get("has_person"):
        _last_face.pop(session_id, None)

    # face_vector는 프론트로 안 보냄 (용량/프라이버시)
    result.pop("face_vector", None)
    result["is_new_person"] = is_new
    return result


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    persona_id: str = "igris"
    custom_prompt: str = ""
    vision_context: str = ""
    use_rag: bool = True


@app.post("/chat")
def chat(req: ChatRequest):
    return respond(
        user_msg=req.message,
        session_id=req.session_id,
        persona_id=req.persona_id,
        custom_prompt=req.custom_prompt,
        vision_context=req.vision_context,
        use_rag=req.use_rag,
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
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok", "device": settings.resolve_device()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
