# server.py 

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil, os, uuid, time
from app.smolvlm_infer import run_smolvlm
from app.llm_chat import generate_response_llm, ChatConfig

app = FastAPI()

# ===== 기본 분석 API ===== 
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    decode: str = Form("true")
):
    start = time.time()
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_filename, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = run_smolvlm(
            image_path=temp_filename,
            prompt=prompt,
            decode=decode.lower() == "true"
        )
        return result

    except Exception as e:
        print(f"[❌ analyze 예외]: {e}")
        return JSONResponse(status_code=500, content={"error": "내부 오류"})

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        print(f"[INFO] /analyze 응답 시간: {time.time() - start:.2f}초")


# ===== LLM 텍스트 응답 API ===== 
class ChatRequest(BaseModel):
    prompt: str
    session_id: str = "default"
    max_tokens: int = 128
    persona: str = "igris-C"
    language: str = "ko"

@app.post("/chat")
async def chat(request: ChatRequest):
    start = time.time()
    try:
        config = ChatConfig(
            prompt=request.prompt,
            persona=request.persona,
            max_tokens=request.max_tokens,
            session_id=request.session_id,
            language=request.language
        )
        response = generate_response_llm(config)
        return {"response": response}

    except Exception as e:
        print(f"[❌ chat 예외]: {e}")
        return JSONResponse(status_code=500, content={"error": "LLM 오류"})

    finally:
        print(f"[INFO] /chat 응답 시간: {time.time() - start:.2f}초")


# ===== 서버 실행 ===== 
if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)
