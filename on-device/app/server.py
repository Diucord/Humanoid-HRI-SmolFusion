# server.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil, os, uuid, time, uvicorn, argparse
from app.vlm_infer import run_vlm
from app.llm_chat import generate_response_llm, ChatConfig

app = FastAPI()

# ===== Basic Image Analysis API ===== 
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    decode: str = Form("true")
):
    """
    Endpoint for analyzing an image using VLM.
    Accepts an uploaded image and a prompt, then returns the VLM result.
    """
    start = time.time()
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_filename, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = run_vlm(
            image_path=temp_filename,
            prompt=prompt,
            decode=decode.lower() == "true"
        )
        return result

    except Exception as e:
        print(f"[❌ analyze exception]: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal error"})

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        print(f"[INFO] /analyze response time: {time.time() - start:.2f} sec")


# ===== LLM Text Response API ===== 
class ChatRequest(BaseModel):
    prompt: str
    session_id: str = "default"
    max_tokens: int = 128
    persona: str = "dummy_model"
    language: str = "ko"

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Endpoint for generating text responses from the LLM.
    Accepts a prompt and configuration, then returns the LLM output.
    """
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
        print(f"[❌ chat exception]: {e}")
        return JSONResponse(status_code=500, content={"error": "LLM error"})

    finally:
        print(f"[INFO] /chat response time: {time.time() - start:.2f} sec")


# ===== Server Entry Point ===== 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)
