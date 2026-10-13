import os
import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
INDEX_FILE = ROOT / "index.html"

app = FastAPI(title="LayerFox AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return FileResponse(INDEX_FILE)

@app.get("/health")
async def health():
    return {"ok": True, "service": "LayerFox AI", "ai_configured": bool(os.getenv("HF_API_NAME"))}

@app.post("/api/process")
async def process_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="ارفع ملف صورة فقط.")
    raw = await file.read()
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="حجم الصورة أكبر من 12 ميغابايت.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="تعذّر قراءة الصورة. جرّب PNG أو JPG.")

    space_id = os.getenv("HF_SPACE_ID", "").strip()
    api_name = os.getenv("HF_API_NAME", "").strip()
    if not space_id or not api_name:
        return JSONResponse(
            status_code=503,
            content={
                "error": "ai_not_configured",
                "message": "الموقع يعمل، لكن نموذج تقسيم الصور لم يُربط بعد. يلزم ضبط HF_SPACE_ID وHF_API_NAME بعد التحقق من واجهة النموذج."
            },
        )

    # Intentionally do not guess a model's input/output schema.
    # Once the Space endpoint has been verified, wire its exact API here.
    return JSONResponse(
        status_code=501,
        content={
            "error": "model_adapter_needed",
            "message": "تم ضبط معرّف النموذج، لكن يلزم إضافة محوّل مطابق لمدخلات ومخرجات الـ Space قبل معالجة الصور."
        },
    )
