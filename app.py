from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import base64
from pathlib import Path


import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

app = FastAPI(title="DentAI — Agiz Ici Analiz Sistemi")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_oral = None
model_ortho = None

# ==================== SINIF RENKLERI (BGR for OpenCV) ====================
CLASS_COLORS = {
    # Agiz sagligi (6 sinif)
    "calculus":          (11, 158, 245),
    "caries":            (68,  68, 239),
    "gingivitis":        (153, 72, 236),
    "hypodontia":        (246, 92, 139),
    "tooth_discolation": (22, 115, 249),
    "ulcer":             (212, 182,  6),
    # Ortodonti (3 sinif)
    "TT":                (94, 197,  34),
    "DO":                (166, 184, 20),
    "MR":                (241, 102, 99),
}

DISPLAY_NAMES = {
    "calculus":          "Dis Tasi",
    "caries":            "Curuk",
    "gingivitis":        "Dis Eti Iltihabi",
    "hypodontia":        "Eksik Dis",
    "tooth_discolation": "Renk Degisikligi",
    "ulcer":             "Agiz Ulseri",
    "TT":                "Dis Torsiyonu",
    "DO":                "Derin Overjet",
    "MR":                "Alt Cene Geriligi",
}


def draw_boxes(img, boxes_data):
    """Her iki modelden gelen tum kutulari tek goruntu uzerine ciz."""
    annotated = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thick = 1
    box_thick = 2

    for box in boxes_data:
        color = CLASS_COLORS.get(box["label"], (180, 180, 180))
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        # Kutu
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thick)

        # Etiket metni
        text = f"{box['label']} {box['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thick)

        # Etiket arka plan + yazi
        label_y = max(y1 - 4, th + 10)
        cv2.rectangle(annotated, (x1, label_y - th - 6), (x1 + tw + 8, label_y + 2), color, -1)
        cv2.putText(annotated, text, (x1 + 4, label_y - 2), font, font_scale,
                    (255, 255, 255), font_thick, cv2.LINE_AA)

    return annotated


def load_models():
    global model_oral, model_ortho
    from ultralytics import YOLO

    oral_path = Path("oral_health_best.pt")
    ortho_path = Path("ortho_best.pt")

    if not oral_path.exists():
        raise FileNotFoundError("oral_health_best.pt (agiz sagligi modeli) bulunamadi!")
    model_oral = YOLO(str(oral_path))
    print(f"Oral model yuklendi: {list(model_oral.names.values())}")

    if ortho_path.exists():
        model_ortho = YOLO(str(ortho_path))
        print(f"Ortho model yuklendi: {list(model_ortho.names.values())}")
    else:
        print("UYARI: ortho_best.pt bulunamadi — sadece agiz sagligi modeli calisacak.")


@app.on_event("startup")
async def startup_event():
    try:
        load_models()
    except Exception as e:
        print(f"HATA: {e}")
        raise


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model_oral is None:
        raise HTTPException(status_code=500, detail="Model yuklenemedi")

    allowed = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Sadece JPG, PNG, WEBP desteklenir")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Goruntu okunamadi")

    boxes_data = []

    # ---- 1. Agiz sagligi modeli ----
    oral_results = model_oral.predict(source=img, conf=0.30, iou=0.5, verbose=False)
    for box in oral_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model_oral.names[cls]
        boxes_data.append({
            "x1": round(x1), "y1": round(y1),
            "x2": round(x2), "y2": round(y2),
            "confidence": round(conf, 3),
            "label": label,
            "category": "oral"
        })

    # ---- 2. Ortodonti modeli ----
    if model_ortho is not None:
        ortho_results = model_ortho.predict(source=img, conf=0.35, iou=0.5, verbose=False)
        for box in ortho_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model_ortho.names[cls]
            boxes_data.append({
                "x1": round(x1), "y1": round(y1),
                "x2": round(x2), "y2": round(y2),
                "confidence": round(conf, 3),
                "label": label,
                "category": "orthodontic"
            })


    annotated = draw_boxes(img, boxes_data)
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    h, w = img.shape[:2]

    return JSONResponse({
        "success": True,
        "detections": len(boxes_data),
        "boxes": boxes_data,
        "image_width": w,
        "image_height": h,
        "annotated_image": f"data:image/jpeg;base64,{img_base64}",
    })


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "oral_model": model_oral is not None,
        "ortho_model": model_ortho is not None,
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
