import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="Person Detection API")

model = YOLO("yolov8n.onnx", task="detect")


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = np.array(image)

    results = model.track(
        frame,
        conf=0.5,
        classes=[0],
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )
    result = results[0]

    detections = []
    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "tracking_id": int(track_id),
                "confidence": round(float(conf), 4),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                }
            })

    return JSONResponse(content={
        "class": "person",
        "count": len(detections),
        "detections": detections
    })