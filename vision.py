import time
import cv2
from ultralytics import YOLO

# Load YOLO model ONCE
model = YOLO("yolov8n.pt")  # nano = fast, CPU friendly

def run_vision(frame):
    start = time.time()

    # Optional: resize for speed (recommended)
    frame_resized = cv2.resize(frame, (640, 640))

    # Run YOLO inference
    results = model(frame_resized, verbose=False)

    detections = []

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[class_id]

            detections.append({
                "label": label,
                "confidence": round(confidence, 3)
            })

    duration_ms = int((time.time() - start) * 1000)

    return {
        "detections": detections,
        "processing_time_ms": duration_ms
    }
