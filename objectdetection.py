import os
import threading
import time
import cv2
from ultralytics import YOLO

# Reduce Ultralytics console noise (startup banner/verbose logs).
os.environ.setdefault("YOLO_VERBOSE", "False")

# Load YOLO model ONCE
model = YOLO("yolov8n.pt")  # nano = fast, CPU friendly
_model_lock = threading.Lock()


def run_object_detection(frame):
    start = time.perf_counter()

    try:
        # Optional: resize for speed (recommended)
        frame_resized = cv2.resize(frame, (640, 640))

        # Run YOLO inference (guard shared model access across threads)
        with _model_lock:
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

        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        return {
            "detections": detections,
            "processing_time_ms": duration_ms,
            "mode": "object_detection"
        }
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "detections": [],
            "processing_time_ms": duration_ms,
            "mode": "object_detection",
            "error": str(exc)
        }
