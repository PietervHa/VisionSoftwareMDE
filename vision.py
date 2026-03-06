from ultralytics import YOLO
from objectdetection import run_object_detection
from ocr import OCR
import threading

# Load YOLO model ONCE
model = YOLO("yolov8n.pt")  # nano = fast, CPU friendly

# Hardcode the vision mode: "ocr" or "object_detection"
#VISION_MODE = "object_detection"  # Change to "ocr" to use OCR mode
VISION_MODE = "ocr"

ocr_instance = OCR()

def run_vision(frame, callback=None):
    """
    Dispatcher function that routes to OCR or object detection
    based on VISION_MODE configuration.

    For OCR mode with callback, runs OCR in a background thread.
    Otherwise, runs synchronously.
    """
    if VISION_MODE == "ocr":
        if callback:
            # Run OCR in background thread
            thread = threading.Thread(target=lambda: callback(ocr_instance.run(frame)), daemon=True)
            thread.start()
            return None  # Return immediately
        else:
            return ocr_instance.run(frame)
    elif VISION_MODE == "object_detection":
        return run_object_detection(frame)
    else:
        raise ValueError(f"Unknown VISION_MODE: {VISION_MODE}")
