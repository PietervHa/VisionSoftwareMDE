from ultralytics import YOLO
from objectdetection import run_object_detection
from ocr import OCR

# Load YOLO model ONCE
model = YOLO("yolov8n.pt")  # nano = fast, CPU friendly

# Hardcode the vision mode: "ocr" or "object_detection"
#VISION_MODE = "object_detection"  # Change to "ocr" to use OCR mode
VISION_MODE = "object_detection"

ocr_instance = OCR()

def run_vision(frame):
    """
    Dispatcher function that routes to OCR or object detection
    based on VISION_MODE configuration.
    """
    if VISION_MODE == "ocr":
        detections = ocr_instance.run(frame)
        return {
            "detections": detections,
            "mode": "ocr"
        }
    elif VISION_MODE == "object_detection":
        return run_object_detection(frame)
    else:
        raise ValueError(f"Unknown VISION_MODE: {VISION_MODE}")
