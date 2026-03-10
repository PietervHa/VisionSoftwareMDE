from objectdetection import run_object_detection
from ocr import OCR
import threading
import time

# Hardcode the vision mode: "ocr" or "object_detection"
#VISION_MODE = "object_detection"  # Change to "ocr" to use OCR mode
VISION_MODE = "object_detection"

ocr_instance = OCR()

def _run_with_callback(fn, frame, callback):
    # Keep vision trigger loop non-blocking by running inference in a daemon worker.
    def worker():
        start = time.perf_counter()
        try:
            result = fn(frame)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            result = {
                "detections": [],
                "processing_time_ms": duration_ms,
                "mode": VISION_MODE,
                "error": str(exc),
            }
        callback(result)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

def run_vision(frame, callback=None):
    """
    Dispatcher function that routes to OCR or object detection
    based on VISION_MODE configuration.

    If callback is provided, runs selected vision mode in a background thread.
    Otherwise, runs synchronously.
    """
    if VISION_MODE == "ocr":
        if callback:
            _run_with_callback(ocr_instance.run, frame, callback)
            return None
        return ocr_instance.run(frame)

    if VISION_MODE == "object_detection":
        if callback:
            _run_with_callback(run_object_detection, frame, callback)
            return None
        return run_object_detection(frame)

    raise ValueError(f"Unknown VISION_MODE: {VISION_MODE}")
