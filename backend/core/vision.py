from backend.detection.objectdetection import run_object_detection
from backend.detection.ocr import OCR
import threading
import time
from backend.core.config_loader import cfg

# Backward-compatible alias; primary source is cfg["vision_mode"].
VISION_MODE = cfg["vision_mode"]

ocr_instance = OCR()
_app_state = None

def bind_app_state(app_state):
    global ocr_instance, _app_state
    _app_state = app_state
    ocr_instance = OCR(app_state=app_state)

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
                "mode": cfg["vision_mode"],
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
    mode = _app_state.get_vision_mode() if _app_state is not None else cfg["vision_mode"]

    if mode == "ocr":
        if callback:
            _run_with_callback(ocr_instance.run, frame, callback)
            return None
        return ocr_instance.run(frame)

    if mode == "object_detection":
        if callback:
            _run_with_callback(run_object_detection, frame, callback)
            return None
        return run_object_detection(frame)

    raise ValueError(f"Unknown VISION_MODE: {mode}")
