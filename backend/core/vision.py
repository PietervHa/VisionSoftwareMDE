from pathlib import Path
from backend.detection.objectdetection import run_object_detection
from backend.detection.ocr import OCR
import threading
import time
from backend.core.config_loader import cfg
from backend.core.inspection_engine import InspectionEngine

# Backward-compatible alias; primary source is cfg["vision_mode"].
VISION_MODE = cfg["vision_mode"]

ocr_instance = OCR()
_inspection_engine = None
_app_state = None

def bind_app_state(app_state):
    global ocr_instance, _inspection_engine, _app_state
    _app_state = app_state
    ocr_instance = OCR(app_state=app_state)
    _inspection_engine = InspectionEngine(app_state)

    # Best effort: auto-load configured classifier model if path exists.
    model_path = cfg.get("object_detection", {}).get("classifier_model_path", "")
    if model_path:
        resolved = Path(model_path)
        if not resolved.is_absolute():
            resolved = Path(__file__).resolve().parents[2] / resolved
        if resolved.exists() and _inspection_engine is not None:
            _inspection_engine.load_classifier(model_path)

def load_classifier(model_path: str) -> bool:
    engine = _inspection_engine
    if engine is None:
        return False
    return bool(engine.load_classifier(model_path))

def _normalize_result(result: dict, mode: str) -> dict:
    normalized = dict(result or {})
    normalized.setdefault("mode", mode)

    if "status" not in normalized:
        if normalized.get("error"):
            normalized["status"] = "NOK"
        elif mode == "ocr":
            normalized["status"] = "OK" if normalized.get("detections") else "NOK"
        else:
            normalized["status"] = "NOK"

    return normalized

def _run_object_detection(frame):
    if _inspection_engine is None:
        # Fallback to legacy output if engine is unavailable, but always return status.
        legacy = run_object_detection(frame)
        legacy["status"] = "NOK"
        legacy.setdefault("error", "no_model")
        return legacy
    return _inspection_engine.evaluate(frame)

def _run_with_callback(fn, frame, callback, mode):
    # Keep vision trigger loop non-blocking by running inference in a daemon worker.
    def worker():
        start = time.perf_counter()
        try:
            result = fn(frame)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            result = {
                "status": "NOK",
                "detections": [],
                "processing_time_ms": duration_ms,
                "mode": mode,
                "error": str(exc),
            }

        callback(_normalize_result(result, mode))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

def run_vision(frame, callback=None):
    """
    Dispatcher function that routes to OCR or object detection
    based on VISION_MODE configuration.

    If callback is provided, runs selected vision mode in a background thread.
    Otherwise, runs synchronously.

    """
    mode_getter = getattr(_app_state, "get_vision_mode", None)
    mode = mode_getter() if callable(mode_getter) else cfg["vision_mode"]

    if mode == "ocr":
        if callback:
            _run_with_callback(ocr_instance.run, frame, callback, mode="ocr")
            return None
        return _normalize_result(ocr_instance.run(frame), mode="ocr")

    if mode == "object_detection":
        if callback:
            _run_with_callback(_run_object_detection, frame, callback, mode="object_detection")
            return None
        return _normalize_result(_run_object_detection(frame), mode="object_detection")

    raise ValueError(f"Unknown VISION_MODE: {mode}")
