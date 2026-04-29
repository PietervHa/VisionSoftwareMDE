from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional
from backend.core.config_loader import cfg
from backend.core.inspection_engine import InspectionEngine
from backend.detection.objectdetection import run_object_detection
from backend.detection.ocr import OCR
from backend.utils.logger import get_logger
import time

logger = get_logger(__name__)

# Backward-compatible alias; primary source is cfg["vision_mode"].
VISION_MODE = cfg["vision_mode"]

ocr_instance = OCR()
_inspection_engine: Optional[InspectionEngine] = None
_app_state: Optional[Any] = None
_cached_vision_mode: str = cfg.get("vision_mode", "object_detection")

_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision_worker")


def _resolve_model_path(model_path: str) -> Path:
    resolved = Path(model_path)
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parents[2] / resolved
    return resolved.resolve()


def _sync_classifier_state(model_path: str, loaded: bool) -> None:
    app_state = _app_state
    if app_state is None or not hasattr(app_state, "set_classifier_loaded"):
        return
    app_state.set_classifier_loaded(model_path if loaded else "")


def bind_app_state(app_state) -> None:
    global ocr_instance, _inspection_engine, _app_state, _cached_vision_mode
    _app_state = app_state
    _cached_vision_mode = cfg.get("vision_mode", "object_detection")
    ocr_instance = OCR(app_state=app_state)
    _inspection_engine = InspectionEngine(app_state)

    od_cfg = cfg.get("object_detection", {})
    backend = str(od_cfg.get("backend", "classifier")).strip().lower()

    # Best effort: auto-load classifier model only in classifier backend mode.
    if backend != "classifier":
        _sync_classifier_state("", False)
        return

    classifier_cfg = od_cfg.get("classifier", {}) if isinstance(od_cfg.get("classifier"), dict) else {}
    model_path = str(classifier_cfg.get("model_path", "")).strip()
    if model_path:
        resolved = _resolve_model_path(model_path)
        if resolved.exists() and _inspection_engine is not None:
            loaded = _inspection_engine.load_classifier(str(resolved))
            _sync_classifier_state(str(resolved), loaded)
        else:
            _sync_classifier_state("", False)


def load_classifier(model_path: str) -> bool:
    engine = _inspection_engine
    if engine is None:
        return False
    resolved = _resolve_model_path(model_path)
    loaded = bool(engine.load_classifier(str(resolved)))
    _sync_classifier_state(str(resolved), loaded)
    return loaded


def _normalize_detections(result: dict, mode: str) -> list:
    detections = result.get("detections")
    if isinstance(detections, list):
        normalized = [d for d in detections if isinstance(d, dict)]
    else:
        normalized = []

    if not normalized and mode == "object_detection" and result.get("label"):
        normalized = [
            {
                "label": result.get("label"),
                "text": result.get("label"),
                "confidence": result.get("confidence", 0.0),
            }
        ]

    return normalized


def _get_active_threshold() -> float:
    app_state = _app_state
    if app_state is not None and hasattr(app_state, "get_threshold"):
        try:
            return float(app_state.get_threshold())
        except Exception:
            pass
    return float(cfg.get("confidence_threshold", 0.0))


def _extract_confidence(normalized: dict) -> float:
    if "confidence" in normalized:
        try:
            return float(normalized.get("confidence", 0.0))
        except (TypeError, ValueError):
            return 0.0

    detections = normalized.get("detections") or []
    best = 0.0
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        try:
            conf = float(detection.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        if conf > best:
            best = conf
    return best


def _normalize_result(result: dict, mode: str) -> dict:
    normalized = dict(result or {})
    normalized.setdefault("mode", mode)
    normalized["detections"] = _normalize_detections(normalized, mode)
    normalized["confidence"] = _extract_confidence(normalized)

    threshold = _get_active_threshold()
    has_error = bool(normalized.get("error"))

    if has_error:
        normalized["status"] = "NOK"
        return normalized

    # Threshold is leading: confidence below threshold is always NOK.
    if normalized["confidence"] < threshold:
        normalized["status"] = "NOK"
        return normalized

    if "status" not in normalized:
        normalized["status"] = "OK" if normalized["detections"] else "NOK"

    return normalized


def _run_object_detection(frame):
    engine = _inspection_engine
    if engine is None:
        # Fallback to legacy output if engine is unavailable, but always return status.
        legacy = run_object_detection(frame)
        legacy["status"] = "OK" if legacy.get("detections") else "NOK"
        legacy.setdefault("error", "no_model")
        return legacy
    return engine.evaluate(frame)


def _run_with_callback(fn, frame, callback, mode: str, profile: bool = False) -> None:
    def worker():
        start = time.perf_counter()
        result = None
        exception_msg = None
        try:
            result = fn(frame, profile=profile) if profile else fn(frame)
        except Exception as exc:
            exception_msg = str(exc)
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            result = {
                "status": "NOK",
                "detections": [],
                "processing_time_ms": duration_ms,
                "mode": mode,
                "error": exception_msg,
            }

        if exception_msg:
            logger.error("Vision worker exception: %s", exception_msg)
        callback(_normalize_result(result, mode))

    _pool.submit(worker)


def run_vision(frame, callback=None, profile: bool = False):
    mode_getter = getattr(_app_state, "get_vision_mode", None)
    mode = mode_getter() if callable(mode_getter) else _cached_vision_mode

    if mode == "ocr":
        if callback:
            _run_with_callback(ocr_instance.run, frame, callback, mode="ocr", profile=profile)
            return None
        return _normalize_result(ocr_instance.run(frame, profile=profile), mode="ocr")

    if mode == "object_detection":
        if callback:
            _run_with_callback(_run_object_detection, frame, callback, mode="object_detection")
            return None
        return _normalize_result(_run_object_detection(frame), mode="object_detection")

    raise ValueError(f"Unknown VISION_MODE: {mode}")