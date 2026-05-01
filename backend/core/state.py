import threading
from backend.core.config_loader import cfg

class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_result = {
            "detections": [],
            "processing_time_ms": 0,
            "status": "NOK",
        }
        self.counters = {
            "ok": 0,
            "nok": 0,
            "total": 0,
        }
        self.confidence_threshold = 0.0
        self.maintenance_mode = True
        self.camera_rotation = 0  # steps: 0, 1, 2, 3 (each = 90° clockwise)
        self.vision_mode = cfg["vision_mode"]
        self.ocr_keyword = cfg["ocr"]["keywords"][0] if cfg["ocr"]["keywords"] else ""
        self.classifier_model_path = ""
        self.classifier_loaded = False
        self.set_threshold(float(cfg["confidence_threshold"]))

    def update_result(self, result: dict):
        with self.lock:
            self.latest_result = result

    def increment_counter(self, status: str):
        with self.lock:
            self.counters["total"] += 1
            if status == "OK":
                self.counters["ok"] += 1
            elif status == "NOK":
                self.counters["nok"] += 1

    def reset_counters(self):
        with self.lock:
            self.counters = {
                "ok": 0,
                "nok": 0,
                "total": 0,
            }

    def get_snapshot(self) -> dict:
        with self.lock:
            return {
                "result": dict(self.latest_result),
                "counters": dict(self.counters),
            }

    def get_threshold(self) -> float:
        with self.lock:
            return self.confidence_threshold

    def set_threshold(self, value: float):
        with self.lock:
            self.confidence_threshold = max(0.0, min(1.0, float(value)))

    def get_maintenance_mode(self) -> bool:
        with self.lock:
            return self.maintenance_mode

    def set_maintenance_mode(self, value: bool):
        with self.lock:
            self.maintenance_mode = bool(value)

    def get_camera_rotation(self) -> int:
        with self.lock:
            return self.camera_rotation

    def rotate_camera(self):
        with self.lock:
            self.camera_rotation = (self.camera_rotation + 1) % 4

    def get_vision_mode(self) -> str:
        with self.lock:
            return self.vision_mode

    def set_vision_mode(self, value: str):
        with self.lock:
            if value in ("ocr", "object_detection"):
                self.vision_mode = value

    def get_ocr_keyword(self) -> str:
        with self.lock:
            return self.ocr_keyword

    def set_ocr_keyword(self, value: str):
        with self.lock:
            self.ocr_keyword = value.strip().lower()

    def get_classifier_status(self) -> dict:
        with self.lock:
            return {
                "loaded": self.classifier_loaded,
                "model_path": self.classifier_model_path,
            }

    def set_classifier_loaded(self, path: str):
        with self.lock:
            self.classifier_loaded = bool(path)
            self.classifier_model_path = path if path else ""

    def load_classifier(self, model_path: str) -> bool:
        model_path = (model_path or "").strip()
        if not model_path:
            self.set_classifier_loaded("")
            return False

        loaded = False
        try:
            # Keep state as orchestrator metadata; actual load happens in vision module.
            from backend.core import vision

            loader = getattr(vision, "load_classifier", None)
            if callable(loader):
                loaded = bool(loader(model_path))
        except Exception:
            loaded = False

        self.set_classifier_loaded(model_path if loaded else "")
        return loaded

