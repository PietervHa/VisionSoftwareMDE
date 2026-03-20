import threading
from config_loader import cfg

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
                "maintenance_mode": self.maintenance_mode,
                "vision_mode": self.vision_mode,
                "ocr_keyword": self.ocr_keyword,
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
