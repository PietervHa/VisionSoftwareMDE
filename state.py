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
