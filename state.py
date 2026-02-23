import threading

lock = threading.Lock()

latest_result = {
    "detections": [],
    "processing_time_ms": 0,
    "status": "NOK"
}

counters = {
    "ok": 0,
    "nok": 0,
    "total": 0
}

confidence_threshold = 0.8  # dit is in procenten dus 0.5 is 50% etc.