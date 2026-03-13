import threading
import keyboard
from camera import Camera
from vision import run_vision
from web import create_app
import state
import time
from config_loader import cfg
from utils.logger import setup_logging, get_logger


log = get_logger(__name__)

def _process_vision_result(result, trigger_time):
    """Callback to handle vision results from background thread"""
    try:
        # Calculate total cycle time from trigger to result
        cycle_time_ms = round((time.perf_counter() - trigger_time) * 1000, 1)

        with state.lock:
            threshold = state.confidence_threshold

        high_conf = [
            d for d in result["detections"]
            if d["confidence"] >= threshold
        ]

        status = "OK" if high_conf else "NOK"

        with state.lock:
            state.latest_result = {
                **result,
                "status": status,
                "confidence_threshold": threshold,
                "cycle_time_ms": cycle_time_ms  # Total time from trigger to result
            }

            state.counters["total"] += 1
            if status == "OK":
                state.counters["ok"] += 1
            else:
                state.counters["nok"] += 1

        detection_count = len(result.get("detections", []))
        log.info(
            "VISION RESULT: status=%s cycle_time_ms=%s detections=%s",
            status,
            cycle_time_ms,
            detection_count,
        )
    except Exception as exc:
        log.error("Failed to process vision result: %s", exc)

def vision_trigger_loop(camera):
    log.info("Press Q to trigger vision. Ctrl+C to exit.")

    while True:
        try:
            keyboard.wait("q")

            frame = camera.get_frame()
            if frame is None:
                log.warning("No frame available")
                continue

            # Track trigger time for cycle time measurement
            trigger_time = time.perf_counter()

            # Trigger OCR in background thread with callback
            run_vision(frame, callback=lambda result: _process_vision_result(result, trigger_time))
            log.debug("Vision processing started (non-blocking)")
        except Exception as exc:
            log.error("Vision trigger loop error: %s", exc)

def main():
    setup_logging()

    try:
        with state.lock:
            state.confidence_threshold = float(cfg["confidence_threshold"])

        camera = Camera(0)

        web_cfg = cfg["web"]
        app = create_app(camera)
        web_thread = threading.Thread(
            target=lambda: app.run(host=web_cfg["host"], port=web_cfg["port"], threaded=True),
            daemon=True,
        )
        web_thread.start()

        vision_trigger_loop(camera)
    except Exception as exc:
        log.error("Fatal startup/runtime error: %s", exc)
        raise

if __name__ == "__main__":
    main()
