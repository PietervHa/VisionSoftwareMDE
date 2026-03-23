import threading
import keyboard
from camera import Camera
from vision import run_vision
from web import create_app
from state import AppState
import time
from config_loader import cfg
from utils.logger import setup_logging, get_logger
from output.result_writer import save_result

log = get_logger(__name__)

def _process_vision_result(result, trigger_time, app_state):
    """Callback to handle vision results from background thread"""
    try:
        # Calculate total cycle time from trigger to result
        cycle_time_ms = round((time.perf_counter() - trigger_time) * 1000, 1)

        threshold = app_state.get_threshold()

        high_conf = [
            d for d in result["detections"]
            if d["confidence"] >= threshold
        ]

        status = "OK" if high_conf else "NOK"

        result_dict = {
            **result,
            "status": status,
            "confidence_threshold": threshold,
            "cycle_time_ms": cycle_time_ms  # Total time from trigger to result
        }
        app_state.update_result(result_dict)
        app_state.increment_counter(status)

        try:
            save_result(result_dict)
        except Exception as exc:
            log.error("Failed to write result file: %s", exc)

        detection_count = len(result.get("detections", []))
        log.info(
            "VISION RESULT: status=%s cycle_time_ms=%s detections=%s",
            status,
            cycle_time_ms,
            detection_count,
        )
    except Exception as exc:
        log.error("Failed to process vision result: %s", exc)

def vision_trigger_loop(camera, app_state):
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
            run_vision(frame, callback=lambda result: _process_vision_result(result, trigger_time, app_state))
            log.debug("Vision processing started (non-blocking)")
        except Exception as exc:
            log.error("Vision trigger loop error: %s", exc)

def main():
    setup_logging()
    app_state = AppState()

    try:
        camera = Camera(0)

        web_cfg = cfg["web"]
        app = create_app(camera, app_state)
        web_thread = threading.Thread(
            target=lambda: app.run(host=web_cfg["host"], port=web_cfg["port"], threaded=True),
            daemon=True,
        )
        web_thread.start()

        vision_trigger_loop(camera, app_state)
    except Exception as exc:
        log.error("Fatal startup/runtime error: %s", exc)
        raise

if __name__ == "__main__":
    main()
