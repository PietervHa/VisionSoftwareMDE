import threading
import keyboard
from backend.core.camera import Camera
from backend.core.vision import run_vision, bind_app_state
from backend.core.tcp_trigger_server import TCPTriggerServer
from frontend.web import create_app
from backend.core.state import AppState
import time
from backend.core.config_loader import cfg
from backend.utils.logger import setup_logging, get_logger
from backend.output.result_writer import save_result

log = get_logger(__name__)
_vision_busy = threading.Event()

def _process_vision_result(result, trigger_time, app_state):
    """Callback to handle vision results from background thread"""
    try:
        # Calculate total cycle time from trigger to result
        cycle_time_ms = round((time.perf_counter() - trigger_time) * 1000, 1)

        threshold = app_state.get_threshold()
        status = result["status"]
        confidence = result.get("confidence", 0)

        result_dict = {
            **result,
            "confidence_threshold": threshold,
            "cycle_time_ms": cycle_time_ms  # Total time from trigger to result
        }
        app_state.update_result(result_dict)
        app_state.increment_counter(status)

        # Deferred I/O: write outside main thread if possible
        try:
            save_result(result_dict)
        except Exception as exc:
            log.error("Failed to write result file: %s", exc)

        detection_count = len(result.get("detections", []))
        log.info(
            "VISION RESULT: status=%s cycle_time_ms=%s confidence=%s detections=%s",
            status,
            cycle_time_ms,
            confidence,
            detection_count,
        )
    except Exception as exc:
        log.error("Failed to process vision result: %s", exc)

def vision_trigger_loop(camera, app_state):
    global _vision_busy
    log.info("Press Q to trigger vision. Ctrl+C to exit.")

    while True:
        try:
            keyboard.wait("q")

            if _vision_busy.is_set():
                continue

            frame = camera.get_frame()
            if frame is None:
                log.warning("No frame available")
                continue

            # Track trigger time for cycle time measurement
            trigger_time = time.perf_counter()

            def _callback(result):
                global _vision_busy
                try:
                    _process_vision_result(result, trigger_time, app_state)
                finally:
                    _vision_busy.clear()

            _vision_busy.set()

            # Trigger vision in background thread with callback
            run_vision(frame, callback=_callback)
            log.debug("Vision processing started (non-blocking)")
        except Exception as exc:
            _vision_busy.clear()
            log.error("Vision trigger loop error: %s", exc)

def main():
    setup_logging()
    app_state = AppState()
    bind_app_state(app_state)

    try:
        camera = Camera(0, app_state=app_state)

        web_cfg = cfg["web"]
        app = create_app(camera, app_state)
        web_thread = threading.Thread(
            target=lambda: app.run(host=web_cfg["host"], port=web_cfg["port"], threaded=True),
            daemon=True,
        )
        web_thread.start()

        trigger_server = TCPTriggerServer(
            camera=camera,
            run_vision_fn=run_vision,
            process_result_fn=lambda result, t: _process_vision_result(result, t, app_state),
        )
        trigger_server.start()

        vision_trigger_loop(camera, app_state)
    except Exception as exc:
        log.error("Fatal startup/runtime error: %s", exc)
        raise

if __name__ == "__main__":
    main()
