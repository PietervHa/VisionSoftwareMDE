import threading
import keyboard
from camera import Camera
from vision import run_vision
from web import create_app
import state

def vision_trigger_loop(camera):
    print("Press 'Q' to trigger vision. Ctrl+C to exit.")

    while True:
        keyboard.wait("q")

        frame = camera.get_frame()
        if frame is None:
            print("No frame available")
            continue

        result = run_vision(frame)

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
                "confidence_threshold": threshold
            }

            state.counters["total"] += 1
            if status == "OK":
                state.counters["ok"] += 1
            else:
                state.counters["nok"] += 1

        print(f"VISION RESULT: {state.latest_result}")

def main():
    camera = Camera(0)

    app = create_app(camera)
    web_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=5000, threaded=True),
        daemon=True,
    )
    web_thread.start()

    vision_trigger_loop(camera)

if __name__ == "__main__":
    main()
