import cv2
from flask import Flask, Response, send_file
from flask_cors import CORS
from flask import jsonify, request
from config_loader import cfg

def create_app(camera, app_state):
    app = Flask(__name__)
    CORS(app)

    def _draw_roi(frame):
        if not cfg["hmi"]["debug_draw_roi"]:
            return frame

        roi = cfg.get("roi")
        if not roi:
            return frame

        h, w = frame.shape[:2]

        def _to_px(value, max_dim):
            if value <= 1.0:
                return int(round(value * max_dim))
            return int(round(value))

        x1 = _to_px(float(roi.get("x_start", 0.0)), w)
        y1 = _to_px(float(roi.get("y_start", 0.0)), h)
        x2 = _to_px(float(roi.get("x_end", 1.0)), w)
        y2 = _to_px(float(roi.get("y_end", 1.0)), h)

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            return frame

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    def generate_frames():
        # Check if video feed is enabled
        if not cfg["hmi"]["enable_video_feed"]:
            # Return a single black frame with text
            import numpy as np
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Video feed disabled", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode(".jpg", blank)
            while True:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
                import time
                time.sleep(1)  # Low CPU usage when disabled

        while True:
            frame = camera.get_frame()
            if frame is None:
                continue

            frame_for_stream = frame
            if cfg["hmi"]["debug_draw_roi"]:
                frame_for_stream = _draw_roi(frame.copy())

            _, buffer = cv2.imencode(".jpg", frame_for_stream)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    @app.route("/")
    def index():
        return send_file("templates/hmi.html")

    @app.route("/status")
    def get_status():
        return jsonify({
            "vision_mode": cfg["vision_mode"],
            "machine_id": cfg["machine_id"],
            "version": "1.0.0"
        })

    @app.route("/result")
    def get_result():
        return jsonify(app_state.get_snapshot())

    @app.route("/threshold")
    def get_threshold():
        return jsonify({"threshold": app_state.get_threshold()})

    @app.route("/threshold", methods=["POST"])
    def set_threshold():
        data = request.json
        new_value = float(data.get("threshold", 0.5))

        app_state.set_threshold(new_value)
        return jsonify({"threshold": app_state.get_threshold()})

    @app.route("/video_feed")
    def video_feed():
        return Response(
            generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/reset_counters", methods=["POST"])
    def reset_counters():
        app_state.reset_counters()
        return jsonify({"status": "counters reset"})

    return app