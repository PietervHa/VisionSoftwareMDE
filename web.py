import cv2
from flask import Flask, Response, send_file
from flask_cors import CORS
import state
from flask import jsonify, request
from vision import VISION_MODE
import config

def create_app(camera):
    app = Flask(__name__)
    CORS(app)

    def _draw_roi(frame):
        if not getattr(config, "DEBUG_DRAW_ROI", False):
            return frame

        roi = getattr(config, "ROI", None)
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
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue

            frame_for_stream = frame
            if getattr(config, "DEBUG_DRAW_ROI", False):
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
        if VISION_MODE == "ocr":
            return send_file("templates/CFF_OCR.html")
        else:
            return send_file("templates/CFF.html")

    @app.route("/result")
    def get_result():
        with state.lock:
            return jsonify({
                "result": state.latest_result,
                "counters": state.counters
            })

    @app.route("/threshold")
    def get_threshold():
        with state.lock:
            return jsonify({"threshold": state.confidence_threshold})

    @app.route("/threshold", methods=["POST"])
    def set_threshold():
        data = request.json
        new_value = float(data.get("threshold", 0.5))

        # Clamp between 0 and 1
        new_value = max(0.0, min(1.0, new_value))

        with state.lock:
            state.confidence_threshold = new_value

        return jsonify({"threshold": new_value})

    @app.route("/video_feed")
    def video_feed():
        return Response(
            generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/reset_counters", methods=["POST"])
    def reset_counters():
        with state.lock:
            state.counters = {
                "ok": 0,
                "nok": 0,
                "total": 0
            }
        return jsonify({"status": "counters reset"})

    return app
