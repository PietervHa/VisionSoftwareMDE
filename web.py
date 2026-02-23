import cv2
from flask import Flask, Response, send_file
from flask_cors import CORS
import state
from flask import jsonify, request

def create_app(camera):
    app = Flask(__name__)
    CORS(app)

    def generate_frames():
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue

            _, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    @app.route("/")
    def index():
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
