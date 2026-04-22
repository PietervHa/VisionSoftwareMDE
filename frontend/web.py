import cv2
from flask import Flask, Response, send_file
from flask_cors import CORS
from flask import jsonify, request
from pathlib import Path
from backend.core.config_loader import cfg
import time


def _resolve_repo_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parents[1] / resolved
    return resolved.resolve()

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
                time.sleep(1)  # Low CPU usage when disabled

        # Frame rate cap at 30fps = 33.33ms per frame
        FRAME_INTERVAL_MS = 33.33
        
        while True:
            frame_start = time.time()
            
            frame = camera.get_frame()
            if frame is None:
                # Sleep 10ms before retrying instead of immediately looping
                time.sleep(0.01)
                continue

            frame_for_stream = frame
            if cfg["hmi"]["debug_draw_roi"] and app_state.get_vision_mode() == "ocr":
                frame_for_stream = _draw_roi(frame.copy())

            _, buffer = cv2.imencode(".jpg", frame_for_stream)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
            
            # Calculate elapsed time and sleep for remaining time in the 33ms window
            elapsed_ms = (time.time() - frame_start) * 1000
            remaining_ms = FRAME_INTERVAL_MS - elapsed_ms
            if remaining_ms > 0:
                time.sleep(remaining_ms / 1000)

    @app.route("/")
    def index():
        template_path = Path(__file__).resolve().parent / "templates" / "hmi.html"
        return send_file(template_path)

    @app.route("/status")
    def get_status():
        return jsonify({
            "vision_mode": app_state.get_vision_mode(),
            "maintenance_mode": app_state.get_maintenance_mode(),
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
        if not app_state.get_maintenance_mode():
            return jsonify({"error": "Not in maintenance mode"}), 403

        data = request.json or {}
        new_value = float(data.get("threshold", 0.5))

        app_state.set_threshold(new_value)
        return jsonify({"threshold": app_state.get_threshold()})

    @app.route("/maintenance_mode", methods=["POST"])
    def set_maintenance_mode():
        data = request.json or {}
        app_state.set_maintenance_mode(data.get("maintenance_mode", False))
        return jsonify({"maintenance_mode": app_state.get_maintenance_mode()})

    @app.route("/vision_mode", methods=["POST"])
    def set_vision_mode():
        if not app_state.get_maintenance_mode():
            return jsonify({"error": "Not in maintenance mode"}), 403

        data = request.json or {}
        app_state.set_vision_mode(data.get("vision_mode", ""))
        return jsonify({"vision_mode": app_state.get_vision_mode()})

    @app.route("/camera_rotation", methods=["POST"])
    def rotate_camera():
        if not app_state.get_maintenance_mode():
            return jsonify({"error": "Not in maintenance mode"}), 403
        app_state.rotate_camera()
        return jsonify({"camera_rotation": app_state.get_camera_rotation()})

    @app.route("/ocr_keyword")
    def get_ocr_keyword():
        return jsonify({"ocr_keyword": app_state.get_ocr_keyword()})

    @app.route("/ocr_keyword", methods=["POST"])
    def set_ocr_keyword():
        if not app_state.get_maintenance_mode():
            return jsonify({"error": "Not in maintenance mode"}), 403
        data = request.json or {}
        new_keyword = data.get("ocr_keyword", "").strip()
        if not new_keyword:
            return jsonify({"error": "ocr_keyword cannot be empty"}), 400
        app_state.set_ocr_keyword(new_keyword)
        return jsonify({"ocr_keyword": app_state.get_ocr_keyword()})

    @app.route("/load_classifier", methods=["POST"])
    def load_classifier():
        if not app_state.get_maintenance_mode():
            return jsonify({"error": "Not in maintenance mode"}), 403

        data = request.json or {}
        model_path = str(data.get("model_path", "")).strip()

        resolved_path = _resolve_repo_path(model_path)

        if not model_path or not resolved_path.exists():
            return jsonify({"error": "model_path does not exist"}), 400

        classifier_loaded = bool(app_state.load_classifier(str(resolved_path)))
        return jsonify({"classifier_loaded": classifier_loaded, "model_path": str(resolved_path)})

    @app.route("/classifier_status")
    def get_classifier_status():
        status = app_state.get_classifier_status()
        return jsonify(
            {
                "classifier_loaded": bool(status.get("loaded", False)),
                "model_path": status.get("model_path") or None,
            }
        )

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