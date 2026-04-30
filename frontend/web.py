import json
from datetime import date, datetime
import cv2
from flask import Flask, Response, send_file
from flask_cors import CORS
from flask import jsonify, request
from pathlib import Path
from backend.core.config_loader import cfg
from backend.utils.roi import draw_roi
import time


def _resolve_repo_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parents[1] / resolved
    return resolved.resolve()


def create_app(camera, app_state):
    app = Flask(__name__)
    CORS(app)

    JPEG_QUALITY = int(cfg.get("hmi", {}).get("stream_quality", 75))

    def generate_frames():
        if not cfg["hmi"]["enable_video_feed"]:
            import numpy as np
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Video feed disabled", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # quality applies to the disabled-feed placeholder too
            _, buffer = cv2.imencode(".jpg", blank,
                                     [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            while True:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
                time.sleep(1)

        FRAME_INTERVAL_MS = 33.33

        while True:
            frame_start = time.time()

            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_for_stream = frame
            roi = cfg.get("roi")
            if cfg["hmi"]["debug_draw_roi"] and app_state.get_vision_mode() == "ocr" and roi:
                # use shared draw_roi instead of inline _draw_roi helper
                frame_for_stream = draw_roi(frame.copy(), roi)

            # encode with configured quality instead of OpenCV default (95)
            _, buffer = cv2.imencode(".jpg", frame_for_stream,
                                     [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

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
        return jsonify({
            "classifier_loaded": bool(status.get("loaded", False)),
            "model_path": status.get("model_path") or None,
        })

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

    @app.route("/analytics")
    def analytics():
        template_path = Path(__file__).resolve().parent / "templates" / "analytics.html"
        return send_file(template_path)

    @app.route("/analytics/data")
    def analytics_data():
        try:
            result_dir = cfg.get("output", {}).get("result_dir", "data/results")
            output_dir = Path(result_dir)
            if not output_dir.is_absolute():
                output_dir = Path(__file__).resolve().parents[1] / output_dir

            daily_file = output_dir / f"{date.today().isoformat()}.jsonl"

            results = []
            if daily_file.exists():
                with daily_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                results.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue

            total = len(results)
            ok_count = sum(1 for r in results if r.get("status") == "OK")
            nok_count = total - ok_count

            processing_times = [
                r["processing_time_ms"] for r in results
                if isinstance(r.get("processing_time_ms"), (int, float))
            ]
            avg_processing_ms = round(
                sum(processing_times) / len(processing_times), 2
            ) if processing_times else 0.0

            # Build timeline: group results by hour (0-23), count OK and NOK per hour
            timeline = {str(h): {"ok": 0, "nok": 0} for h in range(24)}
            for r in results:
                ts = r.get("timestamp", "")
                try:
                    hour = str(datetime.fromisoformat(ts).hour)
                    if r.get("status") == "OK":
                        timeline[hour]["ok"] += 1
                    else:
                        timeline[hour]["nok"] += 1
                except Exception:
                    continue

            # Builds speed timeline in 15-minute buckets
            # Key format: "HH:MM" for each 15-min slot (00:00, 00:15, 00:30, 00:45, 01:00 ...)
            speed_buckets = {}
            for r in results:
                ts = r.get("timestamp", "")
                pt = r.get("processing_time_ms")
                if not isinstance(pt, (int, float)):
                    continue
                try:
                    dt = datetime.fromisoformat(ts)
                    minute_slot = (dt.minute // 15) * 15
                    key = f"{dt.hour:02d}:{minute_slot:02d}"
                    if key not in speed_buckets:
                        speed_buckets[key] = []
                    speed_buckets[key].append(float(pt))
                except Exception:
                    continue

            speed_timeline = {}
            for key, times in speed_buckets.items():
                speed_timeline[key] = {
                    "avg": round(sum(times) / len(times), 2),
                    "min": round(min(times), 2),
                    "max": round(max(times), 2),
                }

            return jsonify({
                "date": date.today().isoformat(),
                "total": total,
                "ok_count": ok_count,
                "nok_count": nok_count,
                "ok_rate": round(ok_count / total * 100, 1) if total > 0 else 0.0,
                "avg_processing_ms": avg_processing_ms,
                "timeline": timeline,
                "speed_timeline": speed_timeline,
            })

        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app