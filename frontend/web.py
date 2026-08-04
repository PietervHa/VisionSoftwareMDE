import json
import os
from datetime import date, datetime, timedelta
import cv2
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from backend.core.config_loader import cfg
from backend.utils.roi import draw_roi
from tools.capture_dataset import DatasetCapture
import time


def _resolve_repo_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parents[1] / resolved
    return resolved.resolve()


class ThresholdBody(BaseModel):
    threshold: float = 0.5


class MaintenanceModeBody(BaseModel):
    maintenance_mode: bool = False


class VisionModeBody(BaseModel):
    vision_mode: str = ""


class OcrKeywordBody(BaseModel):
    ocr_keyword: str = ""


class LoadClassifierBody(BaseModel):
    model_path: str = ""


class DatasetCaptureBody(BaseModel):
    label: str = ""


def create_app(camera, app_state) -> FastAPI:
    app = FastAPI()

    # flask-cors' CORS(app) defaults to allowing all origins/methods/headers
    # without credentials; this mirrors that.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # shared DatasetCapture instance for the lifetime of this app
    dataset_capture = DatasetCapture()

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

    @app.get("/")
    def index():
        template_path = Path(__file__).resolve().parent / "templates" / "hmi.html"
        return FileResponse(template_path)

    @app.get("/status")
    def get_status():
        return {
            "vision_mode": app_state.get_vision_mode(),
            "maintenance_mode": app_state.get_maintenance_mode(),
            "machine_id": cfg["machine_id"],
            "version": "1.0.0",
        }

    @app.get("/result")
    def get_result():
        return app_state.get_snapshot()

    @app.get("/threshold")
    def get_threshold():
        return {"threshold": app_state.get_threshold()}

    @app.post("/threshold")
    def set_threshold(body: ThresholdBody):
        if not app_state.get_maintenance_mode():
            return JSONResponse(status_code=403, content={"error": "Not in maintenance mode"})
        app_state.set_threshold(body.threshold)
        return {"threshold": app_state.get_threshold()}

    @app.post("/maintenance_mode")
    def set_maintenance_mode(body: MaintenanceModeBody):
        app_state.set_maintenance_mode(body.maintenance_mode)
        return {"maintenance_mode": app_state.get_maintenance_mode()}

    @app.post("/vision_mode")
    def set_vision_mode(body: VisionModeBody):
        if not app_state.get_maintenance_mode():
            return JSONResponse(status_code=403, content={"error": "Not in maintenance mode"})
        app_state.set_vision_mode(body.vision_mode)
        return {"vision_mode": app_state.get_vision_mode()}

    @app.post("/camera_rotation")
    def rotate_camera():
        if not app_state.get_maintenance_mode():
            return JSONResponse(status_code=403, content={"error": "Not in maintenance mode"})
        app_state.rotate_camera()
        return {"camera_rotation": app_state.get_camera_rotation()}

    @app.get("/ocr_keyword")
    def get_ocr_keyword():
        return {"ocr_keyword": app_state.get_ocr_keyword()}

    @app.post("/ocr_keyword")
    def set_ocr_keyword(body: OcrKeywordBody):
        if not app_state.get_maintenance_mode():
            return JSONResponse(status_code=403, content={"error": "Not in maintenance mode"})
        new_keyword = body.ocr_keyword.strip()
        if not new_keyword:
            return JSONResponse(status_code=400, content={"error": "ocr_keyword cannot be empty"})
        app_state.set_ocr_keyword(new_keyword)
        return {"ocr_keyword": app_state.get_ocr_keyword()}

    @app.get("/maintenance_password")
    def get_maintenance_password():
        password = os.environ.get("MAINTENANCE_PASSWORD", "")
        return {"maintenance_password": password}

    @app.post("/load_classifier")
    def load_classifier(body: LoadClassifierBody):
        if not app_state.get_maintenance_mode():
            return JSONResponse(status_code=403, content={"error": "Not in maintenance mode"})
        model_path = body.model_path.strip()
        resolved_path = _resolve_repo_path(model_path)
        if not model_path or not resolved_path.exists():
            return JSONResponse(status_code=400, content={"error": "model_path does not exist"})
        classifier_loaded = bool(app_state.load_classifier(str(resolved_path)))
        return {"classifier_loaded": classifier_loaded, "model_path": str(resolved_path)}

    @app.get("/classifier_status")
    def get_classifier_status():
        status = app_state.get_classifier_status()
        return {
            "classifier_loaded": bool(status.get("loaded", False)),
            "model_path": status.get("model_path") or None,
        }

    @app.get("/video_feed")
    def video_feed():
        return StreamingResponse(
            generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/reset_counters")
    def reset_counters():
        app_state.reset_counters()
        return {"status": "counters reset"}

    @app.post("/dataset/capture")
    def dataset_capture_route(body: DatasetCaptureBody):
        label = body.label.lower()
        if label not in ("ok", "defective"):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "label must be 'ok' or 'defective'"},
            )

        frame = camera.get_frame()
        if frame is None:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "no frame available"},
            )

        # apply the same rotation logic as DatasetCapture
        rotated = dataset_capture._apply_rotation(frame)

        try:
            dataset_capture._save_frame(rotated, label)
        except Exception as exc:
            return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})

        return {
            "success": True,
            "ok": dataset_capture.counters.get("ok", 0),
            "defective": dataset_capture.counters.get("defective", 0),
        }

    @app.get("/dataset/counts")
    def dataset_counts():
        return dataset_capture.counters

    @app.get("/analytics")
    def analytics():
        template_path = Path(__file__).resolve().parent / "templates" / "analytics.html"
        return FileResponse(template_path)

    @app.get("/analytics/data")
    def analytics_data(date_param: str = Query(default="", alias="date")):
        try:
            result_dir = cfg.get("output", {}).get("result_dir", "data/results")
            output_dir = Path(result_dir)
            if not output_dir.is_absolute():
                output_dir = Path(__file__).resolve().parents[1] / output_dir

            # Parse requested date, default to today, clamp to 7-day window
            today = date.today()
            min_date = today - timedelta(days=6)  # 7 days including today

            raw_date = date_param
            try:
                requested_date = date.fromisoformat(raw_date) if raw_date else today
            except ValueError:
                requested_date = today

            # Clamp: never go beyond today or before 7 days ago
            if requested_date > today:
                requested_date = today
            if requested_date < min_date:
                requested_date = min_date

            daily_file = output_dir / f"{requested_date.isoformat()}.jsonl"

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

            return {
                "date": requested_date.isoformat(),
                "is_today": requested_date == today,
                "is_min_date": requested_date <= min_date,
                "total": total,
                "ok_count": ok_count,
                "nok_count": nok_count,
                "ok_rate": round(ok_count / total * 100, 1) if total > 0 else 0.0,
                "avg_processing_ms": avg_processing_ms,
                "timeline": timeline,
                "speed_timeline": speed_timeline,
            }

        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    return app