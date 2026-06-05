# Configuration Guide

All settings are managed through `config/default.yaml`. Restart the application after any change.

---

## Quick reference

| Key | Default | Description |
|-----|---------|-------------|
| `vision_mode` | `"ocr"` | `"ocr"` or `"object_detection"` |
| `machine_id` | `"default"` | Identifier used in logs and result files |
| `confidence_threshold` | `0.8` | Min confidence for OK result (0.0–1.0), adjustable at runtime via dashboard |
| `object_detection.backend` | `"classifier"` | `"classifier"`, `"yolo"`, `"template"`, or `"roboflow"` |
| `ocr.engine` | `"tesseract"` | `"tesseract"` or `"paddleocr"` |
| `ocr.preprocess` | `"fast"` | `"fast"` (~50ms) or `"accurate"` (~200ms) |
| `camera.index` | `0` | USB camera index; run `tests/camera_test.py` to find yours |
| `camera.flip` | `-1` | `0` = none, `1` = horizontal, `-1` = vertical |
| `trigger.enabled` | `false` | Set to `true` to activate TCP trigger for PLC |
| `trigger.port` | `5001` | Port the PLC connects to |
| `trigger.timeout_s` | `5.0` | Seconds to wait for result before responding NOK |
| `hmi.enable_video_feed` | `true` | Disable to reduce CPU/bandwidth |
| `web.port` | `5000` | Port for the web dashboard |

---

## Region of Interest (ROI)

Coordinates are normalized 0.0–1.0 where `(0,0)` is top-left and `(1,1)` is bottom-right. The ROI box is drawn live on the dashboard — adjust until it tightly covers the inspection target.

```yaml
roi:
  x_start: 0.2
  y_start: 0.7
  x_end: 0.8
  y_end: 0.9
```

```
   0%    20%          80%   100%
0% +-----+------------+-----+
   |     |            |     |
70%|     +============+     |  ← ROI starts here
   |     |  inspect   |     |
90%|     +============+     |  ← ROI ends here
   |     |            |     |
100+-----+------------+-----+
```

---

## OCR settings

Applies when `vision_mode: "ocr"`. The most commonly tuned options:

```yaml
ocr:
  engine: "tesseract"           # or "paddleocr"
  keywords:
    - "MADE IN GERMANY"         # OK if any keyword is found in extracted text
  date_regex: '\b\d{2}/\d{2}/\d{4}\b'  # extracted dates appear in result
  preprocess: "fast"            # "accurate" for difficult images
  psm: 7                        # 7 = single line, 3 = automatic (slower)
  oem: 3                        # 3 = auto, 1 = LSTM only
  whitelist: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789."
  tesseract_path: "C:/Program Files/Tesseract-OCR/tesseract.exe"
```

Switch to PaddleOCR (no installation required) by setting `engine: "paddleocr"`. All other OCR keys are ignored when using PaddleOCR.

---

## Object detection settings

Applies when `vision_mode: "object_detection"`.

```yaml
object_detection:
  backend: "classifier"         # see options below
  classifier:
    model_path: "./models/classifier/final"
  model_path: "models/yolov8n.pt"   # used when backend is "yolo"
  inference_size: 640               # YOLO input resolution
  template:
    references:
      - "data/templates/template_xyz.jpg"
    match_threshold: 0.6
  roboflow:
    model: "cola_detectie"          # profile name to use
    models:
      cola_detectie:
        api_key: "SET_VIA_ENV"      # use ROBOFLOW_API_KEY env variable
        workspace: "your-workspace"
        workflow: "detect-count-and-visualize"
```

The classifier model directory must contain `config.json`, `model.safetensors`, and `preprocessor_config.json`. After retraining, update `classifier.model_path` to the new checkpoint folder.

---

## Secrets

Never commit API keys or passwords to Git. Set them via environment variables in a `.env` file (see `.env.example`):

```
ROBOFLOW_API_KEY=your_key_here
MAINTENANCE_PASSWORD=your_password_here
```

In `config/default.yaml`, leave the placeholders:

```yaml
roboflow:
  models:
    cola_detectie:
      api_key: "SET_VIA_ENV"

security:
  maintenance_password: "SET_VIA_ENV"
```

---

## Startup validation errors

| Error | Cause | Fix |
|-------|-------|-----|
| `model_path does not exist` | Path to classifier or YOLO model not found | Check path is relative to repo root and model has been trained |
| `invalid backend` | Unknown backend name | Valid values: `"classifier"`, `"yolo"`, `"template"`, `"roboflow"` |
| `classifier profile not found` | `roboflow.model` not in `roboflow.models` | Add the profile or correct the name |
| `ROI coordinates invalid` | ROI values outside 0–1 | All four ROI values must be between 0.0 and 1.0 |

---

## Complete configuration examples

### OCR — reading a product label

```yaml
vision_mode: "ocr"
confidence_threshold: 0.85

roi:
  x_start: 0.1
  y_start: 0.3
  x_end: 0.9
  y_end: 0.7

camera:
  index: 0
  width: 1920
  height: 1440
  flip: -1

ocr:
  engine: "tesseract"
  keywords:
    - "MADE IN GERMANY"
    - "CE"
  date_regex: '\b\d{2}/\d{2}/\d{4}\b'
  preprocess: "accurate"
  psm: 3
```

### Local classifier — OK/defective detection

```yaml
vision_mode: "object_detection"
confidence_threshold: 0.75

object_detection:
  backend: "classifier"
  classifier:
    model_path: "models/classifier/final"

camera:
  index: 0
  width: 1280
  height: 1024
  flip: -1
```

### YOLO with PLC trigger

```yaml
vision_mode: "object_detection"
confidence_threshold: 0.6

object_detection:
  backend: "yolo"
  model_path: "models/yolov8n.pt"
  inference_size: 640

trigger:
  enabled: true
  host: "0.0.0.0"
  port: 5001
  trigger_byte: "0x01"
  response_ok: "OK\n"
  response_nok: "NOK\n"
  timeout_s: 3.0
```

### Roboflow cloud inference

```yaml
vision_mode: "object_detection"
confidence_threshold: 0.7

object_detection:
  backend: "roboflow"
  roboflow:
    model: "cola_detectie"
    models:
      cola_detectie:
        api_key: "SET_VIA_ENV"
        workspace: "pieters-workspace-kugm8"
        workflow: "detect-count-and-visualize"
        api_url: "https://serverless.roboflow.com"
```