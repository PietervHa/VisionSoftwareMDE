# Configuration Guide

All VisionSoftwareMDE settings are managed through `config/default.yaml`. This guide explains every configuration option with examples and best practices.

## Configuration File Location

```
config/default.yaml
```

**To use a different config file:**
```powershell
# Set environment variable (currently not implemented, but infrastructure ready)
$env:VISION_CONFIG = "C:\path\to\custom_config.yaml"
python -m backend.main
```

## Top-Level Settings

### `vision_mode`

**Type:** `string`  
**Valid values:** `"ocr"` | `"object_detection"`  
**Default:** `"object_detection"`

Selects the primary vision task.

```yaml
vision_mode: "ocr"              # Extract text from ROI
vision_mode: "object_detection" # Classify or detect objects
```

### `machine_id`

**Type:** `string`  
**Default:** `"default"`

Identifier for this machine. Used in logs and results to distinguish between multiple instances.

```yaml
machine_id: "line_1"     # Used in result files and logs
machine_id: "station_A"
```

### `confidence_threshold`

**Type:** `float (0.0 - 1.0)`  
**Default:** `0.8`

Minimum confidence required for a positive detection. Values below this threshold are considered failures (NOK).

```yaml
confidence_threshold: 0.8    # 80% confidence required
confidence_threshold: 0.5    # More lenient
confidence_threshold: 0.95   # Very strict
```

Can be changed at runtime via web dashboard or API.

---

## Region of Interest (ROI)

Defines the inspection area within the camera frame. Useful for focusing on a specific product area and ignoring background.

### `roi`

All coordinates are **normalized** (0.0 to 1.0), where:
- `(0, 0)` = top-left corner
- `(1, 1)` = bottom-right corner

```yaml
roi:
  x_start: 0.2    # Start at 20% from left
  y_start: 0.7    # Start at 70% from top
  x_end: 0.8      # End at 80% from left
  y_end: 0.9      # End at 90% from top
```

**Visual Example:**
```
   0%    20%          80%   100%
   |     |            |     |
0% +-----+------------+-----+
   |     |            |     |
50%|     |            |     |
   |     |            |     |
70%|     +============+     |  } ROI area
   |     |            |     |
90%|     +============+     |
   |     |            |     |
100+-----+------------+-----+
```

**Best Practice:** Set ROI to tightly cover your inspection target to improve performance and reduce false positives.

---

## Camera Settings

### `camera`

```yaml
camera:
  index: 0       # Camera device index (0 = default)
  width: 1280    # Frame width in pixels
  height: 1024   # Frame height in pixels
  flip: -1       # Rotation/flip option
```

#### `camera.index`

**Type:** `integer`

Device index for camera selection:
- `0` = Primary/built-in camera
- `1`, `2`, etc. = Additional cameras (USB webcams)

**How to find your camera:**
```powershell
python .\tests\camera_test.py
# Will list available cameras
```

#### `camera.width` and `camera.height`

**Type:** `integer`

Frame resolution. Common options:
- `640 × 480` - Low (fast)
- `1280 × 1024` - Medium (recommended)
- `1920 × 1440` - High (slower, more detail)

Higher resolution = better detail but slower processing.

#### `camera.flip`

**Type:** `integer`

Image transformation:
- `0` = No flip
- `1` = Horizontal flip
- `-1` = Vertical flip (default)
- `2` or `3` = Other rotations

Use this if your camera is mounted upside-down or rotated.

---

## OCR Settings

### `ocr`

Applies only when `vision_mode: "ocr"`

```yaml
ocr:
  engine: "tesseract"
  keywords:
    - "www.theimagingsource.com"
    - "PASS"
  date_regex: '\b\d{2}/\d{2}/\d{4}\b'
  preprocess: "fast"
  downscale: 0.6
  min_dim: 240
  psm: 7
  oem: 3
  whitelist: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789."
  disable_dawgs: true
  tesseract_path: "C:/Program Files/Tesseract-OCR/tesseract.exe"
```

#### `ocr.engine`

**Type:** `string`  
**Valid values:** `"tesseract"` | `"paddleocr"`

- `"tesseract"` - System command-line OCR (requires Tesseract installation)
- `"paddleocr"` - Python OCR engine (auto-downloads model)

**Example: Switch to PaddleOCR:**
```yaml
ocr:
  engine: "paddleocr"
```

#### `ocr.keywords`

**Type:** `list of strings`

Text patterns to search for in extracted OCR text. Result is OK if any keyword is found.

```yaml
keywords:
  - "MADE IN GERMANY"
  - "GERMANY"
  - "MIG"
```

#### `ocr.date_regex`

**Type:** `string` (regex pattern)

Regex pattern to search for dates. Extracted dates are included in result.

```yaml
date_regex: '\b\d{2}/\d{2}/\d{4}\b'  # DD/MM/YYYY or MM/DD/YYYY
date_regex: '\d{4}-\d{2}-\d{2}'      # YYYY-MM-DD
```

#### `ocr.preprocess`

**Type:** `string`  
**Valid values:** `"fast"` | `"accurate"`

Preprocessing mode:
- `"fast"` - Quick preprocessing, good for clear text
- `"accurate"` - Intensive preprocessing, better for poor quality images

```yaml
preprocess: "fast"       # ~50ms per frame
preprocess: "accurate"   # ~200ms per frame
```

#### `ocr.downscale`

**Type:** `float`

Downsample image before text extraction. Reduces detail but speeds up processing.

```yaml
downscale: 0.6   # Use 60% of original size (faster)
downscale: 1.0   # Use full resolution (slower, more detail)
```

#### `ocr.min_dim`

**Type:** `integer`

Minimum dimension after preprocessing. Images smaller than this are upscaled.

```yaml
min_dim: 240   # Upscale if smaller than 240px
```

#### `ocr.psm` (Tesseract only)

**Type:** `integer`

Tesseract Page Segmentation Mode:

- `0` = Orientation and script detection only
- `3` = Fully automatic (slow, thorough)
- `6` = Uniform block of text
- `7` = Single text line
- `11` = Sparse text
- `13` = Raw line

```yaml
psm: 7  # Single line of text (fast)
psm: 3  # Automatic (slow, accurate)
```

#### `ocr.oem` (Tesseract only)

**Type:** `integer`

Tesseract OCR Engine Mode:

- `0` = Original Tesseract
- `1` = Neural nets LSTM
- `2` = Both
- `3` = Default (auto-select)

```yaml
oem: 3  # Automatic (recommended)
oem: 1  # LSTM (better accuracy for modern text)
```

#### `ocr.whitelist`

**Type:** `string`

Allowed characters. OCR will only output these characters.

```yaml
whitelist: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789."
whitelist: "0123456789.-"  # Only digits, dots, hyphens
```

#### `ocr.disable_dawgs`

**Type:** `boolean`

Disable dictionary and word lists (Tesseract becomes more aggressive).

```yaml
disable_dawgs: true   # Disable dictionaries
disable_dawgs: false  # Use dictionaries
```

#### `ocr.tesseract_path`

**Type:** `string` (file path)

Path to Tesseract executable (Windows). Required if using `engine: "tesseract"`.

```yaml
tesseract_path: "C:/Program Files/Tesseract-OCR/tesseract.exe"
```

---

## Object Detection Settings

### `object_detection`

Applies only when `vision_mode: "object_detection"`

#### `object_detection.backend`

**Type:** `string`  
**Valid values:** `"classifier"` | `"yolo"` | `"roboflow"`

Selects which detection engine to use.

```yaml
object_detection:
  backend: "classifier"  # Local trained model
  backend: "yolo"        # Local YOLO model
  backend: "roboflow"    # Cloud Roboflow API
```

### Classifier Backend

For fine-tuned image classification (OK/defective).

```yaml
object_detection:
  backend: "classifier"
  classifier:
    model_path: "./models/classifier/final"
```

#### `classifier.model_path`

**Type:** `string` (directory path)

Path to trained classifier model. Must contain:
- `config.json`
- `model.safetensors` or `pytorch_model.bin`
- `preprocessor_config.json`

```yaml
model_path: "models/classifier/final"
model_path: "models/classifier/epoch-5"
model_path: "/absolute/path/to/model"
```

**Re-training:** Update model path to point to a new trained model.

### YOLO Backend

For general object detection using Ultralytics YOLO.

```yaml
object_detection:
  backend: "yolo"
  model_path: "models/yolov8n.pt"
  inference_size: 640
```

#### `model_path` (YOLO)

**Type:** `string` (file path)

YOLO model checkpoint (`.pt` file).

Available pre-trained models:
- `yolov8n.pt` - Nano (fastest)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (most accurate)

```yaml
model_path: "models/yolov8n.pt"  # Fast, included
model_path: "models/yolov8m.pt"  # Medium speed/accuracy
```

#### `inference_size` (YOLO)

**Type:** `integer`

Input resolution for YOLO inference. Larger = more accurate but slower.

```yaml
inference_size: 320   # Fast, less accurate
inference_size: 640   # Medium (default)
inference_size: 1280  # Slow, very accurate
```

### Roboflow Backend

For cloud-based inference workflows.

```yaml
object_detection:
  backend: "roboflow"
  roboflow:
    model: "cola_detectie"  # Selected profile
    models:
      cola_detectie:
        api_key: "YOUR_API_KEY"
        workspace: "pieters-workspace-kugm8"
        workflow: "detect-count-and-visualize"
        api_url: "https://serverless.roboflow.com"
      another_model:
        api_key: "ANOTHER_KEY"
        workspace: "another-workspace"
        workflow: "another-workflow"
```

#### `roboflow.model`

**Type:** `string`

Name of the profile to use (must exist in `roboflow.models`).

```yaml
roboflow:
  model: "cola_detectie"  # Use this profile
```

#### `roboflow.models.<profile_name>`

Each profile requires:

| Key | Type | Description |
|-----|------|-------------|
| `api_key` | string | Roboflow API key |
| `workspace` | string | Roboflow workspace name |
| `workflow` | string | Workflow name in Roboflow |
| `api_url` | string | API endpoint (optional, default: `https://serverless.roboflow.com`) |

```yaml
models:
  my_detector:
    api_key: "abc123xyz"
    workspace: "my-workspace"
    workflow: "detect-and-classify"
    api_url: "https://serverless.roboflow.com"
```

---

## HMI Settings

### `hmi`

Web dashboard settings.

```yaml
hmi:
  debug_draw_roi: true
  enable_video_feed: true
  stream_quality: 75
```

#### `hmi.debug_draw_roi`

**Type:** `boolean`

Draw ROI rectangle on video feed.

```yaml
debug_draw_roi: true   # Show ROI box
debug_draw_roi: false  # Hide ROI box
```

#### `hmi.enable_video_feed`

**Type:** `boolean`

Enable live video feed on web dashboard.

```yaml
enable_video_feed: true   # Stream video (faster network, higher CPU)
enable_video_feed: false  # Don't stream (lower bandwidth)
```

#### `hmi.stream_quality`

**Type:** `integer (0-100)`

JPEG quality of streamed video.

```yaml
stream_quality: 100  # High quality (larger files, higher bandwidth)
stream_quality: 75   # Medium quality (balanced)
stream_quality: 50   # Low quality (smaller files, lower bandwidth)
```

---

## Web Server Settings

### `web`

```yaml
web:
  host: "0.0.0.0"
  port: 5000
```

#### `web.host`

**Type:** `string`

Bind address for Flask web server.

```yaml
host: "0.0.0.0"    # Listen on all interfaces (accessible remotely)
host: "127.0.0.1"  # Listen only on localhost
host: "192.168.1.100"  # Bind to specific IP
```

#### `web.port`

**Type:** `integer`

HTTP port number.

```yaml
port: 5000  # http://localhost:5000
port: 8080  # http://localhost:8080
port: 80    # http://localhost (requires admin privileges on Windows)
```

---

## Output Settings

### `output`

Result persistence.

```yaml
output:
  result_dir: "data/results"
```

#### `output.result_dir`

**Type:** `string` (directory path)

Where to save vision results as JSON files.

```yaml
result_dir: "data/results"              # Relative to repo root
result_dir: "/absolute/path/to/results" # Absolute path
```

Results are named with timestamp: `result_2026-06-04_10-30-45.json`

---

## Trigger Settings

### `trigger`

PLC/Industrial integration via TCP.

```yaml
trigger:
  enabled: false
  type: "tcp"
  host: "0.0.0.0"
  port: 5001
  trigger_byte: "0x01"
  response_ok: "OK\n"
  response_nok: "NOK\n"
  timeout_s: 5.0
```

#### `trigger.enabled`

**Type:** `boolean`

Enable TCP trigger server.

```yaml
enabled: false  # Disable TCP server
enabled: true   # Enable TCP server (listen on port 5001)
```

#### `trigger.type`

**Type:** `string`

Currently only TCP is supported.

```yaml
type: "tcp"
```

#### `trigger.host`

**Type:** `string`

Bind address for trigger server.

```yaml
host: "0.0.0.0"      # Listen on all interfaces
host: "127.0.0.1"    # Localhost only
host: "192.168.1.50" # Specific interface
```

#### `trigger.port`

**Type:** `integer`

Port for TCP connections.

```yaml
port: 5001
port: 9999
```

#### `trigger.trigger_byte`

**Type:** `string` (hex format)

Byte that PLC sends to trigger vision.

```yaml
trigger_byte: "0x01"   # Decimal 1
trigger_byte: "0xFF"   # Decimal 255
trigger_byte: "0x42"   # ASCII 'B'
```

#### `trigger.response_ok`

**Type:** `string`

Response sent back to PLC when result is OK.

```yaml
response_ok: "OK\n"
response_ok: "PASS"
response_ok: "1"
```

#### `trigger.response_nok`

**Type:** `string`

Response sent back when result is NOK.

```yaml
response_nok: "NOK\n"
response_nok: "FAIL"
response_nok: "0"
```

#### `trigger.timeout_s`

**Type:** `float`

Maximum seconds to wait for vision result before responding NOK.

```yaml
timeout_s: 5.0   # Wait max 5 seconds
timeout_s: 10.0  # Wait max 10 seconds
```

---

## Security Settings

### `security`

```yaml
security:
  maintenance_password: "SET_VIA_ENV"
```

#### `security.maintenance_password`

**Type:** `string`

Password required to enter maintenance mode and access sensitive APIs.

Set this value via the `MAINTENANCE_PASSWORD` environment variable in the project root `.env` file.

```yaml
maintenance_password: "SET_VIA_ENV"
```

**Usage:**
```powershell
# Enter maintenance mode
Invoke-RestMethod -Method Post http://localhost:5000/maintain/password `
  -Body '{"password":"your_password_here"}' `
  -ContentType "application/json"
```

**Security Note:** Change this password in production! Store sensitive values using environment variables or secrets management.
The application loads `.env` automatically at startup, so the recommended local setup is to copy `.env.example` to `.env` and fill in your own values.

---

## Configuration Examples

### Example 1: OCR Setup

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

ocr:
  engine: "tesseract"
  keywords:
    - "MADE IN GERMANY"
    - "CE"
  date_regex: '\b\d{2}/\d{2}/\d{4}\b'
  preprocess: "accurate"
  psm: 3
```

### Example 2: Local Classifier

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
```

### Example 3: YOLO Detection with PLC Integration

```yaml
vision_mode: "object_detection"
confidence_threshold: 0.6

object_detection:
  backend: "yolo"
  model_path: "models/yolov8m.pt"
  inference_size: 640

trigger:
  enabled: true
  host: "0.0.0.0"
  port: 5001
  trigger_byte: "0x01"
  timeout_s: 3.0
```

### Example 4: Roboflow Cloud Detection

```yaml
vision_mode: "object_detection"
confidence_threshold: 0.7

object_detection:
  backend: "roboflow"
  roboflow:
    model: "production_detector"
    models:
      production_detector:
        api_key: "YOUR_API_KEY_HERE"
        workspace: "your-workspace"
        workflow: "quality-check"
```

---

## Configuration Validation

The app validates configuration on startup. If invalid:
1. Error message logged to console and file
2. Application exits
3. Check logs in `data/logs/`

**Common Validation Errors:**

| Error | Cause | Solution |
|-------|-------|----------|
| `model_path does not exist` | Path points to non-existent directory | Verify path exists and is correct |
| `classifier profile not found` | Roboflow profile missing | Add profile to `roboflow.models` |
| `invalid backend` | Unknown backend specified | Use valid: "classifier", "yolo", "roboflow" |
| `ROI coordinates invalid` | ROI values outside 0-1 range | Check ROI values are between 0.0 and 1.0 |

---

## Environment Variables

Currently, configuration is read-only from `config/default.yaml`. Future support planned for:

```powershell
$env:VISION_CONFIDENCE_THRESHOLD = "0.85"
$env:VISION_MODE = "ocr"
```

---

## Hot Reload

Configuration can potentially be reloaded at runtime via API (infrastructure in place, not fully implemented).

For now, restart the application to pick up configuration changes:

```powershell
# Ctrl+C to stop
# Then restart
python -m backend.main
```

---

## Configuration Best Practices

1. **Version Control:** Keep `config/default.yaml` in Git
2. **Environment-Specific:** Create `config/production.yaml` if needed
3. **Secrets:** Never commit API keys; use environment variables or secret managers
4. **ROI Tuning:** Test ROI visually before running production
5. **Threshold Tuning:** Start high (0.9), lower gradually based on results
6. **Performance:** Monitor cycle times; adjust preprocessing, ROI, or model size
7. **Backup:** Keep working configurations backed up before experiments

---

## Resetting to Defaults

To reset to default configuration:

```powershell
# Restore from Git
git checkout config/default.yaml

# Or manually copy from reference
Copy-Item config/default.yaml config/default.yaml.bak
```

