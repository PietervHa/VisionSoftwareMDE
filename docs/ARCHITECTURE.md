# Architecture Guide

This document provides a deep dive into VisionSoftwareMDE's design, components, and data flows.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Web Dashboard                         │
│                    (Flask + HTML/JavaScript)                 │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/JSON
┌─────────────────────▼────────────────────────────────────────┐
│                    Frontend Layer                             │
│  (web.py - /video_feed, /status, /threshold endpoints)      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   Application State                          │
│         (AppState - shared across threads)                  │
└────────────────┬─────────────────────────────────────────────┘
                │
    ┌───────────┼──────────────┐
    │           │              │
    ▼           ▼              ▼
┌─────────────┐ ┌──────────────┐ ┌─────────────────────┐
│   Camera    │ │ Vision Engine│ │ TCP Trigger Server  │
│  Service    │ │ (Background) │ │ (Optional)          │
│             │ │              │ │                     │
│ Capture     │ │ • OCR        │ │ Listens at port 5001│
│ frames      │ │ • Detection  │ │ Triggers vision &   │
│ from        │ │ • Inference  │ │ sends result        │
│ camera      │ │              │ │                     │
└─────────────┘ └──────────────┘ └─────────────────────┘
    │           │              │
    └───────────┼──────────────┘
                │
    ┌───────────▼────────────┐
    │  Config System         │
    │  (config/default.yaml) │
    │  - Vision mode         │
    │  - Detection backend   │
    │  - ROI / Camera params │
    │  - Trigger settings    │
    └────────────────────────┘
    
    ┌──────────────────────────────────────────┐
    │  Detection Engines                       │
    │  ┌──────────────────────────────────────┐│
    │  │ OCR Engine                           ││
    │  │ • Tesseract or PaddleOCR             ││
    │  │ • Text extraction & matching         ││
    │  └──────────────────────────────────────┘│
    │  ┌──────────────────────────────────────┐│
    │  │ Object Detection                     ││
    │  │ • Classifier (PyTorch)               ││
    │  │ • YOLO v8 (Ultralytics)              ││
    │  │ • Roboflow workflow                  ││
    │  └──────────────────────────────────────┘│
    └──────────────────────────────────────────┘
```

## Component Breakdown

### 1. Frontend (`frontend/web.py`)

**Responsibility:** Web UI and HTTP API

**Key Features:**
- Flask web server running on port 5000
- MJPEG video stream endpoint (`/video_feed`)
- REST API for configuration and control
- Static files serving (CSS, JS)
- Dashboard HTML templates

**Entry Points:**
- `GET /` - Web dashboard
- `GET /video_feed` - Live MJPEG stream
- `GET /status` - System status JSON
- `POST /threshold` - Update confidence threshold
- `POST /maintain/password` - Enter maintenance mode

See [API Reference](API.md) for complete endpoints.

### 2. Backend Core (`backend/core/`)

**Responsibility:** Central application logic and coordination

#### `main.py`

Entry point. Orchestrates:
1. Logging setup
2. App state initialization
3. Camera initialization
4. Web server startup (background thread)
5. TCP trigger server startup (if enabled)
6. Vision trigger loop (main thread)

Key thread coordination:
- Main thread: Waits for keyboard input (Q key) or TCP trigger
- Background thread 1: Web server (Flask)
- Background thread 2: Vision processing (as needed)

#### `state.py` (`AppState` class)

Shared state container for thread-safe data access:
- Current vision mode and mode parameters
- Confidence threshold
- Last detection result
- Counters (OK/NOK/Error)
- Classifier load state

**Thread Safety:** Uses locks for concurrent access.

#### `config_loader.py`

Configuration system:
- Loads YAML configuration from `config/default.yaml`
- Validates configuration at startup
- Provides global `cfg` dictionary
- Hot-reload capability (not fully implemented)

#### `camera.py` (`Camera` class)

Camera interaction:
- Initializes camera device (OpenCV)
- Continuous frame capture in background thread
- Frame buffering (single frame buffer)
- Resolution configuration
- Flip/rotation support

**Thread Model:** Camera runs in background thread, main code reads latest frame via `get_frame()`.

#### `vision.py`

Vision pipeline orchestration:
- Routes to OCR or object detection based on config
- Manages OCR instance
- Manages inspection engine
- Handles model loading (classifier)
- ThreadPoolExecutor for background vision processing
- Callback mechanism for results

**Key Functions:**
- `bind_app_state(app_state)` - Initialize with app state
- `run_vision(frame, callback)` - Trigger vision (async)
- `_sync_classifier_state()` - Update app state with model status

#### `inspection_engine.py` (`InspectionEngine` class)

Decision engine that interprets detection results:
- Takes detection output (confidence, labels, boxes)
- Compares against threshold
- Determines overall result (OK / NOK / ERROR)
- Formats result dictionary
- Handles special cases (no objects detected, model errors)

#### `tcp_trigger_server.py` (`TCPTriggerServer` class)

Optional PLC integration:
- Listens on configured port (default 5001)
- Waits for trigger byte (configurable, default `0x01`)
- Triggers vision cycle on receipt
- Waits for result (with timeout)
- Responds with configured response (`OK\n` or `NOK\n`)

Useful for industrial systems:
- PLC sends trigger byte
- Server processes vision
- PLC receives result

### 3. Detection Engines (`backend/detection/`)

#### `ocr/` - OCR Implementation

**Supported Engines:**
- **Tesseract**: System command-line OCR engine
- **PaddleOCR**: Python-based, no installation needed

**Process:**
1. Extract ROI from frame
2. Preprocess image (threshold, upscaling, etc.)
3. Run OCR engine
4. Extract text
5. Match against keywords and regex patterns
6. Return confidence and matched keywords

**Debug Output:** OCR debug images are saved to `debug_ocr/` folder.

#### `objectdetection/` - Object Detection Implementation

**Supported Backends:**

| Backend | Source | Notes |
|---------|--------|-------|
| `classifier` | Local PyTorch model | Train with `train_classifier.py` |
| `yolo` | Ultralytics YOLO | Pre-trained models available |
| `roboflow` | Roboflow cloud API | Requires API credentials |

**Input:** Frame (full or ROI)
**Output:** `{status, confidence, detections, labels}`

**Preprocessing:**
- ROI extraction
- Resizing to model input size
- Normalization

### 4. Output (`backend/output/`)

#### `result_writer.py`

Handles result persistence:
- Formats vision results as JSON
- Saves to `data/results/` with timestamp
- One file per inspection cycle
- Includes cycle time, confidence, detections

**Result Format:**
```json
{
  "timestamp": "2026-06-04T10:30:45.123456",
  "vision_mode": "object_detection",
  "status": "OK",
  "confidence": 0.95,
  "confidence_threshold": 0.8,
  "cycle_time_ms": 125.4,
  "detections": [
    {
      "class": "ok",
      "confidence": 0.98,
      "box": [100, 200, 350, 450]
    }
  ]
}
```

### 5. Utilities (`backend/utils/`)

#### `logger.py`

Centralized logging:
- Rotating file handlers
- Console output
- Configurable log levels
- Global logger instance

Logs saved to: `data/logs/app_YYYYMMDD.log`

#### `roi.py`

ROI utilities:
- Extract ROI from frame
- Draw ROI box on display
- Normalize ROI coordinates

## Data Flow

### Typical Vision Cycle

```
1. USER TRIGGERS (Q key or TCP request)
   ↓
2. CAPTURE FRAME from camera
   ↓
3. START vision background thread
   ├─ Extract ROI
   ├─ Preprocess
   ├─ Run inference (OCR/Detection)
   ├─ Get result (confidence, labels)
   ├─ Run inspection engine → OK/NOK decision
   └─ Format result JSON
   ↓
4. RESULT CALLBACK
   ├─ Update AppState
   ├─ Update counters
   ├─ Save to disk
   └─ Log result
   ↓
5. WEB DASHBOARD shows result
   └─ TCP client receives response (if enabled)

Typical cycle time: 50-300ms (depending on model)
```

## Threading Model

**Thread 1 (Main):**
- Starts up everything
- Runs vision trigger loop
- Listens for Q key
- Calls `run_vision()` with callback

**Thread 2 (Web Server):**
- Flask app listening on port 5000
- Handles HTTP requests
- Serves video stream
- Updates shared AppState (with locks)

**Thread 3+ (Vision Workers):**
- ThreadPoolExecutor manages vision processing
- Runs inference in background
- Calls callback when done
- Main thread not blocked

**Thread 4 (Optional - TCP):**
- TCP trigger server
- Listens for PLC connections
- Triggers vision similar to main thread

**Thread 5 (Camera):**
- Continuously captures frames
- Stores in ring buffer

**Synchronization:**
- `AppState` uses `threading.Lock` for critical sections
- Vision trigger marked with `threading.Event` to prevent overlapping
- Callbacks ensure results processed sequentially

## Configuration Flow

```
config/default.yaml
    ↓
config_loader.py imports as `cfg`
    ↓
Main.py reads cfg
    ├─ Camera initialization
    ├─ Vision mode selection
    ├─ Model path resolution
    ├─ ROI configuration
    └─ TCP trigger setup
    ↓
Runtime: cfg accessed via global import
    ├─ Vision.py reads detection backend
    ├─ Camera.py reads resolution
    ├─ Frontend/web.py reads HMI settings
    └─ TCP server reads port/trigger settings
```

## Model Loading

### Classifier Model Loading

1. **On Startup:**
   - If `backend: "classifier"`, auto-load model from `classifier.model_path`
   - Models are PyTorch/HuggingFace pipelines
   - Takes 5-30 seconds depending on model size

2. **Runtime:**
   - POST `/load_classifier` endpoint
   - Loads new model at runtime
   - Previous model unloaded
   - Useful for testing multiple models

3. **Model Structure:**
   ```
   models/classifier/final/
   ├── config.json
   ├── generation_config.json
   ├── model.safetensors
   ├── preprocessor_config.json
   ├── pytorch_model.bin
   └── tokenizer_config.json
   ```

### YOLO Model Loading

- Model file is `.pt` (PyTorch)
- Loaded once on startup
- Inference size configured (default 640)
- Ultralytics library handles loading

### Roboflow

- No local model
- REST API calls to Roboflow inference server
- Requires internet connectivity

## Performance Considerations

### Bottlenecks

1. **Inference Time** (~60-80% of cycle)
   - Classifier inference: 50-200ms
   - YOLO inference: 50-150ms
   - Tesseract OCR: 100-500ms

2. **Preprocessing** (~10-20% of cycle)
   - ROI extraction: fast
   - Image resize/normalize: fast
   - OCR preprocessing (threshold, upscale): can be slow

3. **Frame Capture** (~5-10% of cycle)
   - Camera frame read: depends on camera

### Optimization Strategies

1. **Model Size:** Smaller models (nano, tiny) are faster
2. **Preprocessing:** "fast" mode in config
3. **ROI Size:** Smaller ROI → faster inference
4. **Batch Processing:** Not currently implemented
5. **GPU:** Use CUDA for significantly faster inference

See [Benchmarking](DEVELOPMENT.md#benchmarking) for performance measurements.

## Error Handling

**Vision Processing Errors:**
- Caught in `vision_trigger_loop`
- Logged with full stack trace
- Result shows `status: "ERROR"`
- Vision thread flag cleared
- Application continues

**Configuration Errors:**
- Validated on startup
- Fatal error if critical (e.g., missing model_path)
- Application exits with error message

**Camera Errors:**
- Logged on first occurrence
- Application continues, returns null frames
- Vision cycle skipped if frame unavailable

## Extensibility Points

### Adding a New Detection Backend

1. Implement inference function in `backend/detection/objectdetection/`
2. Update `backend/detection/__init__.py` to export it
3. Update `backend/core/vision.py` to handle new backend
4. Add configuration section to `config/default.yaml`
5. Implement in inspection engine if needed

### Adding a New OCR Engine

1. Implement in `backend/detection/ocr/`
2. Match interface of existing engines
3. Update ocr.py engine dispatcher
4. Add config section for new engine parameters

### Custom Result Processing

Override `backend/output/result_writer.py`:
- Save to database instead of JSON
- Stream to webhook
- Format for specific system requirements

## State Diagram

```
[IDLE] 
  │
  ├─ Q key pressed
  │
  ▼
[CAPTURING FRAME]
  │
  ▼
[VISION_BUSY] (flag set)
  │
  ├─ Background thread: inference
  │
  ▼
[PROCESSING RESULT]
  │
  ├─ Update AppState
  ├─ Save to disk
  ├─ Log result
  │
  ▼
[VISION_BUSY] (flag cleared)
  │
  ▼
[IDLE]
  │
  └─ Ready for next trigger
```

## Deployment Scenarios

### Scenario 1: Standalone Testing
- Run on development PC
- Manual trigger with Q key
- View results on web dashboard

### Scenario 2: Factory Floor Integration
- Run on edge PC in factory
- PLC integration via TCP trigger
- Results logged to database
- Dashboard monitored remotely

### Scenario 3: Production Line
- Multiple instances on networked PCs
- Coordinated via factory MES
- Results aggregated remotely
- High-availability setup


