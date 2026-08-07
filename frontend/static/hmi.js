let CURRENT_MODE = "production";
let VISION_MODE = null;
let currentThreshold = null; // mirrors backend value
let LAST_APPLIED_MODE = null;
let LAST_DATASET_COUNTS_FETCH = 0;
const DATASET_COUNTS_POLL_MS = 5000;
let DATASET_CAPTURE_ACTIVE = false;

function updateVisionModeButtons() {
    const ocrBtn = document.getElementById("modeOcrBtn");
    const objBtn = document.getElementById("modeObjBtn");

    if (ocrBtn) {
        ocrBtn.classList.toggle("active", VISION_MODE === "ocr");
    }

    if (objBtn) {
        objBtn.classList.toggle("active", VISION_MODE === "object_detection");
    }
}

/* =========================
   RESULT POLLING
========================= */
async function updateResult() {
    try {
        const res = await fetch("/result");
        const data = await res.json();
        const result = data.result || {};
        const detections = Array.isArray(result.detections) ? result.detections : [];

        const statusEl = document.getElementById("status");
        const detEl = document.getElementById("detections");
        const dynamicLabelEl = document.getElementById("dynamicLabel");

        if (result.status === "OK") {
            statusEl.textContent = "OK";
            statusEl.className = "status ok";
        } else {
            statusEl.textContent = "NOK";
            statusEl.className = "status nok";
        }

        const noResultsText = VISION_MODE === "ocr"
            ? "No words detected"
            : "No objects detected";

        detEl.innerHTML = detections.length
            ? detections
                .map(d => {
                    const value = VISION_MODE === "ocr"
                        ? (d.text || d.label || "-")
                        : (d.label || d.text || "-");
                    const confidence = Number(d.confidence ?? 0);
                    return `${value} (${(confidence * 100).toFixed(1)}%)`;
                })
                .join("<br>")
            : noResultsText;

        if (dynamicLabelEl) {
            if (VISION_MODE === "ocr") {
                dynamicLabelEl.textContent = result.searched_word || "-";
            } else {
                dynamicLabelEl.textContent = result.label || "-";
            }
        }

        const timeValue = result.cycle_time_ms || result.processing_time_ms || 0;
        document.getElementById("time").textContent = timeValue + " ms";

        const counters = data.counters || { ok: 0, nok: 0, total: 0 };
        document.getElementById("okCount").textContent = counters.ok;
        document.getElementById("nokCount").textContent = counters.nok;
        document.getElementById("totalCount").textContent = counters.total;

        // Throttle dataset counts polling to once every DATASET_COUNTS_POLL_MS
        try {
            const now = Date.now();
            if (now - LAST_DATASET_COUNTS_FETCH >= DATASET_COUNTS_POLL_MS) {
                LAST_DATASET_COUNTS_FETCH = now;
                (async () => {
                    try {
                        const r = await fetch('/dataset/counts');
                        if (!r.ok) return;
                        const d = await r.json();
                        const okEl = document.getElementById('liveDatasetOkCount');
                        const defEl = document.getElementById('liveDatasetDefCount');
                        if (okEl && typeof d.ok !== 'undefined') okEl.textContent = d.ok;
                        if (defEl && typeof d.defective !== 'undefined') defEl.textContent = d.defective;
                    } catch (e) {
                        console.error('Failed to fetch dataset counts:', e);
                    }
                })();
            }
        } catch (e) {
            console.error(e);
        }

    } catch (e) {
        console.error(e);
    }
}

async function loadStatus() {
    const res = await fetch("/status");
    const data = await res.json();
    VISION_MODE = data.vision_mode;
    updateVisionModeButtons();
}

function getPollingIntervalMs() {
    return VISION_MODE === "ocr" ? 100 : 500;
}

async function startResultPolling() {
    await updateResult();
    setTimeout(startResultPolling, getPollingIntervalMs());
}

/* =========================
   THRESHOLD HANDLING
========================= */
async function loadThreshold() {
    try {
        const res = await fetch("/threshold");
        const data = await res.json();

        currentThreshold = data.threshold;
        document.getElementById("thresholdInput").value = currentThreshold;
    } catch (e) {
        console.error("Failed to load threshold:", e);
    }
}

async function loadOcrKeyword() {
    const res = await fetch("/ocr_keyword");
    const data = await res.json();
    document.getElementById("ocrKeywordInput").value = data.ocr_keyword;
}

function updateDatasetCaptureUI() {
    const datasetBox = document.getElementById("datasetCaptureBox");
    const instructions = document.getElementById("datasetCaptureInstructions");
    const toggleBtn = document.getElementById("captureToggleBtn");
    const counterDisplay = document.getElementById("datasetCaptureCounter");

    const visible = VISION_MODE === "object_detection";
    if (datasetBox) {
        datasetBox.style.display = visible ? "block" : "none";
    }

    if (!visible) {
        DATASET_CAPTURE_ACTIVE = false;
    }

    const active = visible && DATASET_CAPTURE_ACTIVE && CURRENT_MODE === "maintenance";

    if (toggleBtn) {
        toggleBtn.textContent = active ? "Stop Capture" : "Start Capture";
        toggleBtn.classList.toggle("capture-active", active);
        toggleBtn.disabled = CURRENT_MODE === "production";
    }

    if (instructions) {
        instructions.hidden = !active;
    }


    if (counterDisplay) {
        counterDisplay.hidden = !active;
    }
}

function startDatasetCapture() {
    if (VISION_MODE !== "object_detection") return;
    DATASET_CAPTURE_ACTIVE = true;
    updateDatasetCaptureUI();
}

function stopDatasetCapture() {
    DATASET_CAPTURE_ACTIVE = false;
    updateDatasetCaptureUI();
}

function toggleDatasetCapture() {
    if (DATASET_CAPTURE_ACTIVE) {
        stopDatasetCapture();
    } else {
        startDatasetCapture();
    }
}

/* =========================
   DATASET CAPTURE
========================= */
async function captureDatasetImage(label) {
    try {
        if (!DATASET_CAPTURE_ACTIVE) {
            console.error("Dataset capture is not active.");
            return;
        }

        const res = await fetch('/dataset/capture', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label })
        });

        const data = await res.json();
        if (data && data.success) {
            const okEl = document.getElementById('liveDatasetOkCount');
            const defEl = document.getElementById('liveDatasetDefCount');
            if (okEl && typeof data.ok !== 'undefined') okEl.textContent = data.ok;
            if (defEl && typeof data.defective !== 'undefined') defEl.textContent = data.defective;
        } else {
            console.error('Failed to capture dataset image:', data && data.error ? data.error : data);
        }
    } catch (e) {
        console.error('captureDatasetImage error:', e);
    }
}

document.getElementById("applyThreshold").addEventListener("click", async () => {
    if (CURRENT_MODE !== "maintenance") return;

    const value = parseFloat(document.getElementById("thresholdInput").value);

    if (isNaN(value) || value < 0 || value > 1) {
        alert("Threshold must be between 0.0 and 1.0");
        return;
    }

    try {
        await fetch("/threshold", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ threshold: value })
        });

        currentThreshold = value; // freeze this value for production
    } catch (e) {
        console.error("Failed to set threshold:", e);
    }
});

/* =========================
   MODE HANDLING
========================= */
function applyMode() {
    const banner = document.getElementById("modeBanner");
    const input = document.getElementById("thresholdInput");
    const applyBtn = document.getElementById("applyThreshold");
    const ocrApplyBtn = document.getElementById("applyOcrKeyword");
    const prodBtn = document.getElementById("startProductionBtn");
    const maintBtn = document.getElementById("startMaintenanceBtn");
    const visionModeToggle = document.getElementById("visionModeToggle");
    const ocrKeywordSection = document.getElementById("ocrKeywordSection");
    const cycleTimeSection = document.getElementById("cycleTimeSection");
    const rotateBtn = document.getElementById("rotateCameraBtn");

    if (rotateBtn) {
        rotateBtn.style.display = CURRENT_MODE === "maintenance" ? "inline-block" : "none";
    }

    if (CURRENT_MODE === "maintenance") {
        banner.textContent = "MAINTENANCE MODE";
        banner.className = "mode-overlay maintenance";

        input.disabled = false;
        applyBtn.disabled = false;
        if (ocrApplyBtn) ocrApplyBtn.disabled = false;
        const captureToggleBtn = document.getElementById("captureToggleBtn");
        if (captureToggleBtn) captureToggleBtn.disabled = false;
        prodBtn.style.display = "inline-block";
        maintBtn.style.display = "none";

        if (visionModeToggle) {
            visionModeToggle.style.pointerEvents = "auto";
            visionModeToggle.style.opacity = "1";
        }

        if (cycleTimeSection) {
            cycleTimeSection.style.display = "block";
        }

        if (currentThreshold !== null) input.value = currentThreshold;
    }

    if (CURRENT_MODE === "production") {
        banner.textContent = "PRODUCTION MODE";
        banner.className = "mode-overlay production";

        input.disabled = true;
        applyBtn.disabled = true;
        if (ocrApplyBtn) ocrApplyBtn.disabled = true;
        const captureToggleBtn = document.getElementById("captureToggleBtn");
        if (captureToggleBtn) captureToggleBtn.disabled = true;
        prodBtn.style.display = "none";
        maintBtn.style.display = "inline-block";

        if (visionModeToggle) {
            visionModeToggle.style.pointerEvents = "none";
            visionModeToggle.style.opacity = "0.4";
        }

        if (cycleTimeSection) {
            cycleTimeSection.style.display = "none";
        }

        if (currentThreshold !== null) input.value = currentThreshold;
    }

    LAST_APPLIED_MODE = CURRENT_MODE;

    if (ocrKeywordSection) {
        ocrKeywordSection.style.display = (CURRENT_MODE === "maintenance" && VISION_MODE === "ocr")
            ? "flex"
            : "none";
    }

    updateDatasetCaptureUI();
}

async function applyVisionMode(mode) {
    VISION_MODE = mode;
    updateVisionModeButtons();

    const ocrKeywordSection = document.getElementById("ocrKeywordSection");
    if (ocrKeywordSection) {
        ocrKeywordSection.style.display = (CURRENT_MODE === "maintenance" && VISION_MODE === "ocr")
            ? "flex"
            : "none";
    }

    updateDatasetCaptureUI();

    try {
        await fetch("/vision_mode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ vision_mode: mode })
        });

        await updateResult();
    } catch (e) {
        console.error("Failed to sync vision mode:", e);
    }
}

/* =========================
   BUTTON EVENTS
========================= */
document.getElementById("startProductionBtn").addEventListener("click", () => {
    if (!confirm("Start production mode?\n\nConfidence threshold will be locked.")) return;

    fetch("/maintenance_mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ maintenance_mode: false })
    }).then((res) => {
        if (!res.ok) {
            alert("Could not switch to production mode.");
            return;
        }

        CURRENT_MODE = "production";
        applyMode();
    }).catch((e) => {
        console.error("Failed to switch to production mode:", e);
        alert("Could not reach the server. Please try again.");
    });
});

document.getElementById("startMaintenanceBtn").addEventListener("click", async () => {
    const userPass = prompt("Enter password to enter maintenance mode:");
    if (userPass === null) return; // prompt cancelled

    // The password is checked server-side; the client never sees the real value.
    try {
        const res = await fetch("/maintenance_mode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ maintenance_mode: true, password: userPass })
        });

        if (!res.ok) {
            alert("Incorrect password. Access denied.");
            return;
        }
    } catch (e) {
        console.error("Failed to enter maintenance mode:", e);
        alert("Could not reach the server. Please try again.");
        return;
    }

    CURRENT_MODE = "maintenance";
    applyMode();
});

document.getElementById("modeOcrBtn").addEventListener("click", () => {
    if (CURRENT_MODE !== "maintenance") return;
    applyVisionMode("ocr");
});

document.getElementById("modeObjBtn").addEventListener("click", () => {
    if (CURRENT_MODE !== "maintenance") return;
    applyVisionMode("object_detection");
});

document.getElementById("rotateCameraBtn").addEventListener("click", async () => {
    if (CURRENT_MODE !== "maintenance") return;
    await fetch("/camera_rotation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
    });
});

document.getElementById("applyOcrKeyword").addEventListener("click", async () => {
    if (CURRENT_MODE !== "maintenance") return;
    const value = document.getElementById("ocrKeywordInput").value.trim();
    if (!value) { alert("Zoekwoord mag niet leeg zijn."); return; }
    await fetch("/ocr_keyword", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ocr_keyword: value })
    });
});

document.getElementById("resetBtn").addEventListener("click", async () => {
    if (!confirm("Are you sure you want to reset the counters?")) return;
    await fetch("/reset_counters", { method: "POST" });
});

document.getElementById("captureToggleBtn").addEventListener("click", () => {
    toggleDatasetCapture();
});


/* Keyboard shortcuts for dataset capture */
document.addEventListener("keydown", (e) => {
    const key = e.key.toLowerCase();

    if (key === "1" && DATASET_CAPTURE_ACTIVE) {
        e.preventDefault();
        captureDatasetImage("ok");
    } else if (key === "2" && DATASET_CAPTURE_ACTIVE) {
        e.preventDefault();
        captureDatasetImage("defective");
    } else if (key === "3" && DATASET_CAPTURE_ACTIVE) {
        e.preventDefault();
        stopDatasetCapture();
    }
});

/* =========================
    INIT
 ========================= */
async function init() {
    try {
        // Resolve mode first so all downstream behavior is mode-aware.
        await loadStatus();
        await loadThreshold();
        await loadOcrKeyword();
        applyMode();

        startResultPolling();
    } catch (e) {
        console.error("Initialization failed:", e);
    }
}

window.addEventListener("pageshow", (event) => {
    if (event.persisted) {
        window.location.reload();
    }
});

init();
