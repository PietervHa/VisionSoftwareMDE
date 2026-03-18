let CURRENT_MODE = "maintenance";
let VISION_MODE = null;
let currentThreshold = null; // mirrors backend value
let LAST_SYNCED_MODE = null;
let LAST_APPLIED_MODE = null;
const PASSWORD = "@Welkom01"; // hardcoded for now

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

        const statusEl = document.getElementById("status");
        const detEl = document.getElementById("detections");
        const dynamicLabelEl = document.getElementById("dynamicLabel");

        if (data.result.status === "OK") {
            statusEl.textContent = "OK";
            statusEl.className = "status ok";
        } else {
            statusEl.textContent = "NOK";
            statusEl.className = "status nok";
        }

        const noResultsText = VISION_MODE === "ocr"
            ? "No words detected"
            : "No objects detected";

        detEl.innerHTML = data.result.detections.length
            ? data.result.detections
                .map(d => {
                    const value = VISION_MODE === "ocr" ? d.text : d.label;
                    return `${value} (${(d.confidence * 100).toFixed(1)}%)`;
                })
                .join("<br>")
            : noResultsText;

        if (dynamicLabelEl) {
            if (VISION_MODE === "ocr") {
                dynamicLabelEl.textContent = data.result.searched_word || "-";
            } else {
                dynamicLabelEl.textContent = "-";
            }
        }

        const timeValue = data.result.cycle_time_ms || data.result.processing_time_ms || 0;
        document.getElementById("time").textContent = timeValue + " ms";

        document.getElementById("okCount").textContent = data.counters.ok;
        document.getElementById("nokCount").textContent = data.counters.nok;
        document.getElementById("totalCount").textContent = data.counters.total;

    } catch (e) {
        console.error(e);
    }
}

async function loadStatus() {
    const res = await fetch("/status");
    const data = await res.json();
    VISION_MODE = data.vision_mode;
    CURRENT_MODE = data.maintenance_mode ? "maintenance" : "production";
    LAST_SYNCED_MODE = CURRENT_MODE;
    updateVisionModeButtons();
}

function getPollingIntervalMs() {
    return VISION_MODE === "ocr" ? 100 : 500;
}

function startResultPolling() {
    updateResult();
    setInterval(updateResult, getPollingIntervalMs());
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
    const prodBtn = document.getElementById("startProductionBtn");
    const maintBtn = document.getElementById("startMaintenanceBtn");
    const visionModeToggle = document.getElementById("visionModeToggle");
    const cycleTimeSection = document.getElementById("cycleTimeSection");
    const previousMode = LAST_APPLIED_MODE;

    if (CURRENT_MODE === "maintenance") {
        banner.textContent = "MAINTENANCE MODE";
        banner.className = "mode-overlay maintenance";

        input.disabled = false;
        applyBtn.disabled = false;
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

    if (previousMode !== null && previousMode !== CURRENT_MODE) {
        syncModeToBackend();
    }
}

async function syncModeToBackend() {
    if (LAST_SYNCED_MODE === CURRENT_MODE) return;

    LAST_SYNCED_MODE = CURRENT_MODE;

    try {
        await fetch("/maintenance_mode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ maintenance_mode: CURRENT_MODE === "maintenance" })
        });
    } catch (e) {
        LAST_SYNCED_MODE = null;
        console.error("Failed to sync maintenance mode:", e);
    }
}

async function applyVisionMode(mode) {
    VISION_MODE = mode;
    updateVisionModeButtons();

    try {
        await fetch("/vision_mode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ vision_mode: mode })
        });
    } catch (e) {
        console.error("Failed to sync vision mode:", e);
    }
}

/* =========================
   BUTTON EVENTS
========================= */
document.getElementById("startProductionBtn").addEventListener("click", () => {
    if (!confirm("Start production mode?\n\nConfidence threshold will be locked.")) return;

    CURRENT_MODE = "production";
    applyMode();
    syncModeToBackend();
});

document.getElementById("startMaintenanceBtn").addEventListener("click", () => {
    // Hardcoded password prompt
    const userPass = prompt("Enter password to enter maintenance mode:", PASSWORD);
    if (userPass !== PASSWORD) {
        alert("Incorrect password. Access denied.");
        return;
    }

    CURRENT_MODE = "maintenance";
    applyMode();
    syncModeToBackend();
});

document.getElementById("modeOcrBtn").addEventListener("click", () => {
    if (CURRENT_MODE !== "maintenance") return;
    applyVisionMode("ocr");
});

document.getElementById("modeObjBtn").addEventListener("click", () => {
    if (CURRENT_MODE !== "maintenance") return;
    applyVisionMode("object_detection");
});

document.getElementById("resetBtn").addEventListener("click", async () => {
    if (!confirm("Are you sure you want to reset the counters?")) return;
    await fetch("/reset_counters", { method: "POST" });
});

/* =========================
   INIT
========================= */
async function init() {
    try {
        // Resolve mode first so all downstream behavior is mode-aware.
        await loadStatus();
        await loadThreshold();
        applyMode();
        startResultPolling();
    } catch (e) {
        console.error("Initialization failed:", e);
    }
}

init();
