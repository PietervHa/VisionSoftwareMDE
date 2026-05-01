let CURRENT_MODE = "maintenance";
let VISION_MODE = null;
let currentThreshold = null; // mirrors backend value
const PASSWORD = "@Welkom01"; // hardcoded for now

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

    } catch (e) {
        console.error(e);
    }
}

async function loadStatus() {
    const res = await fetch("/status");
    const data = await res.json();
    VISION_MODE = data.vision_mode;
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

    if (CURRENT_MODE === "maintenance") {
        banner.textContent = "MAINTENANCE MODE";
        banner.className = "mode-overlay maintenance";

        input.disabled = false;
        applyBtn.disabled = false;
        prodBtn.style.display = "inline-block";
        maintBtn.style.display = "none";

        if (currentThreshold !== null) input.value = currentThreshold;
    }

    if (CURRENT_MODE === "production") {
        banner.textContent = "PRODUCTION MODE";
        banner.className = "mode-overlay production";

        input.disabled = true;
        applyBtn.disabled = true;
        prodBtn.style.display = "none";
        maintBtn.style.display = "inline-block";

        if (currentThreshold !== null) input.value = currentThreshold;
    }

    LAST_APPLIED_MODE = CURRENT_MODE;

    if (ocrKeywordSection) {
        ocrKeywordSection.style.display = (CURRENT_MODE === "maintenance" && VISION_MODE === "ocr")
            ? "flex"
            : "none";
    }

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

    const ocrKeywordSection = document.getElementById("ocrKeywordSection");
    if (ocrKeywordSection) {
        ocrKeywordSection.style.display = (CURRENT_MODE === "maintenance" && VISION_MODE === "ocr")
            ? "flex"
            : "none";
    }

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

    CURRENT_MODE = "production";
    applyMode();
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
