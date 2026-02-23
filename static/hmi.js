let CURRENT_MODE = "maintenance";
let currentThreshold = null; // mirrors backend value
const PASSWORD = "@Welkom01"; // hardcoded for now

/* =========================
   RESULT POLLING
========================= */
async function updateResult() {
    try {
        const res = await fetch("/result");
        const data = await res.json();

        const statusEl = document.getElementById("status");
        const detEl = document.getElementById("detections");

        if (data.result.status === "OK") {
            statusEl.textContent = "OK";
            statusEl.className = "status ok";
        } else {
            statusEl.textContent = "NOK";
            statusEl.className = "status nok";
        }

        detEl.innerHTML = data.result.detections.length
            ? data.result.detections
                .map(d => `${d.label} (${(d.confidence * 100).toFixed(1)}%)`)
                .join("<br>")
            : "No objects detected";

        document.getElementById("time").textContent =
            data.result.processing_time_ms + " ms";

        document.getElementById("okCount").textContent = data.counters.ok;
        document.getElementById("nokCount").textContent = data.counters.nok;
        document.getElementById("totalCount").textContent = data.counters.total;

    } catch (e) {
        console.error(e);
    }
}

setInterval(updateResult, 500);

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
loadThreshold().then(applyMode);