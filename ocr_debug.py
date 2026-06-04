"""
OCR preprocessing diagnostic
------------------------------
Run this from the project root:
    python ocr_debug.py

It captures ONE frame from the camera, runs every stage of the
preprocessing pipeline, saves each stage as a PNG so you can see
exactly what Tesseract receives, and prints the raw Tesseract token
output BEFORE any keyword filtering.

Output files end up in  debug_ocr/  next to this script.
"""

import os
import sys
import time

import cv2
import numpy as np
import pytesseract

# ── make sure project root is on the path ──────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from backend.core.config_loader import cfg
from backend.utils.roi import apply_roi

OUT_DIR = "debug_ocr"
os.makedirs(OUT_DIR, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = cfg["ocr"]["tesseract_path"]

# ── 1. grab a frame ────────────────────────────────────────────────
cam_index = cfg.get("camera", {}).get("index", 0)
cap = cv2.VideoCapture(cam_index)
time.sleep(0.5)                       # let auto-exposure settle
ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print("ERROR: could not read from camera")
    sys.exit(1)

cv2.imwrite(f"{OUT_DIR}/00_raw_frame.png", frame)
print(f"[00] raw frame  {frame.shape}")

# ── 2. apply ROI ───────────────────────────────────────────────────
roi = cfg.get("roi")
if roi:
    cropped = apply_roi(frame, roi)
else:
    cropped = frame
cv2.imwrite(f"{OUT_DIR}/01_after_roi.png", cropped)
print(f"[01] after ROI  {cropped.shape}")

# ── 3. channel breakdown ───────────────────────────────────────────
b, g, r = cv2.split(cropped)
cv2.imwrite(f"{OUT_DIR}/02_channel_B.png", b)
cv2.imwrite(f"{OUT_DIR}/02_channel_G.png", g)
cv2.imwrite(f"{OUT_DIR}/02_channel_R.png", r)
stds = {"B": b.std(), "G": g.std(), "R": r.std()}
print(f"[02] channel std  B={stds['B']:.1f}  G={stds['G']:.1f}  R={stds['R']:.1f}")
best_ch = max(stds, key=stds.get)
print(f"     → best contrast channel: {best_ch}")
gray_best = {"B": b, "G": g, "R": r}[best_ch]

# also save the standard luminance grayscale for comparison
gray_luma = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
cv2.imwrite(f"{OUT_DIR}/03_gray_luminance.png", gray_luma)
cv2.imwrite(f"{OUT_DIR}/03_gray_best_channel.png", gray_best)

# ── 4. upscaling ──────────────────────────────────────────────────
h, w = gray_best.shape[:2]
short = min(h, w)
if short < 80:
    scale = max(2, int(160 / short))
elif short < 200:
    scale = 2
else:
    scale = 1

if scale > 1:
    gray_up = cv2.resize(gray_best, (w * scale, h * scale),
                         interpolation=cv2.INTER_CUBIC)
    print(f"[04] upscaled ×{scale}  {gray_best.shape} → {gray_up.shape}")
else:
    gray_up = gray_best
    print(f"[04] no upscaling  (short dim = {short} px)")
cv2.imwrite(f"{OUT_DIR}/04_after_upscale.png", gray_up)

# ── 5. adaptive threshold ─────────────────────────────────────────
blurred = cv2.GaussianBlur(gray_up, (3, 3), 0)
binary = cv2.adaptiveThreshold(
    blurred, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=31,
    C=10,
)
nz = cv2.countNonZero(binary)
total = binary.size
ratio = nz / total
print(f"[05] adaptive threshold  white pixels = {ratio:.1%}")
if ratio < 0.3:
    print("     → image is mostly dark, inverting")
    binary = cv2.bitwise_not(binary)
cv2.imwrite(f"{OUT_DIR}/05_adaptive_threshold.png", binary)

# also save Otsu for comparison
_, otsu = cv2.threshold(gray_up, 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite(f"{OUT_DIR}/05_otsu_threshold.png", otsu)

# ── 6. raw Tesseract output (no keyword filter) ────────────────────
ocr_cfg = cfg["ocr"]
tess_cfg = f"--psm {ocr_cfg['psm']} --oem {ocr_cfg['oem']}"
if ocr_cfg.get("whitelist"):
    tess_cfg += f" -c tessedit_char_whitelist={ocr_cfg['whitelist']}"
if ocr_cfg.get("disable_dawgs"):
    tess_cfg += " -c load_system_dawg=0 -c load_freq_dawg=0"

print(f"\n[06] Tesseract config: {tess_cfg}")
print("     Raw token output (ALL tokens, no filtering):")
print(f"     {'text':20s}  {'conf':>6}")
print(f"     {'-'*20}  {'-'*6}")

data = pytesseract.image_to_data(
    binary, lang="eng", config=tess_cfg,
    output_type=pytesseract.Output.DICT,
)
any_text = False
for text, conf in zip(data["text"], data["conf"]):
    if text.strip():
        any_text = True
        print(f"     {text:20s}  {conf:>6}")
if not any_text:
    print("     (no tokens returned — Tesseract read nothing)")

# also run on luminance+Otsu so we can compare
data2 = pytesseract.image_to_data(
    otsu, lang="eng", config=tess_cfg,
    output_type=pytesseract.Output.DICT,
)
print(f"\n[07] Comparison — luminance grayscale + Otsu:")
print(f"     {'text':20s}  {'conf':>6}")
print(f"     {'-'*20}  {'-'*6}")
any_text2 = False
for text, conf in zip(data2["text"], data2["conf"]):
    if text.strip():
        any_text2 = True
        print(f"     {text:20s}  {conf:>6}")
if not any_text2:
    print("     (no tokens returned)")

print(f"\nAll debug images saved to  ./{OUT_DIR}/")
print("Open them to see what each preprocessing stage produces.")