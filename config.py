# ========== ROI ==========
ROI = {
    "x_start": 0.0,
    "y_start": 0.6667,
    "x_end": 1.0,
    "y_end": 1.0
}

# ========== OCR KEYWORDS ==========
EXPECTED_KEYWORDS = [
    "www.theimagingsource.com",

]

DATE_REGEX = r"\b\d{2}/\d{2}/\d{4}\b"

DEBUG_DRAW_ROI = True

# ========== WEB FEED ==========
# Set to False to disable video streaming (for performance testing)
#ENABLE_VIDEO_FEED = False

# ========== OCR PERFORMANCE TUNING ==========
# Options: "clahe" (slower, clearer), "fast" (threshold), "off" (no preprocessing)
OCR_PREPROCESS = "fast"
# Downscale ROI before OCR (0.5-1.0). Use 1.0 to disable.
OCR_DOWNSCALE = 0.6
# Minimum ROI dimension (px) allowed after downscale.
OCR_MIN_DIM = 240
# Tesseract layout mode (6=block, 7=line, 11=sparse)
OCR_PSM = 7
# Tesseract OCR engine mode (1=legacy, 3=default)
OCR_OEM = 3
# Optional whitelist to speed up recognition (empty to disable)
OCR_WHITELIST = "abcdefghijklmnopqrstuvwxyz0123456789."
# Disable dictionary lookups for speed on fixed text patterns
OCR_DISABLE_DAWGS = True
