import cv2
import re
import time
import logging
import os
from paddleocr import PaddleOCR as _PaddleOCR
from backend.core.config_loader import cfg
from backend.utils.logger import get_logger

log = get_logger(__name__)

logging.getLogger("ppocr").setLevel(logging.ERROR)

# Disable oneDNN to avoid compatibility issues with PaddleOCR
os.environ['PADDLE_DISABLE_FAST_MATH'] = '1'
os.environ['FLAGS_use_mkldnn'] = '0'


class PaddleOCR:
    def __init__(self, app_state=None):
        self.app_state = app_state
        # Initialize PaddleOCR (oneDNN disabled via environment variables above)
        self._paddle = _PaddleOCR(lang="en")
        ocr_cfg = cfg["ocr"]
        self.keywords = [w.lower() for w in ocr_cfg["keywords"]]
        self.keyword_set = set(self.keywords)
        self.date_regex = ocr_cfg["date_regex"]
        self._date_pattern = re.compile(self.date_regex) if self.date_regex else None
        self.debug_draw_roi = cfg["hmi"]["debug_draw_roi"]
        self.preprocess_mode = ocr_cfg["preprocess"].lower()
        self.downscale = float(ocr_cfg["downscale"])
        self.min_dim = int(ocr_cfg["min_dim"])
        log.debug(
            "PaddleOCR init: preprocess_mode=%s downscale=%s keywords=%s",
            self.preprocess_mode,
            self.downscale,
            self.keywords,
        )

    def _preprocess_image(self, frame):
        """Enhance image contrast and clarity for better OCR"""
        mode = self.preprocess_mode
        if mode == "off":
            return frame

        # Convert to grayscale for preprocessing
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        if mode == "fast":
            # Otsu thresholding is faster than CLAHE
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Slight blur to reduce noise
        denoised = cv2.medianBlur(enhanced, 3)

        return denoised

    def _downscale_roi(self, frame):
        if self.downscale >= 1.0:
            return frame
        h, w = frame.shape[:2]
        if self.min_dim and min(h, w) <= self.min_dim:
            return frame
        scale = self.downscale
        if self.min_dim:
            scale = max(scale, self.min_dim / float(min(h, w)))
        if scale >= 1.0:
            return frame
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _apply_roi(self, frame):
        roi = cfg.get("roi")
        if not roi:
            return frame
        h, w = frame.shape[:2]

        def _to_px(value, max_dim):
            if value <= 1.0:
                return int(round(value * max_dim))
            return int(round(value))

        x1 = _to_px(float(roi.get("x_start", 0.0)), w)
        y1 = _to_px(float(roi.get("y_start", 0.0)), h)
        x2 = _to_px(float(roi.get("x_end", 1.0)), w)
        y2 = _to_px(float(roi.get("y_end", 1.0)), h)

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            return frame

        if self.debug_draw_roi:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cropped = frame[y1:y2, x1:x2]
        if cropped is None or cropped.size == 0:
            log.warning("ROI crop resulted in an empty or invalid frame")
            return frame
        return cropped

    def run(self, frame):
        # Apply ROI and downscale on frame
        roi_frame = self._apply_roi(frame)
        roi_frame = self._downscale_roi(roi_frame)

        debug_enabled = log.isEnabledFor(logging.DEBUG)
        if debug_enabled:
            log.debug("PaddleOCR input: shape=%s dtype=%s", roi_frame.shape, roi_frame.dtype)

        keywords = (
            [self.app_state.get_ocr_keyword()]
            if self.app_state else self.keywords
        )
        keyword_set = {k.lower() for k in keywords if isinstance(k, str)} if self.app_state else self.keyword_set

        if debug_enabled:
            log.debug("PaddleOCR run: searching for keywords=%s", keywords)

        start_time = time.perf_counter()

        # PaddleOCR expects 3-channel color images (BGR), not grayscale
        # Pass color frame directly without grayscale conversion
        try:
            results = self._paddle.ocr(roi_frame)
            if debug_enabled:
                log.debug("PaddleOCR result pages=%s", len(results) if results else 0)
        except Exception as e:
            log.error("PaddleOCR.ocr() failed: %s", e, exc_info=True)
            results = None

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        detections = []
        total_candidates = 0

        if results:
            # Legacy ocr() returns a list of pages; each page is a list of [bbox, (text, score)]
            for page_results in results:
                if not page_results:
                    continue
                for item in page_results:
                    total_candidates += 1
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue

                    rec = item[1]

                    if not isinstance(rec, (list, tuple)) or len(rec) < 2:
                        continue

                    text = rec[0]
                    score = rec[1]

                    if not text.strip():
                        continue

                    word = text.lower()

                    # Filter by keywords/regex if defined (exact match like TesseractOCR)
                    if keyword_set and word not in keyword_set:
                        if self._date_pattern and not self._date_pattern.search(text):
                            continue

                    detections.append({
                        "text": text,
                        "confidence": round(float(score), 3)
                    })

        processing_time_ms = round(elapsed_ms, 1)
        searched_word = keywords[0] if keywords else ""
        log.debug(
            "PaddleOCR run completed: processing_time_ms=%s candidates=%s detections=%s searched_word=%r",
            processing_time_ms,
            total_candidates,
            len(detections),
            searched_word,
        )
        return {
            "detections": detections,
            "processing_time_ms": processing_time_ms,
            "mode": "ocr",
            "searched_word": searched_word
        }

