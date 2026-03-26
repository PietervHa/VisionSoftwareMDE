import cv2
import re
import time
from paddleocr import PaddleOCR as _PaddleOCR
from config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)

class PaddleOCR:
    def __init__(self, app_state=None):
        self.app_state = app_state
        self._paddle = _PaddleOCR(lang="en")
        ocr_cfg = cfg["ocr"]
        self.keywords = [w.lower() for w in ocr_cfg["keywords"]]
        self.date_regex = ocr_cfg["date_regex"]
        self.debug_draw_roi = cfg["hmi"]["debug_draw_roi"]
        self.preprocess_mode = ocr_cfg["preprocess"].lower()
        self.downscale = float(ocr_cfg["downscale"])
        self.min_dim = int(ocr_cfg["min_dim"])
        log.debug(
            "PaddleOCR init: preprocess_mode=%s downscale=%s",
            self.preprocess_mode,
            self.downscale,
        )

    def _preprocess_image(self, gray):
        """Enhance image contrast and clarity for faster OCR"""
        mode = self.preprocess_mode
        if mode == "off":
            return gray
        if mode == "fast":
            # Otsu thresholding is faster than CLAHE + blur.
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Slight blur to reduce noise
        denoised = cv2.medianBlur(enhanced, 3)

        return denoised

    def _downscale_roi(self, gray):
        if self.downscale >= 1.0:
            return gray

        h, w = gray.shape[:2]
        if self.min_dim and min(h, w) <= self.min_dim:
            return gray

        scale = self.downscale
        if self.min_dim:
            scale = max(scale, self.min_dim / float(min(h, w)))
        if scale >= 1.0:
            return gray

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

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
        return cropped

    def run(self, frame):
        roi_frame = self._apply_roi(frame)
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        keywords = (
            [self.app_state.get_ocr_keyword()]
            if self.app_state else self.keywords
        )

        gray = self._downscale_roi(gray)

        # Preprocess for faster/better OCR
        preprocessed = self._preprocess_image(gray)

        start_time = time.perf_counter()

        results = self._paddle.ocr(preprocessed, cls=True)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        detections = []

        for line in (results[0] or []):
            bbox, (text, score) = line
            if not text.strip():
                continue

            word = text.lower()

            # Filter by keywords/regex if defined
            if keywords and word not in keywords:
                if self.date_regex and not re.search(self.date_regex, text):
                    continue

            detections.append({
                "text": text,
                "confidence": round(float(score), 3)
            })

        # Return detections and processing time
        processing_time_ms = round(elapsed_ms, 1)
        log.debug("PaddleOCR run completed: processing_time_ms=%s", processing_time_ms)
        return {
            "detections": detections,
            "processing_time_ms": processing_time_ms,
            "mode": "ocr",
            "searched_word": keywords[0] if keywords else ""
        }

