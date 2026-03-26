import cv2
import re
import time
import logging
from paddleocr import PaddleOCR as _PaddleOCR
from config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)

logging.getLogger("ppocr").setLevel(logging.ERROR)

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
            "PaddleOCR init: preprocess_mode=%s downscale=%s keywords=%s",
            self.preprocess_mode,
            self.downscale,
            self.keywords,
        )

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
        # Apply ROI and downscale on color frame
        roi_frame = self._apply_roi(frame)
        roi_frame = self._downscale_roi(roi_frame)

        keywords = (
            [self.app_state.get_ocr_keyword()]
            if self.app_state else self.keywords
        )

        log.debug("PaddleOCR run: searching for keywords=%s", keywords)

        start_time = time.perf_counter()

        # Use legacy ocr() API (avoid predict() oneDNN compatibility issues)
        try:
            results = self._paddle.ocr(roi_frame, cls=True)
        except Exception as e:
            log.error("PaddleOCR.ocr() failed: %s", e, exc_info=True)
            results = None

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        detections = []

        if results:
            # Legacy ocr() returns a list of pages; each page is a list of [bbox, (text, score)]
            for page_idx, page_results in enumerate(results or []):
                if not page_results:
                    continue
                for item_idx, item in enumerate(page_results or []):
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    
                    bbox = item[0]
                    rec = item[1]
                    
                    if not isinstance(rec, (list, tuple)) or len(rec) < 2:
                        continue
                    
                    text = rec[0]
                    score = rec[1]

                    if not text.strip():
                        continue

                    log.info("PaddleOCR detected: text=%r confidence=%.3f", text, score)

                    word = text.lower()

                    # Filter by keywords/regex if defined (exact match like TesseractOCR)
                    if keywords and word not in keywords:
                        if self.date_regex and not re.search(self.date_regex, text):
                            log.debug("PaddleOCR filtered out: text=%r (no keyword/date match)", text)
                            continue

                    detections.append({
                        "text": text,
                        "confidence": round(float(score), 3)
                    })

        processing_time_ms = round(elapsed_ms, 1)
        searched_word = keywords[0] if keywords else ""
        log.debug(
            "PaddleOCR run completed: processing_time_ms=%s detections=%s searched_word=%r",
            processing_time_ms,
            len(detections),
            searched_word,
        )
        return {
            "detections": detections,
            "processing_time_ms": processing_time_ms,
            "mode": "ocr",
            "searched_word": searched_word
        }

