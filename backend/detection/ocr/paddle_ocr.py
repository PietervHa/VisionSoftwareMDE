"""
PaddleOCR Integration

Provides the PaddleOCR class for text recognition using the PaddleOCR engine.
"""

import cv2
import re
import time
import logging
import os
from backend.detection.ocr.paddle_worker import PaddleOCRWorker
from backend.core.config_loader import cfg
from backend.utils.logger import get_logger
from backend.utils.roi import apply_roi, draw_roi

log = get_logger(__name__)

logging.getLogger("ppocr").setLevel(logging.ERROR)


class PaddleOCR:
    """
    OCR engine based on PaddleOCR for identifying keywords and dates in frames.
    """
    def __init__(self, app_state=None):
        self.app_state = app_state
        ocr_cfg = cfg["ocr"]
        self._paddle = PaddleOCRWorker(
            lang="en",
            cpu_threads=int(ocr_cfg.get("cpu_threads", 4)),
            use_angle_cls=bool(ocr_cfg.get("use_angle_cls", True)),
        )
        self.keywords = [w.lower() for w in ocr_cfg["keywords"]]
        self.keyword_set = set(self.keywords)
        self.date_regex = ocr_cfg["date_regex"]
        self._date_pattern = re.compile(self.date_regex) if self.date_regex else None
        self.debug_draw_roi = cfg["hmi"]["debug_draw_roi"]
        self.preprocess_mode = ocr_cfg["preprocess"].lower()
        self.downscale = float(ocr_cfg["downscale"])
        self.min_dim = int(ocr_cfg["min_dim"])

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        log.debug(
            "PaddleOCR init: preprocess_mode=%s downscale=%s keywords=%s",
            self.preprocess_mode,
            self.downscale,
            self.keywords,
        )

    def warmup(self) -> None:
        """Runs one throwaway OCR pass so Paddle's native-runtime cold-start
        cost is paid at startup instead of during the first real OCR cycle."""
        try:
            import numpy as np
            self._paddle.ocr(np.zeros((64, 64, 3), dtype="uint8"))
            log.debug("PaddleOCR warm-up inference completed.")
        except Exception as exc:
            log.warning("PaddleOCR warm-up inference failed (non-fatal): %s", exc)

    def _preprocess_image(self, frame):
        """Enhance image contrast and clarity for better OCR"""
        mode = self.preprocess_mode
        if mode == "off":
            return frame

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        if mode == "fast":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary

        enhanced = self._clahe.apply(gray)
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
        """Crop the frame to the configured ROI, optionally drawing the debug overlay.

        delegates to shared roi utility instead of duplicating the
        coordinate math that also lives in web.py and tesseract_ocr.py.
        """
        roi = cfg.get("roi")
        if not roi:
            return frame

        if self.debug_draw_roi:
            draw_roi(frame, roi)

        return apply_roi(frame, roi)

    def _recognize(self, frame, profile=False):
        """
        Shared pipeline: ROI, downscale, and the actual PaddleOCR inference call.

        Returns every text candidate PaddleOCR produced (unfiltered), along
        with timing info. Both `run()` (keyword match, for the OCR mode) and
        `read()` (raw read-out, for the OCRead mode) build on this so the
        ROI/preprocessing/inference path is identical between the two modes.
        """
        profile_data = {} if profile else None

        t0 = time.perf_counter()
        roi_frame = self._apply_roi(frame)
        if profile_data is not None:
            profile_data["roi_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        t1 = time.perf_counter()
        roi_frame = self._downscale_roi(roi_frame)
        if profile_data is not None:
            profile_data["downscale_ms"] = round((time.perf_counter() - t1) * 1000, 3)

        debug_enabled = log.isEnabledFor(logging.DEBUG)
        if debug_enabled:
            log.debug("PaddleOCR input: shape=%s dtype=%s", roi_frame.shape, roi_frame.dtype)

        start_time = time.perf_counter()

        t2 = start_time
        try:
            results = self._paddle.ocr(roi_frame)
            if debug_enabled:
                log.debug("PaddleOCR result pages=%s", len(results) if results else 0)
        except Exception as e:
            log.error("PaddleOCR.ocr() failed: %s", e, exc_info=True)
            results = None

        if profile_data is not None:
            profile_data["ocr_ms"] = round((time.perf_counter() - t2) * 1000, 3)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        candidates = []
        if results:
            for page_results in results:
                if not page_results:
                    continue
                for item in page_results:
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    rec = item[1]
                    if not isinstance(rec, (list, tuple)) or len(rec) < 2:
                        continue
                    text = rec[0]
                    score = rec[1]
                    if not text.strip():
                        continue
                    candidates.append({"text": text, "confidence": round(float(score), 3)})

        return candidates, elapsed_ms, profile_data

    def run(self, frame, profile=False):
        """
        Executes OCR on the provided frame, applying ROI and preprocessing if configured.

        Used by the OCR mode: recognized text is matched against the
        configured/active keyword (and date pattern) here, and only matching
        candidates are returned as detections.
        """
        candidates, elapsed_ms, profile_data = self._recognize(frame, profile=profile)

        keywords = (
            [self.app_state.get_ocr_keyword()]
            if self.app_state else self.keywords
        )
        keyword_set = {k.lower() for k in keywords if isinstance(k, str)} if self.app_state else self.keyword_set

        t3 = time.perf_counter()
        detections = []
        for candidate in candidates:
            text = candidate["text"]
            word = text.lower()
            if keyword_set and word not in keyword_set:
                if self._date_pattern and not self._date_pattern.search(text):
                    continue
            detections.append(candidate)

        if profile_data is not None:
            profile_data["filter_ms"] = round((time.perf_counter() - t3) * 1000, 3)
            profile_data["total_ms"] = round(sum(profile_data.values()), 3)

        processing_time_ms = round(elapsed_ms, 1)
        searched_word = keywords[0] if keywords else ""
        log.debug(
            "PaddleOCR run completed: processing_time_ms=%s candidates=%s detections=%s searched_word=%r",
            processing_time_ms,
            len(candidates),
            len(detections),
            searched_word,
        )
        result = {
            "detections": detections,
            "processing_time_ms": processing_time_ms,
            "mode": "ocr",
            "searched_word": searched_word
        }

        if profile_data is not None:
            result["_profile_ms"] = profile_data

        return result

    def read(self, frame, profile=False):
        """
        Executes OCR on the provided frame and returns every recognized piece
        of text as-is, with no keyword/date matching.

        Used by the OCRead mode: the comparison against the expected text is
        made on the PLC, not here, so nothing is filtered out or judged
        OK/NOK - the raw read-out is simply reported back.
        """
        candidates, elapsed_ms, profile_data = self._recognize(frame, profile=profile)

        processing_time_ms = round(elapsed_ms, 1)
        full_text = " ".join(c["text"] for c in candidates).strip()
        log.debug(
            "PaddleOCR read completed: processing_time_ms=%s detections=%s text=%r",
            processing_time_ms,
            len(candidates),
            full_text,
        )
        result = {
            "detections": candidates,
            "text": full_text,
            "processing_time_ms": processing_time_ms,
            "mode": "ocread",
        }

        if profile_data is not None:
            result["_profile_ms"] = profile_data

        return result
