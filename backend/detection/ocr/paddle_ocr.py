"""
PaddleOCR Integration

Provides the PaddleOCR class for text recognition using the PaddleOCR engine.
"""

import cv2
import numpy as np
import re
import time
import logging
import os
from backend.detection.ocr.paddle_worker import PaddleOCRWorker
from backend.core.config_loader import cfg
from backend.utils.logger import get_logger
from backend.utils.roi import apply_roi, draw_roi
from backend.detection.ocr.text_locator import locate_text_region, deskew_crop

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

        # Dynamic ROI (OCRead mode only): locates the printed text block via
        # classical CV instead of relying on the static cfg["roi"] box, since
        # the text doesn't land in the same place on every bottle. See
        # text_locator.py. Falls back to the static ROI when disabled or
        # when nothing is found on a given frame.
        dynamic_roi_cfg = ocr_cfg.get("dynamic_roi", {}) if isinstance(ocr_cfg.get("dynamic_roi"), dict) else {}
        self.dynamic_roi_enabled = bool(dynamic_roi_cfg.get("enabled", False))
        self.dynamic_roi_cfg = dynamic_roi_cfg
        self.dynamic_roi_debug_dir = dynamic_roi_cfg.get("debug_dir") if dynamic_roi_cfg.get("debug_save") else None

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
        """Enhance the ROI crop for PaddleOCR's recognizer before inference.

        This is deliberately NOT the same treatment as tesseract_ocr.py's
        version. PaddleOCR's recognition model is a CNN trained on natural,
        anti-aliased grayscale/color crops - hard binarization (Otsu/
        adaptive threshold, which Tesseract benefits from) throws away the
        soft edge gradients that CNN was trained on and tends to hurt
        accuracy rather than help it. So this only ever does two things:
        upscale small crops, and gently lift local contrast. It never
        produces a binary mask.

        Upscaling matters a lot here specifically because of dynamic ROI:
        it now hands this function a tight crop of just the printed line(s)
        instead of the old large static box, so a two-line dot-matrix stamp
        can end up as little as ~25-30px tall per line - well below the
        ~40-60px recognition accuracy tends to need. mode == "off" skips
        all of this and is intended for debugging/comparison, not normal
        production use.
        """
        mode = self.preprocess_mode
        if mode == "off":
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        h, w = gray.shape[:2]
        short = min(h, w)
        if short < 200:
            # Same upscale heuristic as tesseract_ocr.py, for consistency -
            # small crops get a stronger boost, capping out around 2-3x
            # rather than scaling indefinitely.
            scale = max(2, int(160 / short)) if short < 80 else 2
            gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        enhanced = self._clahe.apply(gray)

        if mode == "accurate":
            enhanced = cv2.medianBlur(enhanced, 3)

        # Paddle's recognizer expects a 3-channel image - replicate the
        # enhanced grayscale back out rather than passing single-channel,
        # since the print here is essentially monochrome ink on plastic
        # anyway and this avoids any channel-count surprises in the model.
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

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

    def _apply_roi(self, frame, dynamic=False):
        """Crop the frame to the text region.

        dynamic=True is the OCRead-mode path, and it never touches
        cfg["roi"]. It only ever does one of two things: crop to wherever
        locate_text_region() actually found ink on this frame (deskewed via
        deskew_crop), or - if it found nothing at all - hand the full,
        uncropped frame to OCR rather than guessing a fixed rectangle. A
        production line where bottles don't land the same way every time
        means a static box is wrong as often as it's right, so there is no
        static-ROI fallback in this path at all. No debug overlay is drawn
        here either - the detected region changes on every frame, so a box
        drawn on the live view wouldn't mean anything to line staff.

        dynamic=False (the plain "ocr" keyword-match mode) is unrelated to
        OCRead and still uses the static cfg["roi"] box, via the shared roi
        utility so the coordinate math isn't duplicated across web.py /
        tesseract_ocr.py.
        """
        if dynamic:
            if not self.dynamic_roi_enabled:
                log.warning(
                    "OCRead called with dynamic ROI disabled (ocr.dynamic_roi.enabled=false) - "
                    "running OCR on the full frame. Enable it in config to locate text automatically."
                )
                return frame

            located = locate_text_region(frame, self.dynamic_roi_cfg, debug_dir=self.dynamic_roi_debug_dir)
            if not located:
                log.debug("Dynamic ROI: no text region found on this frame - running OCR on the full frame.")
                return frame

            rotated_rect = located.get("rotated_rect")
            # Extra margin here (beyond dynamic_roi.padding_px, which is
            # already baked into rotated_rect) is specifically for
            # PaddleOCR's own internal detector, which tends to need a bit
            # of non-text margin around a region to reliably fire - a crop
            # with characters right up against the edge is more likely to
            # be missed by its detector even when a human would read it
            # fine. This is separate from (and on top of) the tighter
            # padding used for our own CV clustering, which stays as-is
            # since loosening that risks pulling ribs back into detection.
            crop = deskew_crop(frame, rotated_rect, extra_padding_px=20) if rotated_rect else None
            if crop is None:
                # Degenerate rotated rect (shouldn't normally happen) - use
                # the axis-aligned box we already computed rather than the
                # full frame, since we did find *something*.
                x1, y1, x2, y2 = located["bbox"]
                crop = frame[y1:y2, x1:x2]

            return crop

        # Static-ROI path: only reached from run() (the "ocr" keyword mode).
        # read() (OCRead) always passes dynamic=True and never gets here.
        roi = cfg.get("roi")
        if not roi:
            return frame

        if self.debug_draw_roi:
            draw_roi(frame, roi)

        return apply_roi(frame, roi)

    def _recognize(self, frame, profile=False, dynamic_roi=False):
        """
        Shared pipeline: ROI, downscale, and the actual PaddleOCR inference call.

        Returns every text candidate PaddleOCR produced (unfiltered), along
        with timing info. Both `run()` (keyword match, for the OCR mode) and
        `read()` (raw read-out, for the OCRead mode) build on this so the
        downscale/preprocessing/inference path is identical between the two
        modes — only the ROI step differs, via dynamic_roi (OCRead only).
        """
        profile_data = {} if profile else None

        t0 = time.perf_counter()
        roi_frame = self._apply_roi(frame, dynamic=dynamic_roi)
        if profile_data is not None:
            profile_data["roi_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        t1 = time.perf_counter()
        roi_frame = self._downscale_roi(roi_frame)
        if profile_data is not None:
            profile_data["downscale_ms"] = round((time.perf_counter() - t1) * 1000, 3)

        t1b = time.perf_counter()
        roi_frame = self._preprocess_image(roi_frame)
        if profile_data is not None:
            profile_data["preprocess_ms"] = round((time.perf_counter() - t1b) * 1000, 3)

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

        Uses the dynamic (classical-CV) ROI locator when enabled, since
        this is the mode that needs to track the text wherever it actually
        printed on each bottle - see _apply_roi().
        """
        candidates, elapsed_ms, profile_data = self._recognize(frame, profile=profile, dynamic_roi=True)

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