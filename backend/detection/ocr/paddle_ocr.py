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
            enable_mkldnn=bool(ocr_cfg.get("enable_mkldnn", True)),
            text_detection_model_name=ocr_cfg.get("text_detection_model_name") or None,
            text_recognition_model_name=ocr_cfg.get("text_recognition_model_name") or None,
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

        # Rib inpainting (OCRead / dynamic ROI only): on this cap, the
        # mold rib crosses directly through the same 1-2 leading
        # characters of the printed line on almost every cycle, because
        # the rib position and the print position never move relative to
        # each other - confirmed against real debug captures, not a
        # hypothetical. text_locator.py already detects+erases rib lines,
        # but only inside the *clustering* mask used to locate the text -
        # the rib pixels are still physically present in the crop that
        # actually gets handed to the recognizer. This re-detects long
        # straight lines directly on the deskewed crop and inpaints over
        # them before recognition, instead of just working around the rib
        # during localization.
        #
        # Off by default - validate against dynamic_roi.debug_save
        # captures on your own frames before trusting this in production.
        # See _derib_crop() for why the thresholds need checking against
        # real crops rather than assumed safe.
        self.derib_enabled = bool(dynamic_roi_cfg.get("derib_enabled", False))
        self.derib_blackhat_kernel = int(dynamic_roi_cfg.get("derib_blackhat_kernel", 25))
        self.derib_threshold = int(dynamic_roi_cfg.get("derib_threshold", 32))
        self.derib_min_length_frac = float(dynamic_roi_cfg.get("derib_min_length_frac", 0.6))
        self.derib_thickness = int(dynamic_roi_cfg.get("derib_thickness", 7))

        # Orientation retry (OCRead / dynamic ROI only): deskew_crop()
        # resolves *which way* a rotated line is tilted, but not whether
        # it came out upside-down (0 vs 180 degrees) - that ambiguity is
        # left to PaddleOCR's own angle classifier (use_angle_cls). On
        # this print - tiny, sparse dot-matrix digits - that classifier
        # isn't reliable: debug captures show visibly worse reads on
        # cycles where the cap was genuinely upside-down in frame. Rather
        # than trust it blindly every cycle, a low-confidence (or empty)
        # first pass is retried against a 180-degree-rotated copy of the
        # same crop, keeping whichever pass scores higher. Only fires when
        # the first pass looks doubtful, so the common case (upright,
        # confident read) pays no extra latency.
        self.orientation_retry_enabled = bool(dynamic_roi_cfg.get("orientation_retry_enabled", True))
        self.orientation_retry_confidence_threshold = float(
            dynamic_roi_cfg.get("orientation_retry_confidence_threshold", 0.85)
        )
        # Whether low confidence alone (not just an empty result) should
        # trigger the retry - see the long comment above
        # orientation_retry_confidence_threshold in default.yaml. Defaults
        # False now: three threshold values (0.85, 0.6, 0.8) were each
        # tried against real capture data, and none of them cleanly
        # separated genuine upside-down cycles from a persistent,
        # orientation-independent recognition ambiguity on this cap's
        # leading digit - worse, when the retry fired on the latter, it
        # picked the wrong (flipped) answer 13/13 times in one 30-cycle
        # capture. Retrying on a fully empty result is still safe and
        # still on by default; retrying on "found something, just not
        # very confident" is not, until that ambiguity is actually
        # understood rather than threshold-tuned around.
        self.orientation_retry_require_low_confidence = bool(
            dynamic_roi_cfg.get("orientation_retry_require_low_confidence", False)
        )

        # Any cycle slower than this gets its full stage breakdown
        # (roi_ms/derib_ms/ocr_ms/orientation_retry_ms/total_ocr_call_ms)
        # logged as a WARNING, regardless of whether the caller passed
        # profile=True - see the auto-log block in _recognize(). This is
        # what previous multi-second spikes were missing: elapsed_ms alone
        # doesn't say whether the time went into Paddle's own compute or
        # somewhere it shouldn't have.
        self.slow_cycle_log_threshold_ms = float(ocr_cfg.get("slow_cycle_log_threshold_ms", 800))

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        log.debug(
            "PaddleOCR init: preprocess_mode=%s downscale=%s keywords=%s derib_enabled=%s "
            "orientation_retry_enabled=%s",
            self.preprocess_mode,
            self.downscale,
            self.keywords,
            self.derib_enabled,
            self.orientation_retry_enabled,
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

    def _derib_crop(self, crop):
        """
        Detects and inpaints long straight rib lines crossing the deskewed
        OCRead crop, before the recognizer ever sees it.

        Same detection technique as text_locator._erase_long_lines()
        (blackhat -> threshold -> HoughLinesP), but run directly on the
        crop and used to inpaint real pixels rather than blank a
        clustering mask - see the comment above self.derib_enabled in
        __init__ for why this exists.

        min_length_frac is relative to the crop's own largest dimension
        rather than an absolute pixel count, since dynamic ROI crops vary
        in size cycle to cycle (unlike text_locator's search crop, which
        is close to full-frame-sized every time). Keep this conservative:
        a crop is fit tightly around the text, so the *real* printed line
        can itself span a large fraction of the crop's width, and in
        principle a row of characters bridged by HoughLinesP's maxLineGap
        could be mistaken for one long line. In practice individual
        digit/letter shapes rarely present one straight pixel-level edge
        the way a rib does, but verify against dynamic_roi.debug_save
        captures (compare the pre/post image) before trusting this on the
        line, and tighten min_length_frac or thickness if it ever bites
        into real characters instead of the rib.

        Best-effort: any failure or "nothing found" returns the crop
        unchanged, since a missed rib is no worse than today's behaviour.
        """
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            h, w = gray.shape[:2]

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (max(1, self.derib_blackhat_kernel),) * 2
            )
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            _, mask = cv2.threshold(blackhat, self.derib_threshold, 255, cv2.THRESH_BINARY)

            min_length = int(max(w, h) * self.derib_min_length_frac)
            lines = cv2.HoughLinesP(
                mask, 1, np.pi / 180,
                threshold=40, minLineLength=min_length, maxLineGap=10,
            )
            if lines is None:
                return crop

            inpaint_mask = np.zeros((h, w), dtype=np.uint8)
            found = False
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if length < min_length:
                    continue
                cv2.line(inpaint_mask, (x1, y1), (x2, y2), 255, thickness=self.derib_thickness)
                found = True

            if not found:
                return crop

            return cv2.inpaint(crop, inpaint_mask, 3, cv2.INPAINT_TELEA)
        except Exception as exc:
            log.warning("Rib inpainting failed (non-fatal), crop used as-is: %s", exc)
            return crop

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

        Returns (crop, cycle_debug_dir). cycle_debug_dir is the same
        per-cycle folder text_locator.py just wrote its own debug images
        into (None if dynamic_roi debug_save is off, or in the static-ROI
        path, which has no equivalent) - _recognize() uses it to write the
        actual OCR result alongside the localization debug images, so a
        debug capture shows both where the crop came from and what OCR did
        with it, instead of just the former.
        """
        if dynamic:
            if not self.dynamic_roi_enabled:
                log.warning(
                    "OCRead called with dynamic ROI disabled (ocr.dynamic_roi.enabled=false) - "
                    "running OCR on the full frame. Enable it in config to locate text automatically."
                )
                return frame, None

            located = locate_text_region(frame, self.dynamic_roi_cfg, debug_dir=self.dynamic_roi_debug_dir)
            if not located:
                log.debug("Dynamic ROI: no text region found on this frame - running OCR on the full frame.")
                return frame, None

            cycle_debug_dir = located.get("cycle_debug_dir")
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

            return crop, cycle_debug_dir

        # Static-ROI path: only reached from run() (the "ocr" keyword mode).
        # read() (OCRead) always passes dynamic=True and never gets here.
        roi = cfg.get("roi")
        if not roi:
            return frame, None

        if self.debug_draw_roi:
            draw_roi(frame, roi)

        return apply_roi(frame, roi), None

    def _write_ocr_result_debug(self, debug_dir, display_frame, candidates, elapsed_ms, extra_lines=None):
        """Write the actual OCR outcome (not just the crop) into the same
        per-cycle debug folder text_locator.py used, so a debug capture
        shows what was read, not only where the crop came from - reviewing
        localization and recognition together instead of needing a second
        round-trip for log lines.

        extra_lines: optional list of extra "key=value" strings appended
        to the report header - currently used to record whether the
        orientation retry (see _recognize()) ended up using the flipped
        crop, so a debug capture also shows *why* a read looks the way it
        does, not just what it read.
        """
        try:
            full_text = " ".join(c["text"] for c in candidates).strip()
            header_lines = [
                f"full_text={full_text!r}",
                f"elapsed_ms={round(elapsed_ms, 1)}",
                f"candidate_count={len(candidates)}",
            ]
            if extra_lines:
                header_lines.extend(extra_lines)
            body_lines = [f"text={c['text']!r} confidence={c['confidence']}" for c in candidates]
            report = "\n".join(header_lines + body_lines) + "\n"

            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, "ocr_result.txt"), "w", encoding="utf-8") as f:
                f.write(report)

            # Captioned image too, for a quick visual glance without having
            # to open the text file - shows exactly what PaddleOCR received
            # (post-preprocessing) plus what it read from it.
            img = display_frame
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            h, w = img.shape[:2]
            caption_h = 30
            canvas = np.full((h + caption_h, max(w, 320), 3), 255, dtype=np.uint8)
            canvas[:h, :w] = img
            caption = full_text if full_text else "(no text detected)"
            cv2.putText(canvas, caption, (5, h + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(debug_dir, "ocr_result.png"), canvas)
        except Exception as exc:  # pragma: no cover - debug aid only, never fatal
            log.warning("OCR result debug write failed (non-fatal): %s", exc)

    @staticmethod
    def _score_candidates(candidates):
        """Aggregate confidence score used to compare two candidate sets
        (a crop vs its 180-degree-rotated twin, see _recognize()'s
        orientation retry). Sum rather than mean - deliberately rewards an
        orientation that recovers *more* legible text, not just whichever
        set happens to contain one single high-confidence token."""
        return sum(c["confidence"] for c in candidates)

    def _recognize(self, frame, profile=False, dynamic_roi=False):
        """
        Shared pipeline: ROI, downscale, and the actual PaddleOCR inference call.

        Returns every text candidate PaddleOCR produced (unfiltered), along
        with timing info. Both `run()` (keyword match, for the OCR mode) and
        `read()` (raw read-out, for the OCRead mode) build on this so the
        downscale/preprocessing/inference path is identical between the two
        modes — only the ROI step differs, via dynamic_roi (OCRead only).

        For dynamic_roi (OCRead) specifically, two extra passes can run,
        each targeting a documented failure mode rather than general
        robustness padding - see _derib_crop() and the orientation-retry
        block below for the evidence behind each:
          - rib inpainting on the crop itself (derib_enabled)
          - a low-confidence retry against a 180-degree-rotated copy of
            the crop (orientation_retry_enabled), since deskew_crop() only
            resolves line tilt, not whether the line came out upside-down.
        """
        # Collected unconditionally now (a handful of perf_counter() calls
        # costs nothing measurable) rather than only when profile=True, so
        # the slow-cycle logging below always has a real stage breakdown to
        # show - see self.slow_cycle_log_threshold_ms in __init__. profile=
        # still controls whether this comes back in the returned result
        # dict for benchmarks/ocr_benchmark.py.
        profile_data = {}

        t0 = time.perf_counter()
        roi_frame, cycle_debug_dir = self._apply_roi(frame, dynamic=dynamic_roi)
        profile_data["roi_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        if dynamic_roi and self.derib_enabled:
            t_derib = time.perf_counter()
            roi_frame = self._derib_crop(roi_frame)
            profile_data["derib_ms"] = round((time.perf_counter() - t_derib) * 1000, 3)

        t1 = time.perf_counter()
        roi_frame = self._downscale_roi(roi_frame)
        profile_data["downscale_ms"] = round((time.perf_counter() - t1) * 1000, 3)

        t1b = time.perf_counter()
        roi_frame = self._preprocess_image(roi_frame)
        profile_data["preprocess_ms"] = round((time.perf_counter() - t1b) * 1000, 3)

        debug_enabled = log.isEnabledFor(logging.DEBUG)
        if debug_enabled:
            log.debug("PaddleOCR input: shape=%s dtype=%s", roi_frame.shape, roi_frame.dtype)

        start_time = time.perf_counter()

        t2 = start_time
        try:
            # paddle_worker.py parses PaddleOCR's own result object inside
            # the worker process now (see its module docstring) and hands
            # back a plain list of {"text", "confidence"} dicts - nothing
            # further to unpack here regardless of which PaddleOCR version
            # produced it.
            candidates = self._paddle.ocr(roi_frame) or []
            if debug_enabled:
                log.debug("PaddleOCR result count=%s", len(candidates))
        except Exception as e:
            log.error("PaddleOCR.ocr() failed: %s", e, exc_info=True)
            candidates = []

        profile_data["ocr_ms"] = round((time.perf_counter() - t2) * 1000, 3)

        # Orientation retry - see the docstring above and the comment
        # above self.orientation_retry_enabled in __init__. Only pays for
        # a second inference pass when the first one looks doubtful
        # (nothing found, or any candidate below threshold); a clean
        # upright read never takes this branch.
        used_flipped = False
        if dynamic_roi and self.orientation_retry_enabled:
            should_retry = not candidates
            if self.orientation_retry_require_low_confidence and candidates:
                worst_conf = min(c["confidence"] for c in candidates)
                should_retry = should_retry or worst_conf < self.orientation_retry_confidence_threshold
            if should_retry:
                t_retry = time.perf_counter()
                flipped_candidates = []
                flipped_frame = None
                try:
                    flipped_frame = cv2.rotate(roi_frame, cv2.ROTATE_180)
                    flipped_candidates = self._paddle.ocr(flipped_frame) or []
                except Exception as exc:
                    log.warning("Orientation retry failed (non-fatal): %s", exc)
                profile_data["orientation_retry_ms"] = round((time.perf_counter() - t_retry) * 1000, 3)

                if flipped_frame is not None and self._score_candidates(flipped_candidates) > self._score_candidates(candidates):
                    candidates = flipped_candidates
                    roi_frame = flipped_frame  # keep debug capture consistent with what was actually read
                    used_flipped = True
                    log.debug("Orientation retry: flipped crop scored higher, using it.")

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        profile_data["total_ocr_call_ms"] = round(elapsed_ms, 3)

        # Auto-log the stage breakdown for any cycle that crossed the slow
        # threshold, independent of whether the caller passed profile=True -
        # this is what a spike capture was missing so far: elapsed_ms alone
        # doesn't say whether the time went into Paddle's own compute
        # (ocr_ms) or somewhere it shouldn't have (a large gap between
        # total_ocr_call_ms and ocr_ms+orientation_retry_ms points at
        # queueing/IPC/system contention rather than model compute).
        if elapsed_ms > self.slow_cycle_log_threshold_ms:
            log.warning(
                "PaddleOCR slow cycle: elapsed_ms=%.1f (threshold=%.0f) stage_breakdown=%s",
                elapsed_ms, self.slow_cycle_log_threshold_ms, profile_data,
            )

        if cycle_debug_dir:
            self._write_ocr_result_debug(
                cycle_debug_dir, roi_frame, candidates, elapsed_ms,
                extra_lines=[f"used_flipped_orientation={used_flipped}"],
            )

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

        if profile:
            profile_data["filter_ms"] = round((time.perf_counter() - t3) * 1000, 3)
            profile_data["total_ms"] = round(sum(v for v in profile_data.values() if isinstance(v, (int, float))), 3)

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

        if profile:
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

        if profile:
            result["_profile_ms"] = profile_data

        return result