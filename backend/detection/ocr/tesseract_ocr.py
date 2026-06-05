import pytesseract
import cv2
import re
import time
from backend.core.config_loader import cfg
from backend.utils.logger import get_logger
from backend.utils.roi import apply_roi, draw_roi

log = get_logger(__name__)

pytesseract.pytesseract.tesseract_cmd = cfg["ocr"]["tesseract_path"]


class TesseractOCR:
    def __init__(self, app_state=None):
        self.app_state = app_state
        self.languages = "eng"
        ocr_cfg = cfg["ocr"]
        psm = ocr_cfg["psm"]
        oem = ocr_cfg["oem"]
        self.tesseract_config = f"--psm {psm} --oem {oem}"
        whitelist = ocr_cfg["whitelist"]
        if whitelist:
            self.tesseract_config += f" -c tessedit_char_whitelist={whitelist}"
        if ocr_cfg["disable_dawgs"]:
            self.tesseract_config += " -c load_system_dawg=0 -c load_freq_dawg=0"
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
            "OCR init: preprocess_mode=%s downscale=%s",
            self.preprocess_mode,
            self.downscale,
        )

    def _preprocess_image(self, gray):
        """Enhance image contrast and clarity for Tesseract.

        Changes vs original:
        - Upscale small ROIs: Tesseract accuracy drops sharply when text
          height is below ~30 px; 2x cubic upscaling fixes most cases.
        - Default mode now uses adaptive thresholding instead of CLAHE+blur.
          Adaptive threshold handles uneven/coloured backgrounds reliably;
          Otsu and CLAHE+blur both struggle when text/background contrast
          is low in the grayscale domain (e.g. black text on dark red).
        """
        mode = self.preprocess_mode
        if mode == "off":
            return gray

        # Upscale if the shorter dimension is small – keeps text sharp for Tesseract
        h, w = gray.shape[:2]
        short = min(h, w)
        if short < 80:
            scale = max(2, int(160 / short))
            gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        elif short < 200:
            gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        if mode == "fast":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary

        # Default mode: adaptive Gaussian threshold
        # blockSize must be odd and large enough to span a character; C is the
        # constant subtracted from the local mean (tune higher = more aggressive).
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10
        )
        # Tesseract expects dark text on a white background.
        # If the image is mostly dark (inverted polarity), flip it.
        if cv2.countNonZero(binary) < binary.size * 0.3:
            binary = cv2.bitwise_not(binary)
        return binary

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
        """Crop the frame to the configured ROI, optionally drawing the debug overlay.

        delegates to shared roi utility instead of duplicating the
        coordinate math that also lives in web.py and paddle_ocr.py.
        """
        roi = cfg.get("roi")
        if not roi:
            return frame

        if self.debug_draw_roi:
            draw_roi(frame, roi)

        return apply_roi(frame, roi)

    def run(self, frame, profile=False):
        profile_data = {} if profile else None

        t0 = time.perf_counter()
        roi_frame = self._apply_roi(frame)
        if profile_data is not None:
            profile_data["roi_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        t1 = time.perf_counter()
        # Use the BGR channel with the highest standard deviation as grayscale.
        # This gives much better contrast than the standard luminance blend when
        # text sits on a strongly-coloured background (e.g. black on dark red).
        if len(roi_frame.shape) == 3:
            channels = cv2.split(roi_frame)          # B, G, R
            gray = max(channels, key=lambda c: float(c.std()))
        else:
            gray = roi_frame
        if profile_data is not None:
            profile_data["grayscale_ms"] = round((time.perf_counter() - t1) * 1000, 3)

        keywords = (
            [self.app_state.get_ocr_keyword()]
            if self.app_state else self.keywords
        )
        keyword_set = {k.lower() for k in keywords if isinstance(k, str)} if self.app_state else self.keyword_set

        t2 = time.perf_counter()
        gray = self._downscale_roi(gray)
        if profile_data is not None:
            profile_data["downscale_ms"] = round((time.perf_counter() - t2) * 1000, 3)

        t3 = time.perf_counter()
        preprocessed = self._preprocess_image(gray)
        if profile_data is not None:
            profile_data["preprocess_ms"] = round((time.perf_counter() - t3) * 1000, 3)

        start_time = time.perf_counter()

        t4 = start_time
        data = pytesseract.image_to_data(
            preprocessed,
            lang=self.languages,
            config=self.tesseract_config,
            output_type=pytesseract.Output.DICT
        )
        if profile_data is not None:
            profile_data["ocr_ms"] = round((time.perf_counter() - t4) * 1000, 3)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        detections = []
        texts = data.get("text", [])
        confs = data.get("conf", [])

        # --- Collect valid tokens (confidence >= 0, non-empty) ---
        valid_tokens = []
        t5 = time.perf_counter()
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            try:
                conf = float(confs[i])
            except (TypeError, ValueError, IndexError):
                conf = -1.0
            if conf < 0:
                continue
            valid_tokens.append((text, conf))

        # Build the full recognised line for phrase-level matching.
        # Tesseract image_to_data returns one token per call, so a multi-word
        # keyword like "User Manual" would never pass an exact per-token check.
        full_text = " ".join(t for t, _ in valid_tokens).lower()
        avg_conf  = (sum(c for _, c in valid_tokens) / len(valid_tokens)) if valid_tokens else 0.0

        # Keep an original-cased version for reporting
        original_full = " ".join(t for t, _ in valid_tokens)

        if keyword_set:
            # 1. Phrase-level check – preferred path
            #    full_text is lowercased; original_full preserves OCR casing.
            #    We report the original-cased slice so the detection text
            #    matches what Tesseract actually read (e.g. "User Manual",
            #    not the lowercased keyword "user manual").
            for kw in keyword_set:
                idx = full_text.find(kw)
                if idx >= 0:
                    matched_text = original_full[idx: idx + len(kw)]
                    detections.append({"text": matched_text, "confidence": round(avg_conf / 100, 3)})

            # 2. Per-token fallback for split/partial results
            #    Require len >= 2 to avoid single-char noise like "a" matching
            #    as a substring of the keyword (e.g. "user mAnual").
            if not detections:
                for text, conf in valid_tokens:
                    if len(text.strip()) < 2:
                        continue
                    word = text.lower()
                    phrase_match = any(kw in word or word in kw for kw in keyword_set)
                    date_match   = self._date_pattern and self._date_pattern.search(text)
                    if phrase_match or date_match:
                        detections.append({"text": text, "confidence": round(conf / 100, 3)})
        else:
            # No keyword filter – return all tokens that pass the date check
            for text, conf in valid_tokens:
                if self._date_pattern and not self._date_pattern.search(text):
                    continue
                detections.append({"text": text, "confidence": round(conf / 100, 3)})
        if profile_data is not None:
            profile_data["filter_ms"] = round((time.perf_counter() - t5) * 1000, 3)
            profile_data["total_ms"] = round(sum(profile_data.values()), 3)

        processing_time_ms = round(elapsed_ms, 1)
        log.debug("OCR run completed: processing_time_ms=%s", processing_time_ms)
        result = {
            "detections": detections,
            "processing_time_ms": processing_time_ms,
            "mode": "ocr",
            "searched_word": keywords[0] if keywords else ""
        }

        if profile_data is not None:
            result["_profile_ms"] = profile_data

        return result