import pytesseract
import cv2
import re
import time
import easyocr
import numpy as np
from config_loader import cfg

pytesseract.pytesseract.tesseract_cmd = cfg["ocr"]["tesseract_path"]


class OCR:
    def __init__(self):
        self.languages = "eng"  # Only English for speed
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
        self.date_regex = ocr_cfg["date_regex"]
        self.debug_draw_roi = cfg["hmi"]["debug_draw_roi"]
        self.preprocess_mode = ocr_cfg["preprocess"].lower()
        self.downscale = float(ocr_cfg["downscale"])
        self.min_dim = int(ocr_cfg["min_dim"])

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

        return frame[y1:y2, x1:x2]

    """
    def run(self, frame):
        roi_frame = self._apply_roi(frame)
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        start_time = time.perf_counter()

        # EasyOCR returns: [ [bbox, text, confidence], ... ]
        results = reader.readtext(gray)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        detections = []

        for bbox, text, conf in results:
            if not text.strip():
                continue

            word = text.lower()

            # Filter by keywords / regex if defined
            if self.keywords and word not in self.keywords:
                if self.date_regex and not re.search(self.date_regex, text):
                    continue

            detections.append({
                "text": text,
                "confidence": round(float(conf), 3)  # already 0–1
            })

        return {
            "detections": detections,
            "processing_time_ms": round(elapsed_ms, 1),
            "mode": "ocr"
        }
"""
    def run(self, frame):
        roi_frame = self._apply_roi(frame)
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        gray = self._downscale_roi(gray)

        # Preprocess for faster/better OCR
        preprocessed = self._preprocess_image(gray)

        start_time = time.perf_counter()

        data = pytesseract.image_to_data(
            preprocessed,
            lang=self.languages,
            config=self.tesseract_config,
            output_type=pytesseract.Output.DICT
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        detections = []

        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue

            conf = int(data["conf"][i])
            word = text.lower()

            # Filter by keywords/regex if defined
            if self.keywords and word not in self.keywords:
                if self.date_regex and not re.search(self.date_regex, text):
                    continue

            detections.append({
                "text": text,
                "confidence": conf / 100  # normalize 0–1
            })

        # Return detections and processing time
        return {
            "detections": detections,
            "processing_time_ms": round(elapsed_ms, 1),
            "mode": "ocr"
        }


