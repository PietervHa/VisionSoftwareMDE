import pytesseract
import cv2
import re
import config
import time
import easyocr

#reader = easyocr.Reader(['nl','en']) # this needs to run only once to load the model into memory
#reader = easyocr.Reader(['nl','en'], gpu=False) # enable this instead to use CPU only
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class OCR:
    def __init__(self):
        self.languages = "eng+nld"
        self.keywords = [w.lower() for w in getattr(config, "EXPECTED_KEYWORDS", [])]
        self.date_regex = getattr(config, "DATE_REGEX", None)
        self.debug_draw_roi = getattr(config, "DEBUG_DRAW_ROI", False)

    def _apply_roi(self, frame):
        roi = getattr(config, "ROI", None)
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

        return detections, round(elapsed_ms, 1)
"""
    def run(self, frame):
        roi_frame = self._apply_roi(frame)
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        start_time = time.perf_counter()

        data = pytesseract.image_to_data(
            gray,
            lang=self.languages,
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
        return detections, round(elapsed_ms, 1)


    """
    LET OP CHEFKE !!!!
    Return hetzelfde opbouwen als in objectdetection,
    nu is het anders en word het pas in de vision.py gefixt
    """
