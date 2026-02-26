import pytesseract
import cv2
import re
from state import latest_result, lock
import config
import easyocr

reader = easyocr.Reader(['nl','en']) # this needs to run only once to load the model into memory
#reader = easyocr.Reader(['nl','en'], gpu=False) # enable this instead to use CPU only
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class OCR:
    def __init__(self):
        # Languages for Tesseract
        self.languages = "eng+nld"

        # Keywords to detect
        self.keywords = [w.lower() for w in getattr(config, "EXPECTED_KEYWORDS", [])]

        # Optional regex patterns (like dates)
        self.date_regex = getattr(config, "DATE_REGEX", None)

        # Confidence threshold for Tesseract (0-100)
        self.confidence_threshold = 60

        # ROI
        self.roi = getattr(config, "ROI", {
            "x_start": 0.0,
            "y_start": 0.0,
            "x_end": 1.0,
            "y_end": 1.0
        })

        self.debug_draw_roi = getattr(config, "DEBUG_DRAW_ROI", False)

    def _apply_roi(self, frame):
        h, w = frame.shape[:2]
        x1 = int(self.roi["x_start"] * w)
        y1 = int(self.roi["y_start"] * h)
        x2 = int(self.roi["x_end"] * w)
        y2 = int(self.roi["y_end"] * h)
        roi_frame = frame[y1:y2, x1:x2]

        if self.debug_draw_roi:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return roi_frame

    def run(self, frame):
        # Apply ROI
        roi_frame = self._apply_roi(frame)

        # Convert to grayscale for OCR
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        # Run Tesseract OCR
        data = pytesseract.image_to_data(
            gray,
            lang=self.languages,
            output_type=pytesseract.Output.DICT
        )

        detections = []

        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue

            conf = int(data["conf"][i])
            if conf < self.confidence_threshold:
                continue

            word = text.lower()

            # Match keywords
            if self.keywords and word not in self.keywords:
                # Also check regex if defined
                if self.date_regex and not re.search(self.date_regex, text):
                    continue

            detections.append({
                "text": text,
                "confidence": conf
            })

        # Update shared state
        with lock:
            latest_result["mode"] = "ocr"
            latest_result["detections"] = detections

        return detections