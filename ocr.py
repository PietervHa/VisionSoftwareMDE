# ocr.py
import pytesseract
import cv2
from state import latest_result, lock
from config import EXPECTED_KEYWORDS


class OCR:
    def __init__(self):
        self.languages = "eng"
        self.keywords = [w.lower() for w in EXPECTED_KEYWORDS]
        self.confidence_threshold = 60

    def run(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        data = pytesseract.image_to_data(
            gray,
            lang=self.languages,
            output_type=pytesseract.Output.DICT
        )

        detected_words = []

        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue

            conf = int(data["conf"][i])
            word = text.lower()

            if conf < self.confidence_threshold:
                continue

            if self.keywords and word not in self.keywords:
                continue

            detected_words.append({
                "text": text,
                "confidence": conf
            })

        with lock:
            latest_result["mode"] = "ocr"
            latest_result["detections"] = detected_words

        return detected_words