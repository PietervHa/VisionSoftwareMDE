from __future__ import annotations
import time
from typing import Any, Dict, Optional
from backend.detection.preprocessing import preprocess_to_pil
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class InspectionEngine:
    def __init__(self, app_state):
        self.app_state = app_state
        self._classifier = None

    def load_classifier(self, model_path: str) -> bool:
        try:
            from backend.detection.classifier import ImageClassifier

            classifier = ImageClassifier(model_path=model_path)
            if not classifier.is_loaded():
                logger.warning("Classifier could not be loaded: model_path=%s", model_path)
                self._classifier = None
                return False

            self._classifier = classifier
            return True
        except Exception as exc:
            logger.error("Failed to load classifier: model_path=%s error=%s", model_path, exc)
            self._classifier = None
            return False

    def evaluate(self, frame) -> dict:
        start_time = time.perf_counter()

        if self._classifier is None or not self._classifier.is_loaded():
            return {
                "status": "NOK",
                "error": "no_model",
                "detections": [],
                "processing_time_ms": 0,
            }

        try:
            pil_image = preprocess_to_pil(frame, size=224)
            prediction = self._classifier.predict(pil_image)
            threshold = self.app_state.get_threshold()
            status = "OK" if prediction["confidence"] >= threshold and prediction["label"] == "ok" else "NOK"
            processing_time_ms = round((time.perf_counter() - start_time) * 1000, 3)

            return {
                "status": status,
                "label": prediction["label"],
                "confidence": prediction["confidence"],
                "all_scores": prediction["all_scores"],
                "processing_time_ms": processing_time_ms,
                "mode": "object_detection",
            }
        except Exception as exc:
            processing_time_ms = round((time.perf_counter() - start_time) * 1000, 3)
            logger.error("Inspection evaluation failed: %s", exc)
            return {
                "status": "NOK",
                "error": str(exc),
                "detections": [],
                "processing_time_ms": processing_time_ms,
            }

    def has_model(self) -> bool:
        return self._classifier is not None and self._classifier.is_loaded()

