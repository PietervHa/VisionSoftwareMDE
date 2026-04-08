from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, cast
import numpy as np
from backend.detection.preprocessing import preprocess

try:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification
except ImportError:  # pragma: no cover - dependency is validated at runtime
    torch = None
    AutoImageProcessor = None
    AutoModelForImageClassification = None


Size = Tuple[int, int]


class ImageClassifier:
    """Thin wrapper around a Hugging Face image classification model.

    Args:
        model_path: Local path or model name on Hugging Face Hub.
        device: Optional device override (for example: "cpu", "cuda", "cuda:0").
        label_mapping: Optional class index to label mapping.
        threshold: Optional confidence threshold for downstream decision logic.
        input_size: Optional input size as (width, height). If not provided,
            the size is inferred from the processor configuration.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        label_mapping: Optional[Dict[int, str]] = None,
        threshold: Optional[float] = None,
        input_size: Optional[Size] = None,
    ) -> None:
        if torch is None or AutoImageProcessor is None or AutoModelForImageClassification is None:
            raise ImportError(
                "Image classification dependencies are missing. Install 'torch' and 'transformers'."
            )

        processor_cls = cast(Any, AutoImageProcessor)
        model_cls = cast(Any, AutoModelForImageClassification)

        self.model_path = model_path or "google/vit-base-patch16-224"
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        self.processor = processor_cls.from_pretrained(self.model_path)
        self.model = model_cls.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        self.input_size = input_size or self._infer_input_size(self.processor)
        self.label_mapping = label_mapping or self._build_label_mapping(self.model.config.id2label)

    @staticmethod
    def _build_label_mapping(id2label: Optional[Dict[object, str]]) -> Dict[int, str]:
        if not id2label:
            return {}

        mapping = {}
        for key, value in id2label.items():
            try:
                mapping[int(str(key))] = value
            except (TypeError, ValueError):
                continue
        return mapping

    @staticmethod
    def _infer_input_size(processor) -> Size:
        size = getattr(processor, "size", None)
        if isinstance(size, dict):
            height = size.get("height") or size.get("shortest_edge")
            width = size.get("width") or size.get("shortest_edge")
            if isinstance(width, int) and isinstance(height, int):
                return width, height

        # Safe fallback if processor metadata does not expose explicit dimensions.
        return 224, 224

    def _resolve_label(self, class_index: int) -> str:
        return self.label_mapping.get(class_index, str(class_index))

    def predict(self, frame: np.ndarray) -> Dict[str, object]:
        """Run full-frame classification and return label, confidence, and raw output.

        The shared preprocess() function is always used to keep transformations
        identical across inference, testing, and training-related pipelines.
        """
        image = preprocess(frame, size=self.input_size, to_rgb=True)

        inputs = self.processor(
            images=image,
            return_tensors="pt",
            do_rescale=False,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        top_confidence, top_index = torch.max(probabilities[0], dim=-1)
        class_index = int(top_index.item())

        result = {
            "label": self._resolve_label(class_index),
            "confidence": float(top_confidence.item()),
            "raw": {
                "logits": logits[0].detach().cpu().tolist(),
                "probabilities": probabilities[0].detach().cpu().tolist(),
            },
        }

        if self.threshold is not None:
            result["meets_threshold"] = result["confidence"] >= float(self.threshold)

        return result



