"""Shared image preprocessing for model inference and dataset pipelines.

This module defines a single preprocessing entry point that should be reused
across inference, training, and dataset generation to avoid train/inference
mismatch.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


Size = Tuple[int, int]


def _validate_size(size: Size) -> Size:
    if not isinstance(size, tuple) or len(size) != 2:
        raise TypeError("size must be a tuple of (width, height)")

    width, height = size
    if not isinstance(width, int) or not isinstance(height, int):
        raise TypeError("size values must be integers")
    if width <= 0 or height <= 0:
        raise ValueError("size values must be > 0")

    return width, height


def preprocess(frame: np.ndarray, size: Size, to_rgb: bool = True) -> np.ndarray:
    """Preprocess an OpenCV frame into a model-ready float32 image tensor.

    Args:
        frame: Input image as a NumPy array in OpenCV format (H, W, C), BGR order.
        size: Target output size as (width, height).
        to_rgb: If True (default), convert BGR to RGB after resize.

    Returns:
        A NumPy array with shape (height, width, channels), dtype float32,
        normalized to [0.0, 1.0]. Channel order is RGB when to_rgb=True,
        otherwise BGR.

    Raises:
        TypeError: If frame or size types are invalid.
        ValueError: If frame shape or size values are invalid.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a NumPy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape (height, width, 3)")

    width, height = _validate_size(size)

    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    if to_rgb:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    return resized.astype(np.float32) / 255.0


# Example usage: YOLO inference (reusable pipeline step)
# model_input = preprocess(frame, size=(640, 640), to_rgb=True)
# results = yolo_model(model_input)

# Example usage: future classifier pipeline
# classifier_input = preprocess(frame, size=(224, 224), to_rgb=True)
# logits = classifier_model(classifier_input)

