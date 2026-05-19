from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytesseract

from backend.detection.ocr.tesseract_ocr import TesseractOCR


TOTAL_TESTS = 3


def _run_test(test_name: str, test_fn) -> bool:
    try:
        test_fn()
        return True
    except Exception as exc:
        print(f"FAIL: {test_name} — {exc}")
        return False


def _make_ocr() -> TesseractOCR:
    ocr = TesseractOCR()
    ocr._apply_roi = lambda frame: frame  # type: ignore[method-assign]
    ocr._downscale_roi = lambda gray: gray  # type: ignore[method-assign]
    ocr._preprocess_image = lambda gray: gray  # type: ignore[method-assign]
    return ocr


def _with_image_to_data(payload: dict[str, list[Any]], fn) -> Any:
    original = pytesseract.image_to_data
    pytesseract.image_to_data = lambda *args, **kwargs: payload  # type: ignore[assignment]
    try:
        return fn()
    finally:
        pytesseract.image_to_data = original  # type: ignore[assignment]


def test_1_phrase_level_keyword_match() -> None:
    ocr = _make_ocr()
    ocr.keywords = ["user manual"]
    ocr.keyword_set = {"user manual"}

    payload = {
        "text": ["User", "Manual", ""],
        "conf": ["90", "80", "-1"],
    }
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    result = _with_image_to_data(payload, lambda: ocr.run(frame))
    detections = result["detections"]

    assert len(detections) == 1, f"unexpected detections: {detections}"
    assert detections[0]["text"] == "User Manual", f"unexpected text: {detections}"
    assert abs(detections[0]["confidence"] - 0.85) < 1e-9, f"unexpected confidence: {detections}"
    print("PASS: phrase-level keyword matching returns one combined detection")


def test_2_split_token_fallback_match() -> None:
    ocr = _make_ocr()
    ocr.keywords = ["user manual"]
    ocr.keyword_set = {"user manual"}

    payload = {
        "text": ["User", "Manu"],
        "conf": ["90", "80"],
    }
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    result = _with_image_to_data(payload, lambda: ocr.run(frame))
    detections = result["detections"]

    assert len(detections) == 2, f"unexpected detections: {detections}"
    assert {d["text"] for d in detections} == {"User", "Manu"}, f"unexpected texts: {detections}"
    print("PASS: split-token fallback keeps partial token matches")


def test_3_empty_keyword_set_uses_date_filter() -> None:
    ocr = _make_ocr()
    ocr.keywords = []
    ocr.keyword_set = set()
    ocr._date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")

    payload = {
        "text": ["2026-05-19", "Ignore", ""],
        "conf": ["90", "90", "90"],
    }
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    result = _with_image_to_data(payload, lambda: ocr.run(frame))
    detections = result["detections"]

    assert len(detections) == 1, f"unexpected detections: {detections}"
    assert detections[0]["text"] == "2026-05-19", f"unexpected text: {detections}"
    print("PASS: empty keyword set respects date-pattern filtering")


def main() -> None:
    tests = [
        ("TEST 1 - Phrase-level keyword match", test_1_phrase_level_keyword_match),
        ("TEST 2 - Split-token fallback match", test_2_split_token_fallback_match),
        ("TEST 3 - Empty keyword set date filtering", test_3_empty_keyword_set_uses_date_filter),
    ]

    passed = 0
    for test_name, test_fn in tests:
        if _run_test(test_name, test_fn):
            passed += 1

    print(f"{passed}/{TOTAL_TESTS} tests passed")


if __name__ == "__main__":
    main()

