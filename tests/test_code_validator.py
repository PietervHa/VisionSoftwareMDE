"""
Tests for OCRead read validation (code_validator.py) and the consensus loop
in PaddleOCR.read(). The rule being protected: the system may reject a
read, but must never invent or "repair" information into a pass.

Run with:  python -m unittest tests.test_code_validator
"""
import unittest
from unittest import mock

import numpy as np

from backend.detection.ocr.code_validator import CodeTemplate

PATTERN = r"^\d{4} ([01]\d|2[0-3]):[0-5]\d BB (0[1-9]|1[0-2])/\d{2}$"


def T(**kw):
    args = dict(value_pattern=PATTERN, min_confidence=0.8, max_corrections=1, max_ignored_tokens=2)
    args.update(kw)
    return CodeTemplate(["#### ##:##", "BB ##/##"], **args)


def toks(*texts, conf=0.99):
    return [{"text": t, "confidence": conf} for t in texts]


class TemplateTests(unittest.TestCase):
    def test_tokens_assemble_into_the_canonical_code(self):
        r = T().validate(toks("4113", "08:26", "BB", "05/26"))
        self.assertTrue(r["accepted"])
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")

    def test_spacing_from_the_model_does_not_matter(self):
        r = T().validate(toks("411308:26", "BB05/26"))
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")

    def test_lookalikes_in_literal_positions_are_normalised(self):
        r = T().validate(toks("4113 08.26", "B8 05126"))
        self.assertTrue(r["accepted"])
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")
        self.assertEqual(r["corrections"], 0)

    def test_fullwidth_colon_from_multilingual_model(self):
        self.assertEqual(T().validate(toks("4113 08：26", "BB 05/26"))["text"], "4113 08:26 BB 05/26")

    def test_digit_is_never_changed_into_another_digit(self):
        # 08:25 fits the format - the validator alone cannot know it should
        # be 08:26 (that is what consensus is for), but it must not alter it.
        r = T().validate(toks("4113 08:25", "BB 05/26"))
        self.assertEqual(r["text"], "4113 08:25 BB 05/26")

    def test_one_letter_lookalike_in_a_digit_position_is_allowed(self):
        r = T().validate(toks("4I13 08:26", "BB 05/26"))
        self.assertTrue(r["accepted"])
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")
        self.assertEqual(r["corrections"], 1)

    def test_more_corrections_than_allowed_is_rejected(self):
        r = T().validate(toks("4II3 08:26", "BB 05/26"))
        self.assertFalse(r["accepted"])
        self.assertEqual(r["reason"], "format_mismatch")

    def test_truncated_read_is_rejected(self):
        r = T().validate(toks("4113 08:2", "BB 05/26"))
        self.assertEqual(r["reason"], "format_mismatch")

    def test_text_is_never_cut_out_of_a_token(self):
        # "08:250" must not become "08:25".
        r = T().validate(toks("4113 08:250", "BB 05/26"))
        self.assertEqual(r["reason"], "format_mismatch")

    def test_unknown_character_marker_is_rejected(self):
        r = T().validate(toks("4113 08:2?", "BB 05/26"))
        self.assertEqual(r["reason"], "format_mismatch")

    def test_separate_noise_tokens_are_ignored_up_to_the_limit(self):
        r = T().validate(toks("PE/PA", "5203", "21:28", "BB08/27"))
        self.assertTrue(r["accepted"])
        self.assertEqual(r["text"], "5203 21:28 BB 08/27")
        self.assertEqual(r["ignored_tokens"], ["PE/PA"])
        r = T(max_ignored_tokens=0).validate(toks("PE/PA", "5203", "21:28", "BB08/27"))
        self.assertEqual(r["reason"], "format_mismatch")

    def test_upside_down_cap_token_order_is_recovered(self):
        r = T().validate(toks("05/26", "BB", "08:26", "4113"))
        self.assertTrue(r["accepted"])
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")
        self.assertEqual(r["orientation"], "reversed")

    def test_low_confidence_is_rejected(self):
        r = T().validate([{"text": "4113 08:26", "confidence": 0.99}, {"text": "BB 05/26", "confidence": 0.5}])
        self.assertEqual(r["reason"], "low_confidence")

    def test_impossible_value_is_rejected(self):
        self.assertEqual(T().validate(toks("4113 27:26", "BB 05/26"))["reason"], "invalid_value")
        self.assertEqual(T().validate(toks("4113 08:26", "BB 13/26"))["reason"], "invalid_value")

    def test_nothing_read(self):
        self.assertEqual(T().validate([])["reason"], "no_text_detected")


class _ScriptedWorker:
    """Returns the next scripted answer on each call (one per variant read)."""
    script = []

    def __init__(self, **kwargs):
        self.calls = 0

    def ocr(self, frame, timeout=None):
        answer = _ScriptedWorker.script[min(self.calls, len(_ScriptedWorker.script) - 1)]
        self.calls += 1
        if isinstance(answer, Exception):
            raise answer
        return [{"text": t, "confidence": 0.99} for t in answer]


class ConsensusTests(unittest.TestCase):
    def _engine(self, *script, variants=("dot_divide", "dot_subtract", "classic"), min_agree=2):
        from backend.detection.ocr import paddle_ocr
        _ScriptedWorker.script = list(script)
        with mock.patch.object(paddle_ocr, "PaddleOCRWorker", _ScriptedWorker):
            engine = paddle_ocr.PaddleOCR(app_state=None)
        engine.dynamic_roi_debug_dir = None
        engine.orientation_retry_enabled = False
        engine.validator = T()
        engine.consensus_variants = list(variants)
        engine.consensus_min_agree = min_agree
        return engine

    def _frame(self):
        return np.full((480, 744, 3), 190, np.uint8)

    def test_two_agreeing_reads_stop_early(self):
        e = self._engine(["4113 08:26", "BB 05/26"], ["4113", "08:26", "BB05/26"])
        r = e.read(self._frame())
        self.assertTrue(r["validation"]["accepted"])
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")
        self.assertEqual(e._paddle.calls, 2)
        self.assertNotIn("failure_reason", r)

    def test_confident_misread_is_outvoted(self):
        e = self._engine(["4113 08:25", "BB 05/26"], ["4113 08:26", "BB 05/26"], ["4113 08:26", "BB 05/26"])
        r = e.read(self._frame())
        self.assertEqual(r["text"], "4113 08:26 BB 05/26")
        self.assertEqual(e._paddle.calls, 3)

    def test_disagreement_is_a_reject_not_a_guess(self):
        e = self._engine(["4113 08:25", "BB 05/26"], ["4113 08:26", "BB 05/26"], ["4113 08:28", "BB 05/26"])
        r = e.read(self._frame())
        self.assertFalse(r["validation"]["accepted"])
        self.assertEqual(r["failure_reason"], "no_consensus")

    def test_stops_once_agreement_is_impossible(self):
        e = self._engine(["4113 08:2", "BB 05/26"], ["4113 08:2", "BB 05/26"], ["4113 08:26", "BB 05/26"])
        r = e.read(self._frame())
        self.assertEqual(e._paddle.calls, 2)
        self.assertEqual(r["failure_reason"], "format_mismatch")
        self.assertEqual(r["text"], "4113 08:2 BB 05/26")  # raw read kept for the HMI

    def test_engine_failure_is_still_an_engine_error(self):
        from backend.detection.ocr.paddle_worker import PaddleWorkerError
        e = self._engine(["4113 08:26", "BB 05/26"], PaddleWorkerError("worker died"))
        r = e.read(self._frame())
        self.assertIn("worker died", r["error"])

    def test_single_read_mode(self):
        e = self._engine(["4113 08:26", "BB 05/26"], variants=("dot_divide",), min_agree=1)
        self.assertEqual(e.read(self._frame())["text"], "4113 08:26 BB 05/26")
        self.assertEqual(e._paddle.calls, 1)

    def test_tokens_are_put_in_reading_order_by_their_boxes(self):
        from backend.detection.ocr.paddle_ocr import PaddleOCR
        box = lambda x, y: [[x, y], [x + 40, y], [x + 40, y + 20], [x, y + 20]]
        cands = [{"text": "BB", "box": box(10, 52)}, {"text": "08:26", "box": box(80, 12)},
                 {"text": "05/26", "box": box(70, 50)}, {"text": "4113", "box": box(12, 10)}]
        ordered = [c["text"] for c in PaddleOCR._order_candidates(cands)]
        self.assertEqual(ordered, ["4113", "08:26", "BB", "05/26"])


class VisionStatusTests(unittest.TestCase):
    def test_rejected_read_is_nok_with_its_reason(self):
        try:
            from backend.core import vision
        except Exception as exc:  # heavy ML deps not installed here
            self.skipTest(f"backend.core.vision not importable: {exc}")
        n = vision._normalize_result({"detections": [{"text": "4113 08:2", "confidence": 0.9}],
                                      "text": "4113 08:2", "mode": "ocread",
                                      "validation": {"accepted": False, "reason": "format_mismatch"},
                                      "failure_reason": "format_mismatch"}, "ocread")
        self.assertEqual((n["status"], n["failure_reason"]), ("NOK", "format_mismatch"))


if __name__ == "__main__":
    unittest.main()
