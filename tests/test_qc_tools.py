"""Tests for the QC helpers: session report and frame-folder resolution."""
import json
import tempfile
import unittest
from pathlib import Path

from QC_tools import ocread_session_report as report


def _cycle(text, status="OK", reason=None, reads=(), t="2026-10-13 09:43:03", cycle_ms=3100.0):
    r = {"timestamp": t + ".000", "mode": "ocread", "status": status, "text": text,
         "cycle_time_ms": cycle_ms, "processing_time_ms": cycle_ms - 100}
    if reason:
        r["failure_reason"] = reason
    if reads:
        r["validation"] = {"accepted": status == "OK", "reason": reason,
                           "reads": [dict(variant=v, text=tx, reason=rs, min_token_confidence=c) for v, tx, rs, c in reads]}
    return r


class SessionReportTests(unittest.TestCase):
    def _write(self, rows):
        d = tempfile.mkdtemp()
        p = Path(d) / "2026-10-13.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return p

    def test_counts_reasons_and_variants(self):
        good, bad = "4113 08:26 BB 05/26", "4113 08:25 BB 05/26"
        rows = [
            _cycle(good, reads=[("dot_divide", good, None, 0.95), ("dot_subtract", good, None, 0.91)]),
            _cycle(bad, reads=[("dot_divide", bad, None, 0.97), ("dot_subtract", bad, None, 0.96)], t="2026-10-13 09:44:00"),
            _cycle("4113 08:2 BB 05/26", status="NOK", reason="format_mismatch",
                   reads=[("dot_divide", None, "format_mismatch", None)], t="2026-10-13 09:45:00"),
            {"timestamp": "2026-10-13 09:46:00", "mode": "ocr", "status": "OK"},  # other mode: ignored
        ]
        out = report.summarize(report.load_cycles([self._write(rows)]), expected=good)
        self.assertIn("3 OCRead cycles", out)
        self.assertIn("OK 2   NOK 1", out)
        self.assertIn("format_mismatch 1", out)
        self.assertIn("correct OK 1   WRONG OK 1   rejected 1", out)
        self.assertIn("dot_divide", out)
        self.assertIn("'4113 08:2 BB 05/26'", out)

    def test_time_filter_and_old_results_without_validation(self):
        rows = [_cycle("4113 08:26 BB 05/26", t="2026-10-13 08:00:00"),
                _cycle("4113 08:26 BB 05/26", t="2026-10-13 10:00:00")]
        cycles = report.load_cycles([self._write(rows)], since="09:00")
        self.assertEqual(len(cycles), 1)
        self.assertIn("1 OCRead cycles", report.summarize(cycles))

    def test_missing_file_is_reported(self):
        self.assertEqual(report.main(["does/not/exist.jsonl"]), 1)


class TimeWindowTests(unittest.TestCase):
    def test_folder_time_filter(self):
        from QC_tools.compare_ocr_preprocessing import _in_window
        p = Path("data/debug/dynamic_roi/20261013_105627_073/text_locator_search.png")
        self.assertTrue(_in_window(p, "10:56", "10:57:37"))
        self.assertTrue(_in_window(p, "10:56:27", None))
        self.assertTrue(_in_window(p, None, "10:56"))          # HH:MM includes the whole minute
        self.assertFalse(_in_window(p, "10:57", None))
        self.assertFalse(_in_window(p, None, "10:56:20"))
        self.assertFalse(_in_window(Path("frames/x.png"), "10:00", None))


class PipelineListTests(unittest.TestCase):
    def test_every_variant_is_measured_on_its_own(self):
        from unittest import mock
        from backend.detection.ocr import paddle_ocr
        from QC_tools.compare_ocr_preprocessing import _pipelines
        class W:
            def __init__(self, **k): pass
        with mock.patch.object(paddle_ocr, "PaddleOCRWorker", W):
            e = paddle_ocr.PaddleOCR(app_state=None)
        names = [n for n, _ in _pipelines(e, [0.7])]
        for v in e.PREPROCESS_VARIANTS:
            self.assertIn(f"single:{v}", names)
        self.assertIn("configured", names); self.assertIn("bridge=0.7", names)


class RecDatasetTests(unittest.TestCase):
    def test_font_covers_the_dictionary_and_codes_fit_the_template(self):
        import re
        import numpy as np
        from QC_tools import make_rec_dataset as m
        self.assertTrue(all(c in m.FONT for c in m.DICT))
        gen = m.Generator.__new__(m.Generator); gen.rng = np.random.default_rng(0)
        for _ in range(50):
            l1, l2 = gen._code_lines()
            self.assertRegex(l1, r"^\d{4} ?\d{2}:\d{2}$"); self.assertRegex(l2, r"^BB ?\d{2}/\d{2}$")

    def test_rib_keeps_size_and_only_changes_brightness_locally(self):
        import numpy as np
        from QC_tools import make_rec_dataset as m
        img = np.full((60, 200, 3), 150, np.uint8)
        out = m.add_rib(img, (100, 30), 90)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(int(out[:, 0].mean()), 150)                    # far from the rib: unchanged
        self.assertLess(int(out[:, 60:140].min()), 150)                 # shadow side of the rib


class OCReadRecognizerTests(unittest.TestCase):
    def test_ocread_only_recognizer_is_loaded_and_used(self):
        from unittest import mock
        import numpy as np
        from backend.detection.ocr import paddle_ocr
        made = []
        class W:
            def __init__(self, **k): self.k = k; self.calls = 0; made.append(self)
            def ocr(self, *a, **k): self.calls += 1; return []
        real = dict(paddle_ocr.cfg["ocr"].get("ocread_recognition") or {})
        with mock.patch.dict(paddle_ocr.cfg["ocr"], {"ocread_recognition": {"model_dir": "models/ocread_rec", "model_name": "PP-OCRv5_mobile_rec"}}), \
             mock.patch.object(paddle_ocr, "PaddleOCRWorker", W):
            e = paddle_ocr.PaddleOCR(app_state=None)
        self.assertEqual(len(made), 2)
        general, ocread = made
        self.assertTrue(ocread.k["text_recognition_model_dir"].replace("\\", "/").endswith("models/ocread_rec"))
        self.assertIs(e._rec_worker(True), ocread)       # OCRead
        self.assertIs(e._rec_worker(False), general)     # OCR pipeline keeps the general model


class FrameFolderTests(unittest.TestCase):
    def test_missing_folder_says_where_it_looked(self):
        from QC_tools import compare_ocr_preprocessing as tool
        self.assertIsNone(tool._resolve_frames_root(Path("no/such/folder")))

    def test_relative_debug_dir_is_project_root_relative(self):
        from backend.detection.ocr.paddle_ocr import PROJECT_ROOT, resolve_project_path
        self.assertEqual(Path(resolve_project_path("data/debug/dynamic_roi")), PROJECT_ROOT / "data/debug/dynamic_roi")
        absolute = str(Path.cwd().resolve())
        self.assertEqual(resolve_project_path(absolute), absolute)

if __name__ == "__main__":
    unittest.main()
