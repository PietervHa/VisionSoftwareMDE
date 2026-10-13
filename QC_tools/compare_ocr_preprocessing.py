"""
Compare OCRead pipelines on saved camera frames, using the real PaddleOCR
engine and the real OCRead code path (locate -> deskew -> preprocess ->
recognise -> orientation retry -> output filter -> read validation).

Pipelines:
  previous             derib + CLAHE, no validation (configuration before 2026-10)
  dot_text_raw         dot_text, no validation (any text = OK)
  validated_single     read validation, one read (first consensus variant)
  validated_consensus  read validation + consensus, as configured in default.yaml
  bridge=X             validated_consensus with dot_text bridge_sigma_px=X (--bridge)

What to look at, with --expected: "WRONG OK" is the number that matters in
production - a code reported as OK that is not what is printed. A reject is
safe (NOK, with its reason); a wrong OK is not. Without --expected you get
OK/NOK counts, reject reasons and the most common reads.

Getting frames: set ocr.dynamic_roi.debug_save: true, run a session, then
point this at data/debug/dynamic_roi - every cycle folder contains
text_locator_search.png, the raw frame OCR saw. Any folder of .png/.jpg
frames works too (--glob). Run once per bottle/expected code.

Usage (from the repo root):
    python -m QC_tools.compare_ocr_preprocessing data/debug/dynamic_roi --expected "4113 08:26 BB 05/26"
    python -m QC_tools.compare_ocr_preprocessing frames/ --glob "*.png" --bridge 0.5,1.0 --csv out.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


def _norm(text) -> str:
    return "".join(str(text or "").split())


def _find_frames(root: Path, pattern: str | None) -> list[Path]:
    if root.is_file():
        return [root]
    if pattern:
        return sorted(root.glob(pattern))
    frames = sorted(root.glob("**/text_locator_search.png"))
    if frames:
        return frames
    return sorted([*root.glob("**/*.png"), *root.glob("**/*.jpg"), *root.glob("**/*.jpeg")])


def _pipelines(engine, bridge_values):
    dot_cfg = dict(engine.dot_text_cfg or {})
    validator = engine.validator
    variants = list(engine.consensus_variants)
    need = engine.consensus_min_agree
    p = [
        ("previous", dict(validator=None, consensus_variants=[None], consensus_min_agree=1,
                          dot_text_enabled=False, derib_enabled=True, dot_text_cfg=dot_cfg)),
        ("dot_text_raw", dict(validator=None, consensus_variants=[None], consensus_min_agree=1,
                              dot_text_enabled=True, derib_enabled=False, dot_text_cfg=dot_cfg)),
    ]
    if validator is not None:
        p += [
            ("validated_single", dict(validator=validator, consensus_variants=variants[:1], consensus_min_agree=1,
                                      dot_text_enabled=True, derib_enabled=False, dot_text_cfg=dot_cfg)),
            ("validated_consensus", dict(validator=validator, consensus_variants=variants, consensus_min_agree=need,
                                         dot_text_enabled=True, derib_enabled=False, dot_text_cfg=dot_cfg)),
        ]
        for b in bridge_values:
            p.append((f"bridge={b:g}", dict(validator=validator, consensus_variants=variants, consensus_min_agree=need,
                                            dot_text_enabled=True, derib_enabled=False,
                                            dot_text_cfg=dict(dot_cfg, bridge_sigma_px=b))))
    else:
        print("Note: ocr.read_validation is disabled in the config - validated pipelines skipped.")
    return p


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frames", type=Path, help="folder (searched recursively) or a single image")
    parser.add_argument("--expected", default=None, help='expected code, e.g. "4113 08:26 BB 05/26"')
    parser.add_argument("--glob", default=None, help="glob pattern inside the folder (default: debug_save frames)")
    parser.add_argument("--bridge", default="", help="comma-separated bridge_sigma_px values to add")
    parser.add_argument("--csv", type=Path, default=None, help="write one row per frame and pipeline")
    parser.add_argument("--limit", type=int, default=0, help="only use the first N frames")
    parser.add_argument("--config", default=None, help="config YAML (read by backend.core.config_loader)")
    args = parser.parse_args(argv)

    frames = _find_frames(args.frames, args.glob)
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        print(f"No frames found under {args.frames}")
        return 1

    from backend.detection.ocr.paddle_ocr import PaddleOCR

    engine = PaddleOCR(app_state=None)   # starts the real PaddleOCR worker
    engine.dynamic_roi_debug_dir = None  # don't write debug images while comparing
    pipelines = _pipelines(engine, [float(v) for v in args.bridge.split(",") if v.strip()])
    expected = _norm(args.expected) if args.expected else None
    print(f"{len(frames)} frames, {len(pipelines)} pipelines"
          + (f", expected {args.expected!r}" if args.expected else ""))

    stats = {name: collections.Counter() for name, _ in pipelines}
    reasons = {name: collections.Counter() for name, _ in pipelines}
    reads = {name: collections.Counter() for name, _ in pipelines}
    ok_conf = {name: [] for name, _ in pipelines}
    ms = {name: [] for name, _ in pipelines}
    rows = []
    for i, path in enumerate(frames, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  skipped unreadable {path}")
            continue
        for name, settings in pipelines:
            for attr, value in settings.items():
                setattr(engine, attr, value)
            r = engine.read(frame)
            ms[name].append(r.get("processing_time_ms", 0.0))
            validation = r.get("validation")
            if r.get("error"):
                ok, reason = False, "ocr_engine_error"
            elif validation is not None:
                ok, reason = bool(validation["accepted"]), validation.get("reason")
            else:
                ok = bool(_norm(r.get("text")))
                reason = None if ok else "no_text_detected"
            text = r.get("text", "")
            reads[name][(("OK  " if ok else "NOK ") + text)] += 1
            if ok:
                hit = expected is None or _norm(text) == expected
                stats[name]["correct_ok" if hit else "WRONG_OK"] += 1
                if validation:
                    used = [x.get("min_token_confidence") for x in validation["reads"]
                            if x.get("accepted") and x.get("text") == text]
                    ok_conf[name] += [c for c in used if c is not None]
            else:
                stats[name]["rejected"] += 1
                reasons[name][reason] += 1
            rows.append({"frame": str(path), "pipeline": name, "status": "OK" if ok else "NOK",
                         "text": text, "reason": reason or "",
                         "correct": ("" if expected is None else int(ok and _norm(text) == expected))})
        if i % 10 == 0:
            print(f"  {i}/{len(frames)} frames done")

    n = len({r["frame"] for r in rows})
    print()
    for name, _ in pipelines:
        s = stats[name]
        if expected is not None:
            head = (f"correct OK {s['correct_ok']:4d}   WRONG OK {s['WRONG_OK']:4d}   "
                    f"rejected {s['rejected']:4d}   (of {n})")
        else:
            head = f"OK {s['correct_ok']:4d}   NOK {s['rejected']:4d}   (of {n})"
        print(f"{name:20s} {head}   ocr {np.mean(ms[name]):.0f} ms/frame")
        if reasons[name]:
            print("      reject reasons: " + ", ".join(f"{k}={v}" for k, v in reasons[name].most_common()))
        if ok_conf[name]:
            print(f"      min token confidence of accepted reads: lowest {min(ok_conf[name]):.3f}, "
                  f"5th percentile {np.percentile(ok_conf[name], 5):.3f}")
        for text, count in reads[name].most_common(4):
            print(f"      {count:4d}x  {text!r}")

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["frame", "pipeline", "status", "text", "reason", "correct"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-frame results written to {args.csv}")

    try:
        engine._paddle.shutdown()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
