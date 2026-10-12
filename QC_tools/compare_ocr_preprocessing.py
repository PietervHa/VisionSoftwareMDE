"""
Compare OCRead preprocessing variants on saved camera frames, using the real
PaddleOCR engine and the real OCRead pipeline (locate -> deskew -> preprocess
-> recognise -> orientation retry -> output filter).

Variants:
  previous   derib on, dot_text off  (the configuration before 2026-10)
  no_derib   derib off, dot_text off
  dot_text   dot_text on, settings from config/default.yaml
  bridge=X   dot_text with bridge_sigma_px=X   (one per value in --bridge)

Getting frames: set ocr.dynamic_roi.debug_save: true, run a normal session,
then point this at data/debug/dynamic_roi. Every cycle folder there contains
text_locator_search.png, the raw frame OCR saw (full frame when search_roi is
the whole image). Any other folder of .png/.jpg frames works too (--glob).

Usage (from the repo root):
    python -m QC_tools.compare_ocr_preprocessing data/debug/dynamic_roi --expected "4113 08:26 BB 05/26"
    python -m QC_tools.compare_ocr_preprocessing frames/ --glob "*.png" --bridge 0.5,1.0 --csv out.csv

--expected is compared ignoring whitespace. Without it, only the read
distribution per variant is printed. Mix bottles only if they share one
expected text, or run once per bottle.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

import cv2


def _norm(text: str) -> str:
    return "".join(str(text).split())


def _find_frames(root: Path, pattern: str | None) -> list[Path]:
    if root.is_file():
        return [root]
    if pattern:
        return sorted(root.glob(pattern))
    frames = sorted(root.glob("**/text_locator_search.png"))
    if frames:
        return frames
    return sorted([*root.glob("**/*.png"), *root.glob("**/*.jpg"), *root.glob("**/*.jpeg")])


def _variants(engine, bridge_values: list[float]):
    base_dot_cfg = dict(engine.dot_text_cfg or {})
    variants = [
        ("previous", dict(dot_text_enabled=False, derib_enabled=True, dot_text_cfg=base_dot_cfg)),
        ("no_derib", dict(dot_text_enabled=False, derib_enabled=False, dot_text_cfg=base_dot_cfg)),
        ("dot_text", dict(dot_text_enabled=True, derib_enabled=False, dot_text_cfg=base_dot_cfg)),
    ]
    for b in bridge_values:
        variants.append((f"bridge={b:g}", dict(dot_text_enabled=True, derib_enabled=False,
                                                dot_text_cfg=dict(base_dot_cfg, bridge_sigma_px=b))))
    return variants


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frames", type=Path, help="folder (searched recursively) or a single image")
    parser.add_argument("--expected", default=None, help='expected text, e.g. "4113 08:26 BB 05/26"')
    parser.add_argument("--glob", default=None, help="glob pattern inside the folder (default: debug_save frames)")
    parser.add_argument("--bridge", default="", help="comma-separated bridge_sigma_px values to add as variants")
    parser.add_argument("--csv", type=Path, default=None, help="write one row per frame and variant")
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
    bridges = [float(v) for v in args.bridge.split(",") if v.strip()]
    variants = _variants(engine, bridges)
    expected = _norm(args.expected) if args.expected else None

    print(f"{len(frames)} frames, {len(variants)} variants"
          + (f", expected {args.expected!r}" if args.expected else ""))
    rows = []
    reads = {name: collections.Counter() for name, _ in variants}
    exact = collections.Counter()
    errors = collections.Counter()
    for i, path in enumerate(frames, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  skipped unreadable {path}")
            continue
        for name, settings in variants:
            for attr, value in settings.items():
                setattr(engine, attr, value)
            result = engine.read(frame)
            text = result.get("text", "")
            if result.get("error"):
                errors[name] += 1
            reads[name][text] += 1
            hit = expected is not None and _norm(text) == expected
            exact[name] += hit
            rows.append({"frame": str(path), "variant": name, "text": text,
                         "exact": int(hit) if expected else "", "error": result.get("error", "")})
        if i % 10 == 0:
            print(f"  {i}/{len(frames)} frames done")

    n = len({r["frame"] for r in rows})
    print()
    for name, _ in variants:
        line = f"{name:12s}"
        if expected is not None:
            line += f" exact {exact[name]:4d}/{n} ({100.0 * exact[name] / max(1, n):5.1f}%)"
        if errors[name]:
            line += f"  engine errors {errors[name]}"
        print(line)
        for text, count in reads[name].most_common(5):
            print(f"      {count:4d}x  {text!r}")

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["frame", "variant", "text", "exact", "error"])
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
