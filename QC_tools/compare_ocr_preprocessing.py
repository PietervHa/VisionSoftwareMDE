"""
Compare OCRead settings on saved camera frames with the REAL PaddleOCR engine
and the real OCRead code path (locate -> deskew -> preprocess -> recognise ->
orientation retry -> output filter -> read validation).

Pipelines:
  previous        derib + CLAHE, no validation (configuration before 2026-10)
  dot_text_raw    dot_text, no validation (any text = OK)
  configured      read validation + consensus exactly as in config/default.yaml
  bridge=X        "configured" with dot_text bridge_sigma_px=X (--bridge)
  single:<v>      read validation on ONE read with preprocessing variant <v>,
                  for every variant this version has - shows which
                  preprocessing your PaddleOCR actually reads correctly

With --expected, "WRONG OK" is the number that matters in production: a code
reported OK that is not what is printed. A reject is NOK with a reason.

Frames: set ocr.dynamic_roi.debug_save: true, run a session, then run this.
Every debug cycle folder holds text_locator_search.png, the raw frame OCR saw.
The folder defaults to the configured debug_dir (also checks backend/<debug_dir>,
where older captures ended up). Select one bottle with --since / --until
(time of day from the cycle folder names, YYYYMMDD_HHMMSS_mmm).

Usage (from the repo root):
    python -m QC_tools.compare_ocr_preprocessing --since 10:54:14 --until 10:56:17 --expected "4113 08:26 BB 05/26"
    python -m QC_tools.compare_ocr_preprocessing some/folder --glob "*.png" --csv out.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np


def _norm(text) -> str:
    return "".join(str(text or "").split())


def _resolve_frames_root(given):
    from backend.core.config_loader import cfg
    from backend.detection.ocr.paddle_ocr import PROJECT_ROOT, resolve_project_path

    configured = (cfg.get("ocr", {}).get("dynamic_roi", {}) or {}).get("debug_dir") or "data/debug/dynamic_roi"
    rel = Path(given) if given is not None else Path(configured)
    candidates = [rel] if rel.is_absolute() else [Path.cwd() / rel, Path(resolve_project_path(rel)),
                                                  PROJECT_ROOT / "backend" / rel]
    for c in candidates:
        if c.exists():
            return c
    print(f"Folder not found: {rel}")
    print("Looked in:\n  " + "\n  ".join(str(c) for c in candidates))
    return None


def _find_frames(root: Path, pattern: str | None) -> list[Path]:
    if root.is_file():
        return [root]
    if pattern:
        return sorted(root.glob(pattern))
    frames = sorted(root.glob("**/text_locator_search.png"))
    if frames:
        return frames
    return sorted([*root.glob("**/*.png"), *root.glob("**/*.jpg"), *root.glob("**/*.jpeg")])


def _in_window(path, since, until):
    """Debug cycle folders are named YYYYMMDD_HHMMSS_mmm; filter on that time of day."""
    def norm(s):
        return None if not s else (s if len(s) == 8 else s + ":00")
    for part in Path(path).parts[::-1]:
        m = re.match(r"\d{8}_(\d{2})(\d{2})(\d{2})", part)
        if m:
            t = ":".join(m.groups())
            lo = norm(since)
            hi = None if not until else (until if len(until) == 8 else until + ":59")
            return (lo is None or t >= lo) and (hi is None or t <= hi)
    return False


def _pipelines(engine, bridge_values):
    dot_cfg = dict(engine.dot_text_cfg or {})
    v = engine.validator
    p = [
        ("previous", dict(validator=None, consensus_variants=[None], consensus_min_agree=1,
                          dot_text_enabled=False, derib_enabled=True, dot_text_cfg=dot_cfg)),
        ("dot_text_raw", dict(validator=None, consensus_variants=[None], consensus_min_agree=1,
                              dot_text_enabled=True, derib_enabled=False, dot_text_cfg=dot_cfg)),
    ]
    if v is None:
        print("Note: ocr.read_validation is disabled in the config - validated pipelines skipped.")
        return p
    configured = dict(validator=v, consensus_variants=list(engine.consensus_variants),
                      consensus_min_agree=engine.consensus_min_agree, dot_text_enabled=engine.dot_text_enabled,
                      derib_enabled=engine.derib_enabled, dot_text_cfg=dot_cfg)
    p.append(("configured", configured))
    for b in bridge_values:
        p.append((f"bridge={b:g}", dict(configured, dot_text_cfg=dict(dot_cfg, bridge_sigma_px=b))))
    for variant in sorted(engine.PREPROCESS_VARIANTS):
        p.append((f"single:{variant}", dict(validator=v, consensus_variants=[variant], consensus_min_agree=1,
                                            dot_text_enabled=True, derib_enabled=False, dot_text_cfg=dot_cfg)))
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=Path, nargs="?", default=None,
                    help="folder (searched recursively) or one image; default: the configured debug_dir")
    ap.add_argument("--expected", default=None, help='expected code, e.g. "4113 08:26 BB 05/26"')
    ap.add_argument("--since", help="only cycles captured from HH:MM[:SS]")
    ap.add_argument("--until", help="only cycles captured up to HH:MM[:SS]")
    ap.add_argument("--glob", default=None, help="glob pattern inside the folder (default: debug_save frames)")
    ap.add_argument("--bridge", default="", help="comma-separated bridge_sigma_px values to add")
    ap.add_argument("--csv", type=Path, default=None, help="write one row per frame and pipeline")
    ap.add_argument("--limit", type=int, default=0, help="only use the first N frames")
    ap.add_argument("--config", default=None, help="config YAML (read by backend.core.config_loader)")
    args = ap.parse_args(argv)

    root = _resolve_frames_root(args.frames)
    if root is None:
        return 1
    frames = _find_frames(root, args.glob)
    if args.since or args.until:
        frames = [f for f in frames if _in_window(f, args.since, args.until)]
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        print(f"No frames found under {root}" + (f" between {args.since} and {args.until}" if args.since or args.until else ""))
        return 1
    print(f"Reading {len(frames)} frames from {root}")

    from backend.detection.ocr.paddle_ocr import PaddleOCR

    engine = PaddleOCR(app_state=None)   # starts the real PaddleOCR worker
    engine.dynamic_roi_debug_dir = None  # don't write debug images while comparing
    pipelines = _pipelines(engine, [float(x) for x in args.bridge.split(",") if x.strip()])
    expected = _norm(args.expected) if args.expected else None

    stats = {n: collections.Counter() for n, _ in pipelines}
    reasons = {n: collections.Counter() for n, _ in pipelines}
    reads = {n: collections.Counter() for n, _ in pipelines}
    conf = {n: [] for n, _ in pipelines}
    ms = {n: [] for n, _ in pipelines}
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
            val = r.get("validation")
            if r.get("error"):
                ok, reason = False, "ocr_engine_error"
            elif val is not None:
                ok, reason = bool(val["accepted"]), val.get("reason")
            else:
                ok = bool(_norm(r.get("text"))); reason = None if ok else "no_text_detected"
            text = r.get("text", "")
            reads[name][("OK  " if ok else "NOK ") + text] += 1
            if ok:
                stats[name]["correct_ok" if expected is None or _norm(text) == expected else "WRONG_OK"] += 1
                if val:
                    conf[name] += [x["min_token_confidence"] for x in val.get("reads", [])
                                   if x.get("accepted") and x.get("text") == text and x.get("min_token_confidence") is not None]
            else:
                stats[name]["rejected"] += 1; reasons[name][reason] += 1
            rows.append({"frame": str(path), "pipeline": name, "status": "OK" if ok else "NOK", "text": text,
                         "reason": reason or "", "correct": "" if expected is None else int(ok and _norm(text) == expected)})
        if i % 5 == 0:
            print(f"  {i}/{len(frames)} frames done")

    n = len({r["frame"] for r in rows})
    def line(name):
        s = stats[name]
        head = (f"correct OK {s['correct_ok']:3d}   WRONG OK {s['WRONG_OK']:3d}   rejected {s['rejected']:3d}"
                if expected is not None else f"OK {s['correct_ok']:3d}   NOK {s['rejected']:3d}")
        return f"{name:24s} {head}   (of {n})   ocr {np.mean(ms[name]):.0f} ms/frame"
    print()
    for name, _ in pipelines:
        if name.startswith("single:"):
            continue
        print(line(name))
        if reasons[name]:
            print("      reject reasons: " + ", ".join(f"{k}={v}" for k, v in reasons[name].most_common()))
        if conf[name]:
            print(f"      min token confidence of accepted reads: lowest {min(conf[name]):.3f}, "
                  f"5th percentile {np.percentile(conf[name], 5):.3f}")
        for text, count in reads[name].most_common(3):
            print(f"      {count:4d}x  {text!r}")
    singles = [nm for nm, _ in pipelines if nm.startswith("single:")]
    if singles:
        print("\nEach preprocessing variant on its own (one validated read):")
        for name in sorted(singles, key=lambda nm: (-stats[nm]["correct_ok"], stats[nm]["WRONG_OK"])):
            print("  " + line(name))
            top = [t for t, _ in reads[name].most_common(2)]
            print("        most common: " + " | ".join(repr(t) for t in top))
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["frame", "pipeline", "status", "text", "reason", "correct"])
            w.writeheader(); w.writerows(rows)
        print(f"\nPer-frame results written to {args.csv}")
    try:
        engine._paddle.shutdown()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
