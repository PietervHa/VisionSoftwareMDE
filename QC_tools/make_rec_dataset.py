"""
Training data for fine-tuning PaddleOCR's text recognizer on this printer's
dot print (docs/finetune_ocread.md). Run on the laptop, upload the zip to the
Colab notebook (QC_tools/finetune_paddleocr_rec.ipynb).

Each sample is ONE text line, as PaddleOCR's detector would hand it to the
recognizer inside OCRead:
  - random codes in the template layout ("dddd dd:dd" / "BB dd/dd", spaces
    printed or not), so every digit appears in every position
  - 5x7 dot font with varied dot size, spacing, slant, missing dots, smear
  - blue-black ink on real cap backgrounds cut from your debug captures
  - the rib profile measured on your caps laid through the print (0-2 ribs)
  - the neighbouring line sometimes touching / overlapping (5307 bottle)
  - moulded marks (PE/PA, PP, HDPE, recycling triangles) and rib edges right
    beside the code, still labelled with only the code - so the model learns
    to look past them; plus some embossing-only lines labelled with their
    letters, so a separately detected "PE/PA" is read as letters (and
    ignored by the format check) instead of being turned into digits
  - then exactly OCRead's preprocessing (dot_text divide / subtract, or the
    classic CLAHE path), so the model trains on what it will really get

Output: <out>/images/*.png, train.txt, val.txt (PaddleOCR "path<TAB>label"),
dict.txt, and <out>.zip ready for upload.

Usage (from the repo root):
    python -m QC_tools.make_rec_dataset                 (20000 train / 2000 val)
    python -m QC_tools.make_rec_dataset --train 30000 --val 3000 --out data/ocread_rec_dataset
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

FONT = {
    "0": [[".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."], [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."]],
    "1": [["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."], ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", "..#.."]],
    "2": [[".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"]],
    "3": [["#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."], [".###.", "#...#", "....#", "..##.", "....#", "#...#", ".###."]],
    "4": [["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."], ["#..#.", "#..#.", "#..#.", "#####", "...#.", "...#.", "...#."]],
    "5": [["#####", "#....", "####.", "....#", "....#", "#...#", ".###."], ["#####", "#....", "#....", "####.", "....#", "#...#", ".###."]],
    "6": [["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."], [".###.", "#....", "#....", "####.", "#...#", "#...#", ".###."]],
    "7": [["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."], ["#####", "....#", "...#.", "..#..", "..#..", "..#..", "..#.."]],
    "8": [[".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."]],
    "B": [["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."]],
    ":": [[".....", ".##..", ".##..", ".....", ".##..", ".##..", "....."]],
    "/": [[".....", "....#", "...#.", "..#..", ".#...", "#....", "....."]],
}
FONT["9"] = [[r[::-1] for r in g[::-1]] for g in FONT["6"]]
# Output alphabet: digits, all capital letters and the punctuation in codes
# and embossing. Letters matter: with digits only, the model turned embossed
# "PE/PA" into "BB/90" and a rib edge into an extra "1" ("08:261").
DICT = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ:/.-"
EMBOSS_TEXTS = ["PE/PA", "PE", "PP", "HDPE", "LDPE", "PET", "PS", "PE-HD", "PP5", "2", "5"]

# Cross-profile of a cap rib, measured on the real frames (brightness factor
# across the rib; bright top at index 34, shadow on one side).
RIB_PROFILE = np.array([
    1.020, 1.020, 1.020, 1.030, 1.030, 1.020, 1.020, 1.020, 1.020, 1.010, 1.010, 1.010, 1.005, 1.000, 1.000,
    0.990, 0.980, 0.980, 0.990, 0.980, 0.980, 0.970, 0.960, 0.950, 0.941, 0.931, 0.911, 0.901, 0.881, 0.851,
    0.891, 0.990, 1.069, 1.119, 1.129, 1.114, 1.089, 1.059, 0.990, 0.931, 0.832, 0.762, 0.743, 0.762, 0.792,
    0.812, 0.832, 0.871, 0.886, 0.891, 0.901, 0.911, 0.931, 0.941, 0.950, 0.955, 0.965, 0.970, 0.980, 0.980,
    0.980, 0.980, 0.980, 0.980, 0.980, 0.990, 0.990, 0.980, 0.980, 0.980], np.float32)
RIB_CENTER = 34


def add_rib(img, point, angle_deg):
    h, w = img.shape[:2]
    a = np.deg2rad(angle_deg)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = (xx - point[0]) * -np.sin(a) + (yy - point[1]) * np.cos(a) + RIB_CENTER
    fac = np.interp(d.ravel(), np.arange(len(RIB_PROFILE), dtype=np.float32), RIB_PROFILE, left=1.0, right=1.0)
    return np.clip(img.astype(np.float32) * fac.reshape(h, w)[..., None], 0, 255).astype(np.uint8)


def emboss(img, x, y, height, what, rng):
    """Draw raised plastic (no ink): light on one side, shadow on the other,
    like the moulded "PE/PA" and recycling marks on the caps. `what` is a
    text or "triangle". Returns the (x0, y0, x1, y1) box it covers."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    if what == "triangle":
        s = height * rng.uniform(1.2, 2.0)
        pts = np.array([[x, y + s], [x + s * 0.55, y], [x + s * 1.1, y + s]], np.int32)
        cv2.polylines(mask, [pts], True, 1.0, max(1, int(height / 7)))
        box = (x, y, x + s * 1.1, y + s)
    else:
        scale = height / 22.0
        thick = max(1, int(round(height / 9)))
        (tw, th), _ = cv2.getTextSize(what, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(mask, what, (int(x), int(y + th)), cv2.FONT_HERSHEY_SIMPLEX, scale, 1.0, thick, cv2.LINE_AA)
        box = (x, y, x + tw, y + th)
    mask = cv2.GaussianBlur(mask, (0, 0), rng.uniform(0.6, 1.4))
    d = max(1, int(round(height / 14)))
    light = np.roll(np.roll(mask, -d, 0), -d, 1); shade = np.roll(np.roll(mask, d, 0), d, 1)
    fac = 1.0 + rng.uniform(0.06, 0.16) * (light - shade) + rng.uniform(0.0, 0.05) * mask
    out = np.clip(img.astype(np.float32) * fac[..., None], 0, 255).astype(np.uint8)
    return out, box


class Generator:
    def __init__(self, frames, seed=0, pool=400):
        from backend.core.config_loader import cfg
        from backend.detection.ocr.dot_text import clean_dot_text
        from backend.detection.ocr.paddle_ocr import PaddleOCR
        self.rng = np.random.default_rng(seed)
        self.clean_dot_text = clean_dot_text
        self.dot_cfg = dict((cfg["ocr"].get("dynamic_roi", {}) or {}).get("dot_text") or {})
        self.classic = PaddleOCR.__new__(PaddleOCR)          # only its preprocessing is used
        self.classic.preprocess_mode = str(cfg["ocr"].get("preprocess", "fast")).lower()
        self.classic._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))   # same as PaddleOCR.__init__
        self.backgrounds = self._background_pool(frames, pool)

    def _background_pool(self, frames, n):
        """Print-free cap patches: away from the real code (located per frame,
        plus a margin) and with almost no ink-like pixels left."""
        from backend.core.config_loader import cfg
        from backend.detection.ocr.text_locator import locate_text_region
        dyn = cfg["ocr"].get("dynamic_roi", {}) or {}
        sources = []
        for f in frames:
            im = cv2.imread(str(f))
            if im is None:
                continue
            loc = locate_text_region(im, dyn)
            if not loc:
                continue                      # print not found -> can't tell where it is: skip frame
            x1, y1, x2, y2 = loc["bbox"]; m = 40
            sources.append((im, (x1 - m, y1 - m, x2 + m, y2 + m)))
        if not sources:
            raise SystemExit("No usable background frames found - capture a session with debug_save: true first.")
        out, tries = [], 0
        while len(out) < n and tries < n * 50:
            tries += 1
            im, (bx1, by1, bx2, by2) = sources[self.rng.integers(len(sources))]
            h, w = int(self.rng.integers(80, 140)), int(self.rng.integers(160, 280))
            if im.shape[0] <= h or im.shape[1] <= w:
                continue
            y, x = int(self.rng.integers(0, im.shape[0] - h)), int(self.rng.integers(0, im.shape[1] - w))
            if not (x + w < bx1 or x > bx2 or y + h < by1 or y > by2):
                continue                      # overlaps the real code
            p = im[y:y + h, x:x + w]
            ink = (255.0 - self.clean_dot_text(p, {"upscale": 1})[..., 0]) / 225.0
            if (ink > 0.5).mean() < 0.02:
                flip = int(self.rng.integers(-1, 3))      # -1/0/1 = mirror, 2 = as is
                out.append(p.copy() if flip == 2 else cv2.flip(p, flip))
        if len(out) < 20:
            raise SystemExit(f"Only {len(out)} print-free background patches found - capture more frames.")
        return out

    def _code_lines(self):
        d = lambda: str(self.rng.integers(10))
        l1 = "".join(d() for _ in range(4)) + (" " if self.rng.random() < 0.9 else "") + d() + d() + ":" + d() + d()
        l2 = "BB" + (" " if self.rng.random() < 0.7 else "") + d() + d() + "/" + d() + d()
        return [l1, l2]

    def sample(self):
        """One rendered 2-line code -> list of (line_image, label) after OCRead preprocessing."""
        rng = self.rng
        ch = rng.uniform(15, 24); py = ch / 6.5; px = py * rng.uniform(0.9, 1.15); adv = 6 * px * rng.uniform(0.95, 1.1)
        slant = rng.uniform(-0.05, 0.25); r = py * rng.uniform(0.33, 0.55)
        gap = ch * (rng.uniform(-0.05, 0.12) if rng.random() < 0.3 else rng.uniform(0.3, 0.6))   # touching lines 30%
        lines = self._code_lines()
        W = int(adv * 11 + 60); H = int(2 * ch + max(gap, 0) + 50)
        img = cv2.resize(self.backgrounds[rng.integers(len(self.backgrounds))], (W, H)).astype(np.float32)
        S = 4; mask = np.zeros((H * S, W * S), np.float32); boxes = []
        x0, y0 = 25 + rng.uniform(0, 8), 22 + rng.uniform(0, 6)
        for li, text in enumerate(lines):
            y = y0 + li * (ch + gap); x = x0
            for c in text:
                if c != " ":
                    glyph = FONT[c][rng.integers(len(FONT[c]))]
                    for rr, row in enumerate(glyph):
                        for kk, on in enumerate(row):
                            if on != "#" or rng.random() < 0.05:
                                continue
                            cy = y + rr * py + rng.normal(0, 0.15); cx = x + kk * px + slant * (6 - rr) * py + rng.normal(0, 0.15)
                            cv2.circle(mask, (int(cx * S), int(cy * S)), max(1, int(r * S * rng.uniform(0.85, 1.15))), rng.uniform(0.7, 1.0), -1)
                x += adv
            boxes.append((text, x0 - 0.5 * px, y - 0.5 * py, x - adv + 5.5 * px + slant * 6 * py, y + 6.5 * py))
        m = cv2.GaussianBlur(cv2.resize(mask, (W, H), interpolation=cv2.INTER_AREA), (0, 0), rng.uniform(0.4, 1.1))
        tint = np.array([0.75, 1.0, 1.0], np.float32) * rng.uniform(0.9, 1.1)
        img = np.clip(img * (1 - rng.uniform(0.45, 0.7) * m[..., None] * tint), 0, 255).astype(np.uint8)
        for _ in range(rng.choice([0, 1, 1, 2])):
            img = add_rib(img, (rng.uniform(0.2, 0.8) * W, rng.uniform(0.3, 0.7) * H), rng.uniform(0, 180))
        if rng.random() < 0.4:                       # rib edge right after / before a line (the "08:261" case)
            text, bx0, by0, bx1, by1 = boxes[int(rng.integers(len(boxes)))]
            xr = (bx1 + rng.uniform(-0.1, 0.6) * adv) if rng.random() < 0.7 else (bx0 - rng.uniform(0.0, 0.6) * adv)
            img = add_rib(img, (xr, (by0 + by1) / 2), rng.uniform(75, 105))
        if rng.random() < 0.35:                      # moulded mark beside a code line, label stays the code
            li = int(rng.integers(len(boxes)))
            text, bx0, by0, bx1, by1 = boxes[li]
            what = "triangle" if rng.random() < 0.3 else EMBOSS_TEXTS[int(rng.integers(len(EMBOSS_TEXTS)))]
            eh = (by1 - by0) * rng.uniform(0.7, 1.3)
            if rng.random() < 0.5:                   # left of the line
                ex = max(2.0, bx0 - eh * (2.5 if what != "triangle" else 2.0) - rng.uniform(0, 0.5) * adv)
                img, ebox = emboss(img, ex, by0 + rng.uniform(-0.3, 0.3) * eh, eh, what, rng)
                boxes[li] = (text, min(bx0, ebox[0]), by0, bx1, by1)
            else:                                    # right of the line
                ex = bx1 + rng.uniform(0.1, 0.6) * adv
                img, ebox = emboss(img, ex, by0 + rng.uniform(-0.3, 0.3) * eh, eh, what, rng)
                boxes[li] = (text, bx0, by0, max(bx1, min(W - 2, ebox[2])), by1)
        img = np.clip(img + rng.normal(0, rng.uniform(1, 4), img.shape), 0, 255).astype(np.uint8)

        extra = []
        if rng.random() < 0.12:                      # embossing on its own -> read as its letters
            what = EMBOSS_TEXTS[int(rng.integers(len(EMBOSS_TEXTS)))]
            eh = rng.uniform(14, 26)
            bg = cv2.resize(self.backgrounds[rng.integers(len(self.backgrounds))], (int(eh * len(what) * 0.9 + 40), int(eh + 30)))
            eimg, ebox = emboss(bg, 18, 14, eh, what, rng)
            extra.append((eimg, what))

        kind = rng.choice(["dot_divide", "dot_subtract", "classic"], p=[0.4, 0.3, 0.3])
        if kind == "classic":
            proc = self.classic._preprocess_image(img)
        else:
            proc = self.clean_dot_text(img, dict(self.dot_cfg, illumination=kind.split("_")[1]))
        if proc.ndim == 2:
            proc = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
        s = proc.shape[1] / img.shape[1]
        out = []
        for text, bx0, by0, bx1, by1 in boxes:                    # a detector-like box around one line
            h = by1 - by0
            a = int(max(0, (bx0 - rng.uniform(0.2, 1.0) * adv) * s)); b = int(min(proc.shape[1], (bx1 + rng.uniform(0.2, 1.0) * adv) * s))
            t = int(max(0, (by0 - rng.uniform(0.05, 0.35) * h) * s)); u = int(min(proc.shape[0], (by1 + rng.uniform(0.05, 0.35) * h) * s))
            if b - a > 8 and u - t > 8:
                out.append((proc[t:u, a:b], text))
        for eimg, label in extra:
            p = self.classic._preprocess_image(eimg) if kind == "classic" else \
                self.clean_dot_text(eimg, dict(self.dot_cfg, illumination=kind.split("_")[1]))
            if p.ndim == 2:
                p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
            out.append((p, label))
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=Path, nargs="?", default=None, help="debug captures folder (default: configured debug_dir)")
    ap.add_argument("--out", type=Path, default=Path("data/ocread_rec_dataset"))
    ap.add_argument("--train", type=int, default=20000, help="training line images")
    ap.add_argument("--val", type=int, default=2000, help="validation line images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None, help="config YAML (read by backend.core.config_loader)")
    args = ap.parse_args(argv)

    from QC_tools.compare_ocr_preprocessing import _find_frames, _resolve_frames_root
    root = _resolve_frames_root(args.frames)
    if root is None:
        return 1
    gen = Generator(_find_frames(root, None), seed=args.seed)
    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "images").mkdir(parents=True)
    (args.out / "dict.txt").write_text("\n".join(DICT) + "\n", encoding="utf-8")
    for split, n in (("train", args.train), ("val", args.val)):
        rows, k = [], 0
        while len(rows) < n:
            for img, text in gen.sample():
                if len(rows) >= n:
                    break
                name = f"images/{split}_{k:06d}.png"; k += 1
                cv2.imwrite(str(args.out / name), img)
                rows.append(f"{name}\t{text}")
            if len(rows) % 2000 < 2:
                print(f"  {split}: {len(rows)}/{n}")
        (args.out / f"{split}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    archive = shutil.make_archive(str(args.out), "zip", root_dir=args.out)
    print(f"Dataset written to {args.out}  ->  upload {archive} to the Colab notebook")
    return 0


if __name__ == "__main__":
    sys.exit(main())
