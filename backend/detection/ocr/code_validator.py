"""
backend/detection/ocr/code_validator.py

Turns an OCR read into either a validated code or an explicit rejection -
so the system never forwards a guess as a pass.

A general-purpose recognizer (PaddleOCR) will always return *something*:
"08.26", "B8 05/26", "4113 08:250", "4113 08:2". Before this module every
non-empty read was status OK and went to the PLC as-is. Industrial code
readers instead "field" the read against the known print format and only
report a code that fits it; this does the same.

Template syntax (one string per printed line), e.g.  ["#### ##:##", "BB ##/##"]
  #   a digit                 (information - corrections limited)
  A   an uppercase letter     (information - corrections limited)
  *   any allowed character   (information)
  space   gap between fields (ignored when matching, kept in the output)
  anything else   a literal that must be printed exactly there ("B", ":", "/")

Matching rules - chosen so validation can reject, but never invent:
  - A digit is NEVER changed into another digit (no 5 <-> 6 "fixes").
  - In a digit position, only letters that look like exactly one digit
    are mapped (O->0, I/l->1, Z->2, S->5, G->6, B->8, g->9); each counts as a
    correction and at most max_corrections are allowed per read.
  - Literal positions carry no information, so a known look-alike there is
    normalised for free (":" read as "." or ";", "B" read as "8", "/" read
    as "1"/"l"). Anything else in a literal position is a mismatch.
  - Fields must be made of whole OCR tokens (boxes). Extra tokens around
    them (embossed "PE/PA", a stray ".") may be ignored, up to
    max_ignored_tokens, but text is never cut out of the middle of a token:
    "08:250" does not yield "08:25".
  - The token stream is also tried fully reversed, which is how a cap lying
    upside-down under the camera comes back (each line corrected by the
    textline classifier, but lines and boxes in reverse order).
  - Every token used must have confidence >= min_confidence.
  - Optionally the canonical text must fully match value_pattern (e.g. a
    regex that also checks hours 00-23 and months 01-12).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

DIGIT_LOOKALIKES = {
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "I": "1", "l": "1", "|": "1", "i": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "G": "6", "b": "6",
    "B": "8",
    "g": "9", "q": "9",
}
LETTER_LOOKALIKES = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
PUNCT_LOOKALIKES = {
    ":": set(".;,'=i1lI|!"),
    "/": set("1lI|\\7"),
    ".": set(",:"),
    "-": set("_~="),
}

REJECT_NO_TEXT = "no_text_detected"
REJECT_FORMAT = "format_mismatch"
REJECT_CONFIDENCE = "low_confidence"
REJECT_VALUE = "invalid_value"
REJECT_CONSENSUS = "no_consensus"


def normalize_token(text: str) -> str:
    """NFKC (full-width "：" -> ":"), and no whitespace inside a token."""
    return "".join(unicodedata.normalize("NFKC", str(text)).split())


@dataclass(order=True)
class _Match:
    corrections: int
    ignored: int
    literal_fixes: int
    canonical_lines: list = field(compare=False)
    used_confidences: list = field(compare=False)
    ignored_tokens: list = field(compare=False)
    orientation: str = field(compare=False, default="normal")


class CodeTemplate:
    def __init__(self, lines, value_pattern=None, min_confidence=0.0,
                 max_corrections=1, max_ignored_tokens=2):
        if isinstance(lines, str):
            lines = [lines]
        self.lines = [str(line) for line in lines if str(line).strip()]
        if not self.lines:
            raise ValueError("code template needs at least one line")
        self.patterns = [line.replace(" ", "") for line in self.lines]
        self.value_pattern = re.compile(value_pattern) if value_pattern else None
        self.min_confidence = float(min_confidence)
        self.max_corrections = int(max_corrections)
        self.max_ignored_tokens = int(max_ignored_tokens)

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(
            cfg.get("template") or [],
            value_pattern=cfg.get("value_pattern") or None,
            min_confidence=cfg.get("min_confidence", 0.0),
            max_corrections=cfg.get("max_corrections", 1),
            max_ignored_tokens=cfg.get("max_ignored_tokens", 2),
        )

    # -- character level ---------------------------------------------------

    @staticmethod
    def _match_char(read: str, tmpl: str):
        """Returns (output_char, corrections, literal_fixes) or None."""
        if tmpl == "#":
            if read.isdigit() and read.isascii():
                return read, 0, 0
            if read in DIGIT_LOOKALIKES:
                return DIGIT_LOOKALIKES[read], 1, 0
            return None
        if tmpl == "A":
            if read.isascii() and read.isalpha():
                return read.upper(), 0, 0
            if read in LETTER_LOOKALIKES:
                return LETTER_LOOKALIKES[read], 1, 0
            return None
        if tmpl == "*":
            return (read, 0, 0) if read != "?" else None
        # literal
        if read == tmpl:
            return tmpl, 0, 0
        if tmpl.isalpha() and (read.upper() == tmpl or LETTER_LOOKALIKES.get(read) == tmpl):
            return tmpl, 0, 1
        if tmpl.isdigit() and DIGIT_LOOKALIKES.get(read) == tmpl:
            return tmpl, 0, 1
        if read in PUNCT_LOOKALIKES.get(tmpl, ()):
            return tmpl, 0, 1
        return None

    def _match_line(self, text: str, k: int):
        pattern = self.patterns[k]
        if len(text) != len(pattern):
            return None
        out, corr, lit = [], 0, 0
        for r, t in zip(text, pattern):
            m = self._match_char(r, t)
            if m is None:
                return None
            out.append(m[0]); corr += m[1]; lit += m[2]
        # Re-insert the template's spaces for the canonical output.
        chars = iter(out)
        canonical = "".join(" " if t == " " else next(chars) for t in self.lines[k])
        return canonical, corr, lit

    # -- token stream ----------------------------------------------------------

    def _match_stream(self, tokens, orientation):
        """tokens: list of (text, confidence). Fields = contiguous runs of
        whole tokens, in template order; tokens outside them are ignored."""
        n, best = len(tokens), None

        def search(k, i, acc_lines, corr, lit, confs, used):
            nonlocal best
            if k == len(self.patterns):
                ignored = [tokens[j][0] for j in range(n) if j not in used]
                m = _Match(corr, len(ignored), lit, list(acc_lines), list(confs), ignored, orientation)
                if best is None or m < best:
                    best = m
                return
            target = len(self.patterns[k])
            for s in range(i, n):
                text = ""
                for e in range(s, n):
                    text += tokens[e][0]
                    if len(text) > target:
                        break
                    if len(text) < target:
                        continue
                    m = self._match_line(text, k)
                    if m is None:
                        break
                    span = set(range(s, e + 1))
                    search(k + 1, e + 1, acc_lines + [m[0]], corr + m[1], lit + m[2],
                           confs + [tokens[j][1] for j in span], used | span)
                    break

        search(0, 0, [], 0, 0, [], set())
        return best

    def validate(self, candidates) -> dict:
        tokens = [(normalize_token(c.get("text", "")), float(c.get("confidence", 0.0)))
                  for c in candidates or []]
        tokens = [t for t in tokens if t[0]]
        # For display/logging: the read as the model gave it (spacing kept).
        raw_text = " ".join(
            " ".join(unicodedata.normalize("NFKC", str(c.get("text", ""))).split())
            for c in candidates or [] if normalize_token(c.get("text", ""))
        )
        base = {"raw_text": raw_text, "text": None, "accepted": False}
        if not tokens:
            return dict(base, reason=REJECT_NO_TEXT)

        matches = [m for m in (self._match_stream(tokens, "normal"),
                               self._match_stream(tokens[::-1], "reversed")) if m is not None]
        matches = [m for m in matches if m.corrections <= self.max_corrections
                   and m.ignored <= self.max_ignored_tokens]
        if not matches:
            return dict(base, reason=REJECT_FORMAT)
        best = min(matches)
        canonical = " ".join(best.canonical_lines)
        info = dict(base, text=canonical, corrections=best.corrections,
                    literal_fixes=best.literal_fixes, ignored_tokens=best.ignored_tokens,
                    orientation=best.orientation,
                    min_token_confidence=round(min(best.used_confidences), 3))
        if min(best.used_confidences) < self.min_confidence:
            return dict(info, reason=REJECT_CONFIDENCE)
        if self.value_pattern and not self.value_pattern.fullmatch(canonical):
            return dict(info, reason=REJECT_VALUE)
        return dict(info, accepted=True, reason=None)
