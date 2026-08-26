"""
PaddleOCR inference worker process.

Runs the PaddlePaddle/PaddleOCR native runtime in a dedicated OS process,
fully isolated from PyTorch's native runtime in the main process. This is
what actually eliminates the OpenMP/MKL thread-pool contention that made
the classifier run slower after OCR had been used — the two runtimes no
longer share a process, so there's nothing left to contend over.

The main process only ever sends a preprocessed numpy frame across and gets
back a plain list of {"text", "confidence"} dicts - the parsing from
PaddleOCR's own result object happens inside this worker process, not in
paddle_ocr.py. This matters more under PaddleOCR 3.x than it used to:
.predict() returns OCRResult objects (dict subclasses carrying detection
polygons, model settings, and other inference-internal state alongside the
text/scores), and there's no guarantee everything bundled in there is safe
or cheap to pickle across the multiprocessing.Queue boundary. Reducing to
plain str/float before it ever touches the queue sidesteps that question
entirely and keeps the IPC payload small regardless of what future
PaddleOCR versions decide to stuff into their result objects.
"""
from __future__ import annotations
import multiprocessing as mp


def _parse_ocr_result(results) -> list:
    """Flattens a PaddleOCR 3.x .predict() return value into plain
    {"text", "confidence"} dicts.

    .predict() returns a list of OCRResult objects, one per input
    page/frame (always one here, since we only ever pass a single frame).
    OCRResult is a dict subclass (paddlex.inference.common.result.
    base_result.BaseResult) exposing rec_texts (list[str]) and rec_scores
    (list[float]) directly via subscript access - confirmed by reading the
    installed paddleocr/paddlex source directly rather than assumed, since
    getting this wrong fails silently (empty results, no exception) rather
    than loudly.
    """
    candidates = []
    if not results:
        return candidates
    for page in results:
        try:
            texts = page["rec_texts"]
            scores = page["rec_scores"]
        except (KeyError, TypeError):
            continue
        for text, score in zip(texts, scores):
            if not text or not str(text).strip():
                continue
            candidates.append({"text": str(text), "confidence": round(float(score), 3)})
    return candidates


def _worker_loop(request_q: mp.Queue, response_q: mp.Queue, lang: str, cpu_threads: int, use_angle_cls: bool) -> None:
    import os
    # PaddleOCR 3.x moved MKL-DNN control to the enable_mkldnn constructor
    # argument (set explicitly below) rather than reading the old
    # FLAGS_use_mkldnn env var this app relied on under 2.x. The env var is
    # left set too (harmless either way), but enable_mkldnn=False on the
    # PaddleOCR(...) call below is what actually disables it now - see
    # _common_args.py / prepare_common_init_args in the installed
    # paddleocr package if this ever needs re-checking against a future
    # version.
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("PADDLE_DISABLE_FAST_MATH", "1")
    import logging
    logging.getLogger("ppocr").setLevel(logging.ERROR)

    import numpy as np
    from paddleocr import PaddleOCR as _PaddleOCR

    # use_textline_orientation (called use_angle_cls in 2.x, still accepted
    # as a deprecated alias but renamed here to avoid the deprecation
    # warning firing on every worker start) loads and runs PaddleOCR's
    # text-direction classifier, so characters that are rotated/
    # upside-down relative to the frame (common with dot-print/
    # pin-stamped markings) get rotated upright before the recognition
    # model sees them, instead of being fed in sideways.
    #
    # use_doc_orientation_classify / use_doc_unwarping are both off - both
    # exist for scanned document pages (whole-page rotation / dewarping),
    # not relevant to an already-cropped product-cap image, and turning
    # them off avoids loading two extra models and running two extra
    # inference passes every single cycle for nothing.
    ocr = _PaddleOCR(
        lang=lang,
        cpu_threads=cpu_threads,
        use_textline_orientation=use_angle_cls,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        enable_mkldnn=False,
    )

    # Pay Paddle's own cold-start cost here, once, at worker startup.
    try:
        ocr.predict(np.zeros((64, 64, 3), dtype=np.uint8))
    except Exception:
        pass

    response_q.put(("__ready__", None))

    while True:
        try:
            job_id, frame = request_q.get()
        except (EOFError, OSError):
            break
        if job_id is None:  # shutdown sentinel
            break
        try:
            result = ocr.predict(frame)
            candidates = _parse_ocr_result(result)
            response_q.put((job_id, candidates))
        except Exception as exc:
            response_q.put((job_id, {"__error__": str(exc)}))


class PaddleOCRWorker:
    """Main-process handle to the PaddleOCR subprocess.

    .ocr(frame) returns a plain list of {"text", "confidence"} dicts,
    already parsed inside the worker process (see _parse_ocr_result) -
    paddle_ocr.py consumes this directly and doesn't need to know which
    PaddleOCR version, or which raw result-object shape, produced it."""

    def __init__(self, lang: str = "en", cpu_threads: int = 4, use_angle_cls: bool = True, ready_timeout: float = 60.0):
        ctx = mp.get_context("spawn")
        self._request_q = ctx.Queue()
        self._response_q = ctx.Queue()
        self._process = ctx.Process(
            target=_worker_loop,
            args=(self._request_q, self._response_q, lang, cpu_threads, use_angle_cls),
            daemon=True,
        )
        self._process.start()
        self._job_counter = 0
        job_id, _ = self._response_q.get(timeout=ready_timeout)
        if job_id != "__ready__":
            raise RuntimeError("PaddleOCR worker failed to start correctly.")

    def ocr(self, frame, timeout: float = 30.0):
        self._job_counter += 1
        job_id = self._job_counter
        self._request_q.put((job_id, frame))
        while True:
            resp_id, payload = self._response_q.get(timeout=timeout)
            if resp_id != job_id:
                continue  # stale response from a previous timed-out call
            if isinstance(payload, dict) and "__error__" in payload:
                raise RuntimeError(f"PaddleOCR worker error: {payload['__error__']}")
            return payload

    def shutdown(self) -> None:
        try:
            self._request_q.put((None, None))
        except Exception:
            pass
        self._process.join(timeout=5)