"""
PaddleOCR inference worker process.

Runs the PaddlePaddle/PaddleOCR native runtime in a dedicated OS process,
fully isolated from PyTorch's native runtime in the main process. This is
what actually eliminates the OpenMP/MKL thread-pool contention that made
the classifier run slower after OCR had been used — the two runtimes no
longer share a process, so there's nothing left to contend over.

The main process only ever sends a preprocessed numpy frame across and gets
back exactly what `PaddleOCR.ocr()` would have returned. All ROI, keyword,
and date-matching logic stays in paddle_ocr.py, unchanged.
"""
from __future__ import annotations
import multiprocessing as mp


def _worker_loop(request_q: mp.Queue, response_q: mp.Queue, lang: str, cpu_threads: int) -> None:
    import os
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("PADDLE_DISABLE_FAST_MATH", "1")
    import logging
    logging.getLogger("ppocr").setLevel(logging.ERROR)

    import numpy as np
    from paddleocr import PaddleOCR as _PaddleOCR

    ocr = _PaddleOCR(lang=lang, cpu_threads=cpu_threads)

    # Pay Paddle's own cold-start cost here, once, at worker startup.
    try:
        ocr.ocr(np.zeros((64, 64, 3), dtype=np.uint8))
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
            result = ocr.ocr(frame)
            response_q.put((job_id, result))
        except Exception as exc:
            response_q.put((job_id, {"__error__": str(exc)}))


class PaddleOCRWorker:
    """Main-process handle to the PaddleOCR subprocess. Drop-in replacement
    for calling `_PaddleOCR(...).ocr(frame)` directly — same `.ocr(frame)`
    call, same return shape, just routed through a separate process."""

    def __init__(self, lang: str = "en", cpu_threads: int = 4, ready_timeout: float = 60.0):
        ctx = mp.get_context("spawn")
        self._request_q = ctx.Queue()
        self._response_q = ctx.Queue()
        self._process = ctx.Process(
            target=_worker_loop,
            args=(self._request_q, self._response_q, lang, cpu_threads),
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