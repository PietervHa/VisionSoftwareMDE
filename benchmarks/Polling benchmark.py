"""
Polling_benchmark.py
====================
Measures the impact of the polling bottleneck and MJPEG thread
competition described in the performance analysis.

Starts a real Flask instance (web.py) with a mock camera and a mock
AppState that simulates a live vision backend updating results every
~150 ms. Then runs four test scenarios back-to-back:

  1. Interval mode,  no stream  -- current setInterval behaviour
  2. Interval mode,  with stream -- current behaviour + MJPEG competing
  3. Chained mode,   no stream  -- fixed setTimeout-chain behaviour
  4. Chained mode,   with stream -- fixed behaviour + MJPEG still running

Metrics per scenario
--------------------
  response_latency_ms   -- how long each GET /result took end-to-end
  concurrent_peak       -- max requests in-flight at the same time
  result_age_ms         -- how old the data was when it arrived
                           (backend embeds a timestamp; age = now - that)
  stale_rate_pct        -- % of responses identical to the previous one
                           (caused by requests piling up and all returning
                            the same cached snapshot)
  requests_fired        -- total HTTP requests sent
  unique_results        -- distinct result snapshots actually received

"""

import argparse
import http.client
import json
import math
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup – allow running from benchmarks/ or project root
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve().parent
_root = _here if (_here / "web.py").exists() else _here.parent
_default_benchmark_output_dir = _root / "benchmarks" / "benchmark_results"
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from web import create_app


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------

class MockCamera:
    """Returns a blank 640x480 frame; no real hardware needed."""

    def __init__(self):
        self._frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.app_state = None  # not used by web.py video_feed path

    def get_frame(self):
        return self._frame.copy()

    def release(self):
        pass


class MockAppState:
    """
    Simulates a running vision backend.

    A background thread updates latest_result every `cycle_ms` milliseconds,
    embedding a result_timestamp so the benchmark can compute data age on
    the receiving side.
    """

    def __init__(self, cycle_ms=150):
        self._lock = threading.Lock()
        self._cycle_ms = cycle_ms
        self._result = self._fresh_result()
        self._counters = {"ok": 0, "nok": 0, "total": 0}
        self._maintenance = True
        self._vision_mode = "ocr"
        self._threshold = 0.8
        self._ocr_keyword = "benchmark"
        self._camera_rotation = 0
        self._running = True
        self._update_thread = threading.Thread(
            target=self._update_loop, daemon=True
        )
        self._update_thread.start()

    # -- Internal helpers ---------------------------------------------------

    def _fresh_result(self):
        return {
            "detections": [{"text": "BENCHMARK", "confidence": 0.95}],
            "processing_time_ms": 120,
            "status": "OK",
            "cycle_time_ms": 148,
            "confidence_threshold": 0.8,
            # Embed wall-clock timestamp so the poller can measure data age.
            "result_timestamp": time.perf_counter(),
            "sequence": 0,
        }

    def _update_loop(self):
        seq = 0
        while self._running:
            time.sleep(self._cycle_ms / 1000.0)
            seq += 1
            with self._lock:
                self._result = {
                    **self._fresh_result(),
                    "sequence": seq,
                    "result_timestamp": time.perf_counter(),
                }
                self._counters["ok"] += 1
                self._counters["total"] += 1

    def stop(self):
        self._running = False

    # -- AppState interface (must match what web.py calls) ------------------

    def get_snapshot(self):
        with self._lock:
            return {
                "result": dict(self._result),
                "counters": dict(self._counters),
                "maintenance_mode": self._maintenance,
                "vision_mode": self._vision_mode,
                "ocr_keyword": self._ocr_keyword,
            }

    def get_vision_mode(self):
        with self._lock:
            return self._vision_mode

    def set_vision_mode(self, value):
        with self._lock:
            if value in ("ocr", "object_detection"):
                self._vision_mode = value

    def get_maintenance_mode(self):
        with self._lock:
            return self._maintenance

    def set_maintenance_mode(self, value):
        with self._lock:
            self._maintenance = bool(value)

    def get_threshold(self):
        with self._lock:
            return self._threshold

    def set_threshold(self, value):
        with self._lock:
            self._threshold = max(0.0, min(1.0, float(value)))

    def get_camera_rotation(self):
        with self._lock:
            return self._camera_rotation

    def rotate_camera(self):
        with self._lock:
            self._camera_rotation = (self._camera_rotation + 1) % 4

    def get_ocr_keyword(self):
        with self._lock:
            return self._ocr_keyword

    def set_ocr_keyword(self, value):
        with self._lock:
            self._ocr_keyword = value.strip().lower()

    def reset_counters(self):
        with self._lock:
            self._counters = {"ok": 0, "nok": 0, "total": 0}

    def update_result(self, result):
        with self._lock:
            self._result = result

    def increment_counter(self, status):
        with self._lock:
            self._counters["total"] += 1
            if status == "OK":
                self._counters["ok"] += 1
            elif status == "NOK":
                self._counters["nok"] += 1


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

def _find_free_port():
    import socket
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _start_flask(app, host, port):
    """Start Flask in a daemon thread. Returns when the port is ready."""
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, threaded=True, use_reloader=False),
        daemon=True,
    )
    t.start()
    # Poll until the port accepts connections.
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        try:
            c = http.client.HTTPConnection(host, port, timeout=0.2)
            c.request("GET", "/status")
            c.getresponse()
            c.close()
            return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError(f"Flask server did not start on {host}:{port} within 5 s")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_json(host, port, path="/result", timeout=5.0):
    """GET a JSON endpoint. Returns (latency_ms, parsed_body)."""
    t0 = time.perf_counter()
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return latency_ms, json.loads(body)


def _consume_mjpeg_stream(host, port, stop_event):
    """
    Reads the MJPEG /video_feed stream continuously until stop_event is set.
    Simulates the <img> tag in the browser HMI.
    """
    try:
        conn = http.client.HTTPConnection(host, port, timeout=60)
        conn.request("GET", "/video_feed")
        resp = conn.getresponse()
        buf = b""
        while not stop_event.is_set():
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            # Keep buffer from growing indefinitely; we just want to consume.
            if len(buf) > 1_000_000:
                buf = buf[-100_000:]
        conn.close()
    except Exception:
        pass  # Stream closed when server shuts down; that's fine.


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

class PollingBenchmark:

    def __init__(self, duration_s=30, poll_interval_ms=100, host="127.0.0.1"):
        self.duration_s = duration_s
        self.poll_interval_ms = poll_interval_ms
        self.host = host
        self.port = _find_free_port()

        self._camera = MockCamera()
        self._state = MockAppState(cycle_ms=150)

        app = create_app(self._camera, self._state)
        _start_flask(app, self.host, self.port)

    # -- Polling strategies -------------------------------------------------

    def _run_interval_mode(self, with_stream=False):
        """
        Simulates the current setInterval behaviour:
        fires a new request every poll_interval_ms regardless of whether
        the previous one has finished.
        """
        results = []
        lock = threading.Lock()
        inflight = [0]
        inflight_peak = [0]
        stop = threading.Event()

        def _do_request():
            request_time = time.perf_counter()
            with lock:
                inflight[0] += 1
                if inflight[0] > inflight_peak[0]:
                    inflight_peak[0] = inflight[0]
            try:
                latency_ms, body = _get_json(self.host, self.port)
                result_ts = body["result"].get("result_timestamp", request_time)
                age_ms = (request_time - result_ts) * 1000.0
                seq = body["result"].get("sequence", -1)
                with lock:
                    results.append({
                        "latency_ms": round(latency_ms, 2),
                        "age_ms": round(max(age_ms, 0.0), 2),
                        "sequence": seq,
                    })
            except Exception as exc:
                with lock:
                    results.append({"error": str(exc)})
            finally:
                with lock:
                    inflight[0] = max(0, inflight[0] - 1)

        stream_stop = threading.Event()
        if with_stream:
            threading.Thread(
                target=_consume_mjpeg_stream,
                args=(self.host, self.port, stream_stop),
                daemon=True,
            ).start()

        deadline = time.perf_counter() + self.duration_s
        while time.perf_counter() < deadline:
            threading.Thread(target=_do_request, daemon=True).start()
            time.sleep(self.poll_interval_ms / 1000.0)

        # Wait for in-flight requests to land (up to 3 s).
        flush = time.perf_counter() + 3.0
        while time.perf_counter() < flush:
            with lock:
                if inflight[0] == 0:
                    break
            time.sleep(0.02)

        stream_stop.set()
        return results, inflight_peak[0]

    def _run_chained_mode(self, with_stream=False):
        """
        Simulates the fixed recursive setTimeout chain:
        the next request only fires after the previous one completes.
        """
        results = []
        stop_at = time.perf_counter() + self.duration_s

        stream_stop = threading.Event()
        if with_stream:
            threading.Thread(
                target=_consume_mjpeg_stream,
                args=(self.host, self.port, stream_stop),
                daemon=True,
            ).start()

        while time.perf_counter() < stop_at:
            request_time = time.perf_counter()
            try:
                latency_ms, body = _get_json(self.host, self.port)
                result_ts = body["result"].get("result_timestamp", request_time)
                age_ms = (request_time - result_ts) * 1000.0
                seq = body["result"].get("sequence", -1)
                results.append({
                    "latency_ms": round(latency_ms, 2),
                    "age_ms": round(max(age_ms, 0.0), 2),
                    "sequence": seq,
                })
            except Exception as exc:
                results.append({"error": str(exc)})

            remaining = stop_at - time.perf_counter()
            if remaining <= 0:
                break
            # Wait poll_interval_ms before next request (or until deadline).
            time.sleep(min(self.poll_interval_ms / 1000.0, remaining))

        stream_stop.set()
        return results, 1  # chained mode is always exactly 1 in-flight

    # -- Statistics ---------------------------------------------------------

    @staticmethod
    def _compute_stats(results, inflight_peak):
        valid = [r for r in results if "error" not in r]
        errors = len(results) - len(valid)

        if not valid:
            return {"error": "no valid results"}

        latencies = [r["latency_ms"] for r in valid]
        ages = [r["age_ms"] for r in valid]
        seqs = [r["sequence"] for r in valid]

        # Stale = response has the same sequence number as the previous one.
        stale = sum(
            1 for i in range(1, len(seqs)) if seqs[i] == seqs[i - 1]
        )
        stale_pct = round(stale / max(len(seqs) - 1, 1) * 100, 1)
        unique_seqs = len(set(seqs))

        def _s(data):
            return {
                "min_ms":    round(min(data), 2),
                "max_ms":    round(max(data), 2),
                "mean_ms":   round(statistics.mean(data), 2),
                "median_ms": round(statistics.median(data), 2),
                "stdev_ms":  round(statistics.stdev(data), 2) if len(data) > 1 else 0,
                "p95_ms":    round(sorted(data)[math.floor(len(data) * 0.95)], 2),
            }

        return {
            "requests_fired":    len(results),
            "valid_responses":   len(valid),
            "errors":            errors,
            "unique_results":    unique_seqs,
            "stale_rate_pct":    stale_pct,
            "concurrent_peak":   inflight_peak,
            "response_latency":  _s(latencies),
            "result_age":        _s(ages),
        }

    # -- Public run ---------------------------------------------------------

    def run_all(self):
        scenarios = [
            ("interval_no_stream",   self._run_interval_mode, False),
            ("interval_with_stream", self._run_interval_mode, True),
            ("chained_no_stream",    self._run_chained_mode,  False),
            ("chained_with_stream",  self._run_chained_mode,  True),
        ]

        all_stats = {}
        for name, fn, stream in scenarios:
            label = f"{name.replace('_', ' ')}"
            print(f"  Running: {label} ({self.duration_s}s)...", end="", flush=True)
            raw, peak = fn(with_stream=stream)
            stats = self._compute_stats(raw, peak)
            all_stats[name] = stats
            print(f"  done ({stats.get('valid_responses', 0)} responses)")

        self._state.stop()
        return all_stats

    # -- Output -------------------------------------------------------------

    @staticmethod
    def print_summary(all_stats):
        SEP = "=" * 70

        def _row(label, key, sub, unit="ms"):
            vals = []
            for name in ["interval_no_stream", "interval_with_stream",
                         "chained_no_stream", "chained_with_stream"]:
                s = all_stats.get(name, {})
                v = s.get(key, {}).get(sub, s.get(key, "?")) if sub else s.get(key, "?")
                vals.append(str(v) + (f" {unit}" if unit else ""))
            print(f"  {label:<32} {vals[0]:<18} {vals[1]:<18} {vals[2]:<18} {vals[3]}")

        print(f"\n{SEP}")
        print("POLLING BOTTLENECK BENCHMARK SUMMARY")
        print(SEP)
        print(f"  {'':32} {'[B] Interval':18} {'[B]+stream':18} {'[F] Chained':18} {'[F]+stream'}")
        print(f"  {'':32} {'(current)':18} {'(current+B2)':18} {'(fixed)':18} {'(fixed+B2)'}")
        print(f"  {'-'*32} {'-'*17} {'-'*17} {'-'*17} {'-'*17}")
        print("  THROUGHPUT")
        _row("Requests fired",      "requests_fired",    None, "")
        _row("Valid responses",     "valid_responses",   None, "")
        _row("Unique results seen", "unique_results",    None, "")
        _row("Stale response rate", "stale_rate_pct",    None, "%")
        _row("Concurrent peak",     "concurrent_peak",   None, "req")
        print("  RESPONSE LATENCY")
        _row("  Mean",              "response_latency",  "mean_ms")
        _row("  Median",            "response_latency",  "median_ms")
        _row("  p95",               "response_latency",  "p95_ms")
        _row("  StdDev",            "response_latency",  "stdev_ms")
        print("  RESULT AGE (data staleness)")
        _row("  Mean",              "result_age",        "mean_ms")
        _row("  Median",            "result_age",        "median_ms")
        _row("  p95",               "result_age",        "p95_ms")
        print(SEP)
        print("  [B] = Before fix (current code)   [F] = After fix (recommended)")
        print(SEP)

    @staticmethod
    def save_results(all_stats, output_dir=_default_benchmark_output_dir, duration_s=30, interval_ms=100):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = path / f"polling_benchmark_{ts}.json"
        report = {
            "timestamp": datetime.now().isoformat(),
            "benchmark_duration_seconds_per_scenario": duration_s,
            "poll_interval_ms": interval_ms,
            "description": (
                "Compares setInterval (current) vs setTimeout-chain (fixed) polling "
                "with and without a competing MJPEG /video_feed stream."
            ),
            "column_key": {
                "interval_no_stream":   "Current code, no video stream",
                "interval_with_stream": "Current code + MJPEG stream competing",
                "chained_no_stream":    "Fixed code, no video stream",
                "chained_with_stream":  "Fixed code + MJPEG stream competing",
            },
            "statistics": all_stats,
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Results saved to: {filename}")
        return filename


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Polling bottleneck benchmark")
    p.add_argument("--duration",    type=int,   default=30,
                   help="Seconds per scenario (default 30)")
    p.add_argument("--interval",    type=int,   default=100,
                   help="Poll interval in ms (default 100, OCR mode)")
    p.add_argument("--output-dir",  default=str(_default_benchmark_output_dir),
                   help="Directory for JSON output (default benchmarks/benchmark_results)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("POLLING BOTTLENECK BENCHMARK")
    print("=" * 70)
    print(f"  Duration per scenario : {args.duration} s")
    print(f"  Poll interval         : {args.interval} ms")
    print(f"  Scenarios             : 4 (interval/chained x no-stream/stream)")
    print(f"  Total runtime         : ~{args.duration * 4} s")
    print()
    print("  Starting Flask server with mock camera...", end="", flush=True)

    bench = PollingBenchmark(
        duration_s=args.duration,
        poll_interval_ms=args.interval,
    )
    print("  ready.\n")

    all_stats = bench.run_all()

    PollingBenchmark.print_summary(all_stats)
    PollingBenchmark.save_results(
        all_stats,
        output_dir=args.output_dir,
        duration_s=args.duration,
        interval_ms=args.interval,
    )


if __name__ == "__main__":
    main()

