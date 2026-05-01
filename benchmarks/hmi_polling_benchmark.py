"""
hmi_polling_benchmark.py

Measures the frontend polling bottleneck by simulating what the
browser does against the running Flask app.

Two scenarios are tested back-to-back:
  - INTERVAL mode  : fires a new /result request every POLL_MS regardless of
                     whether the previous one has finished (current hmi.js behaviour)
  - CHAINED mode   : waits for each /result response before scheduling the next
                     (behaviour after the setTimeout fix)

Optionally opens the /video_feed MJPEG stream during both runs to simulate
(video stream competing for Flask threads).

Run the app first:  python main.py
Then run this:      python benchmarks/hmi_polling_benchmark.py --duration 30

                    # If your video feed is disabled in config, add:
                    python benchmarks/hmi_polling_benchmark.py --duration 30 --skip-video

Requirements: pip install requests
"""

import argparse
import json
import requests
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

_default_benchmark_output_dir = Path(__file__).resolve().parent / "benchmark_results"


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

class HMIPollingBenchmark:
    """
    Simulates browser-side polling against a live Flask instance and records
    per-request latency, in-flight concurrency, and response-bunching.
    """

    POLL_MS = 100       # mirrors getPollingIntervalMs() for OCR mode
    POLL_S  = POLL_MS / 1000.0

    def __init__(self, base_url: str, duration_seconds: int = 30):
        self.base_url        = base_url.rstrip("/")
        self.duration        = duration_seconds
        self.result_url      = f"{self.base_url}/result"
        self.video_url       = f"{self.base_url}/video_feed"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_result(self, session):
        """Single blocking GET /result. Returns (latency_ms, ok: bool)."""
        t0 = time.perf_counter()
        try:
            r = session.get(self.result_url, timeout=5)
            r.raise_for_status()
            latency_ms = (time.perf_counter() - t0) * 1000
            return round(latency_ms, 2), True
        except Exception:
            latency_ms = (time.perf_counter() - t0) * 1000
            return round(latency_ms, 2), False

    def _open_video_stream(self, stop_event):
        """
        Opens /video_feed in a background thread and reads it continuously
        until stop_event is set. Simulates the <img> MJPEG consumer in the HMI.
        """
        def _reader():
            try:
                with requests.get(self.video_url, stream=True, timeout=None) as resp:
                    for _ in resp.iter_content(chunk_size=4096):
                        if stop_event.is_set():
                            break
            except Exception:
                pass
        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        return t

    def _detect_bunching(self, arrival_times_ms, window_ms=50):
        """
        Count how many responses arrived within `window_ms` of the previous one.
        High bunching = requests resolving in bursts (the freeze symptom).
        """
        if len(arrival_times_ms) < 2:
            return 0
        bunched = 0
        for i in range(1, len(arrival_times_ms)):
            if (arrival_times_ms[i] - arrival_times_ms[i - 1]) < window_ms:
                bunched += 1
        return bunched

    # ------------------------------------------------------------------
    # Scenario A: setInterval (current broken behaviour)
    # ------------------------------------------------------------------

    def run_interval_mode(self, with_video_stream: bool = False):
        """
        Fires a new request every POLL_MS regardless of completion.
        Tracks peak in-flight concurrency and bunching.
        """
        latencies        = []
        arrival_times    = []
        errors           = 0
        lock             = threading.Lock()
        stop_event       = threading.Event()
        peak_inflight    = 0
        inflight_counter = [0]   # mutable int in list for thread access

        if with_video_stream:
            self._open_video_stream(stop_event)
            time.sleep(0.3)   # let stream establish

        session    = requests.Session()
        start_time = time.perf_counter()

        def fire_request():
            nonlocal errors, peak_inflight
            with lock:
                inflight_counter[0] += 1
                if inflight_counter[0] > peak_inflight:
                    peak_inflight = inflight_counter[0]

            latency_ms, ok = self._fetch_result(session)
            arrival_abs = (time.perf_counter() - start_time) * 1000

            with lock:
                inflight_counter[0] -= 1
                latencies.append(latency_ms)
                arrival_times.append(arrival_abs)
                if not ok:
                    errors += 1

        threads = []

        while (time.perf_counter() - start_time) < self.duration:
            t = threading.Thread(target=fire_request, daemon=True)
            t.start()
            threads.append(t)
            time.sleep(self.POLL_S)

        for t in threads:
            t.join(timeout=5)

        stop_event.set()
        session.close()

        bunched = self._detect_bunching(arrival_times)
        return self._build_stats(latencies, errors, bunched, peak_inflight, "interval")

    # ------------------------------------------------------------------
    # Scenario B: recursive setTimeout (fixed behaviour)
    # ------------------------------------------------------------------

    def run_chained_mode(self, with_video_stream: bool = False):
        """
        Waits for each /result response before scheduling the next.
        In-flight count is always 0 or 1.
        """
        latencies     = []
        arrival_times = []
        errors        = 0
        stop_event    = threading.Event()

        if with_video_stream:
            self._open_video_stream(stop_event)
            time.sleep(0.3)

        session    = requests.Session()
        start_time = time.perf_counter()

        while (time.perf_counter() - start_time) < self.duration:
            latency_ms, ok = self._fetch_result(session)
            arrival_abs = (time.perf_counter() - start_time) * 1000

            latencies.append(latency_ms)
            arrival_times.append(arrival_abs)
            if not ok:
                errors += 1

            # wait the remainder of the poll window before the next request
            remaining = self.POLL_S - (latency_ms / 1000)
            if remaining > 0:
                time.sleep(remaining)

        stop_event.set()
        session.close()

        bunched = self._detect_bunching(arrival_times)
        return self._build_stats(latencies, errors, bunched, 1, "chained")

    # ------------------------------------------------------------------
    # Stats builder
    # ------------------------------------------------------------------

    def _build_stats(self, latencies, errors, bunched, peak_inflight, mode):
        if not latencies:
            return {}
        return {
            "mode":                  mode,
            "total_requests":        len(latencies),
            "failed_requests":       errors,
            "peak_inflight":         peak_inflight,
            "bunched_responses":     bunched,
            "bunching_rate_pct":     round(bunched / max(len(latencies) - 1, 1) * 100, 1),
            "latency_ms": {
                "min":    round(min(latencies), 2),
                "max":    round(max(latencies), 2),
                "mean":   round(statistics.mean(latencies), 2),
                "median": round(statistics.median(latencies), 2),
                "stdev":  round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
                "p95":    round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            },
        }

    # ------------------------------------------------------------------
    # Save + print
    # ------------------------------------------------------------------

    def save_results(self, results: dict, output_dir: str = str(_default_benchmark_output_dir)):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = output_path / f"hmi_polling_benchmark_{timestamp}.json"
        report    = {
            "timestamp":        datetime.now().isoformat(),
            "poll_interval_ms": self.POLL_MS,
            "duration_seconds": self.duration,
            "results":          results,
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nResults saved to: {filename}")
        return filename

    def print_comparison(self, results: dict):
        WIDTH = 62

        def section(label):
            print(f"\n{'=' * WIDTH}")
            print(f"  {label}")
            print(f"{'=' * WIDTH}")

        def row(label, a, b, unit="ms", lower_is_better=True):
            try:
                delta = b - a
                pct   = (delta / a * 100) if a else 0
                arrow = "↓" if delta < 0 else "↑"
                good  = (delta < 0) == lower_is_better
                tag   = "BETTER" if good and delta != 0 else ("WORSE" if not good and delta != 0 else "—")
                print(f"  {label:<28} {a:>8.1f}{unit}   {b:>8.1f}{unit}   {arrow}{abs(pct):>5.1f}%  {tag}")
            except Exception:
                print(f"  {label:<28} {'n/a':>9}   {'n/a':>9}")

        for tag, (k_int, k_chain) in [
            ("WITHOUT video stream", ("interval", "chained")),
            ("WITH video stream",    ("interval_video", "chained_video")),
        ]:
            if k_int not in results:
                continue

            a = results[k_int]
            b = results[k_chain]

            section(f"SCENARIO: {tag}")
            print(f"  {'Metric':<28} {'BEFORE':>9}   {'AFTER':>9}   {'Delta':>8}  Result")
            print(f"  {'-' * 58}")

            row("Mean latency",      a["latency_ms"]["mean"],   b["latency_ms"]["mean"])
            row("Median latency",    a["latency_ms"]["median"], b["latency_ms"]["median"])
            row("P95 latency",       a["latency_ms"]["p95"],    b["latency_ms"]["p95"])
            row("Max latency",       a["latency_ms"]["max"],    b["latency_ms"]["max"])
            row("Stdev latency",     a["latency_ms"]["stdev"],  b["latency_ms"]["stdev"])
            row("Peak in-flight",    a["peak_inflight"],        b["peak_inflight"],   unit="", lower_is_better=True)
            row("Bunching rate",     a["bunching_rate_pct"],    b["bunching_rate_pct"], unit="%", lower_is_better=True)
            row("Failed requests",   a["failed_requests"],      b["failed_requests"], unit="", lower_is_better=True)

        print(f"\n{'=' * WIDTH}")
        print("  KEY")
        print(f"  {'Bunching rate':<18} % of responses arriving within 50ms of each")
        print(f"  {'':18} other — high value = bulk freeze symptom")
        print(f"  {'Peak in-flight':<18} max simultaneous open requests to /result")
        print(f"  {'P95 latency':<18} 95th-percentile response time")
        print(f"{'=' * WIDTH}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark HMI polling bottleneck (B1 + B2)")
    parser.add_argument("--host",       default="127.0.0.1",        help="Flask host")
    parser.add_argument("--port",       default=5000,  type=int,    help="Flask port")
    parser.add_argument("--duration",   default=30,    type=int,    help="Seconds per scenario")
    parser.add_argument(
        "--output-dir",
        default=str(_default_benchmark_output_dir),
        help="Output directory (default benchmarks/benchmark_results)",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip the with-video-stream scenarios (use if video feed is disabled in config)"
    )
    return parser.parse_args()


def main():
    args     = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    print(f"\nHMI Polling Benchmark")
    print(f"Target : {base_url}")
    print(f"Duration per scenario : {args.duration}s")
    print(f"Poll interval : 100ms  (OCR mode)")

    # quick connectivity check
    try:
        requests.get(f"{base_url}/status", timeout=3).raise_for_status()
        print("Connection OK\n")
    except Exception as e:
        print(f"\nERROR: Could not reach {base_url}/status — is the app running?\n{e}\n")
        return

    bench   = HMIPollingBenchmark(base_url, duration_seconds=args.duration)
    results = {}

    print("[ 1 / 4 ]  INTERVAL mode  (no video stream)  — simulating current hmi.js ...")
    results["interval"] = bench.run_interval_mode(with_video_stream=False)

    print("[ 2 / 4 ]  CHAINED mode   (no video stream)  — simulating setTimeout fix ...")
    results["chained"]  = bench.run_chained_mode(with_video_stream=False)

    if not args.skip_video:
        print("[ 3 / 4 ]  INTERVAL mode  (with video stream) ...")
        results["interval_video"] = bench.run_interval_mode(with_video_stream=True)

        print("[ 4 / 4 ]  CHAINED mode   (with video stream) ...")
        results["chained_video"]  = bench.run_chained_mode(with_video_stream=True)

    bench.print_comparison(results)
    bench.save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
