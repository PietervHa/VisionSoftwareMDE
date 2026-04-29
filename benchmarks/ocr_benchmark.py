import argparse
from collections import deque
import json
import math
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from backend.core.camera import Camera
from backend.core.config_loader import cfg
from backend.utils.logger import setup_logging, get_logger
from backend.core import vision

log = get_logger(__name__)

class OCRBenchmark:
    """
    Benchmark OCR throughput and per-frame timing.
    Uses the same async callback pattern as main.py.
    """

    def __init__(self, duration_seconds=30, flush_wait_seconds=2.0, max_inflight=1):
        self.duration = duration_seconds
        self.flush_wait_seconds = flush_wait_seconds
        self.max_inflight = max_inflight
        self.results = []
        self.lock = threading.Lock()
        self.running = False
        self.start_time = None
        self.frame_count = 0
        self.completed_count = 0
        self.error_count = 0
        self._inflight = 0
        self._recent = deque(maxlen=10)
        self._recent_proc_sum = 0.0
        self._recent_cycle_sum = 0.0
        self._recent_det_sum = 0
        self._profile_stage_totals = {}
        self._profile_samples = 0

    def _on_vision_result(self, result, trigger_time, frame_num):
        """Callback to handle async OCR results."""
        cycle_time_ms = round((time.perf_counter() - trigger_time) * 1000, 2)
        processing_time_ms = float(result.get("processing_time_ms", 0.0))
        detections = result.get("detections", [])
        detection_count = len(detections)
        error_message = result.get("error")
        profile = result.get("_profile_ms")
        error_to_log = None
        debug_to_log = None

        with self.lock:
            self.results.append(
                {
                    "frame_num": frame_num,
                    "cycle_time_ms": cycle_time_ms,
                    "processing_time_ms": processing_time_ms,
                    "detection_count": detection_count,
                    "error": error_message,
                }
            )
            self.completed_count += 1
            if error_message:
                self.error_count += 1
                error_to_log = (frame_num, error_message)
            self._inflight = max(0, self._inflight - 1)

            if len(self._recent) == self._recent.maxlen:
                old_proc, old_cycle, old_det = self._recent[0]
                self._recent_proc_sum -= old_proc
                self._recent_cycle_sum -= old_cycle
                self._recent_det_sum -= old_det

            self._recent.append((processing_time_ms, cycle_time_ms, detection_count))
            self._recent_proc_sum += processing_time_ms
            self._recent_cycle_sum += cycle_time_ms
            self._recent_det_sum += detection_count

            completed_frames = self.completed_count
            if completed_frames % 10 == 0 and len(self._recent) == 10:
                avg_proc = math.ceil(self._recent_proc_sum / 10)
                avg_cycle = math.ceil(self._recent_cycle_sum / 10)
                total_detections = self._recent_det_sum
                batch_start = completed_frames - 9
                batch_end = completed_frames
                debug_to_log = (batch_start, batch_end, avg_proc, avg_cycle, total_detections)

            if isinstance(profile, dict):
                self._profile_samples += 1
                for stage, value in profile.items():
                    if stage == "total_ms":
                        continue
                    try:
                        self._profile_stage_totals[stage] = self._profile_stage_totals.get(stage, 0.0) + float(value)
                    except (TypeError, ValueError):
                        continue

        if error_to_log:
            log.error("Frame %s OCR error: %s", error_to_log[0], error_to_log[1])
        if debug_to_log:
            log.debug(
                "Frames %s-%s: %sms avg OCR, %sms avg cycle, %s detections",
                debug_to_log[0],
                debug_to_log[1],
                debug_to_log[2],
                debug_to_log[3],
                debug_to_log[4],
            )

    def run_benchmark(self, camera):
        log.info("Starting OCR Benchmark (%s seconds)...", self.duration)
        log.info("Processing frames asynchronously (matches production behavior)...")

        self.running = True
        self.start_time = time.perf_counter()
        self.frame_count = 0
        self.completed_count = 0
        self.error_count = 0
        self._inflight = 0
        self._recent.clear()
        self._recent_proc_sum = 0.0
        self._recent_cycle_sum = 0.0
        self._recent_det_sum = 0
        self._profile_stage_totals = {}
        self._profile_samples = 0
        deadline = self.start_time + self.duration

        while self.running:
            if time.perf_counter() >= deadline:
                self.running = False
                break

            frame = camera.get_frame()
            if frame is None:
                continue

            self.frame_count += 1
            slot_acquired = False

            # Apply backpressure to avoid unlimited OCR worker buildup.
            while True:
                with self.lock:
                    if self._inflight < self.max_inflight:
                        self._inflight += 1
                        slot_acquired = True
                        break
                if time.perf_counter() >= deadline:
                    break
                time.sleep(0.001)

            if time.perf_counter() >= deadline:
                if slot_acquired:
                    with self.lock:
                        self._inflight = max(0, self._inflight - 1)
                self.frame_count -= 1
                break

            # Cycle starts when the async OCR job is actually dispatched.
            trigger_time = time.perf_counter()
            vision.run_vision(
                frame,
                profile=True,
                callback=lambda result, tt=trigger_time, fn=self.frame_count: self._on_vision_result(
                    result, tt, fn
                ),
            )

        elapsed = time.perf_counter() - self.start_time
        log.info("Benchmark time elapsed. Waiting for remaining OCR threads to finish...")

        wait_start = time.perf_counter()
        while True:
            with self.lock:
                inflight = self._inflight
            if inflight == 0:
                break
            if (time.perf_counter() - wait_start) >= self.flush_wait_seconds:
                log.warning("Flush timeout reached with %s in-flight job(s) still running.", inflight)
                break
            time.sleep(0.01)

        log.info(
            "Benchmark complete! Triggered %s frames in %.1fs; completed %s, errors %s.",
            self.frame_count,
            elapsed,
            self.completed_count,
            self.error_count,
        )
        return self.results

    def get_statistics(self):
        if not self.results:
            return None

        cycle_times = [r["cycle_time_ms"] for r in self.results]
        processing_times = [r["processing_time_ms"] for r in self.results]
        detection_counts = [r["detection_count"] for r in self.results]

        stats = {
            "total_frames_triggered": self.frame_count,
            "total_frames_completed": len(self.results),
            "failed_frames": self.error_count,
            "duration_seconds": self.duration,
            "frames_per_second": round(len(self.results) / self.duration, 2),
            "cycle_time": {
                "min_ms": round(min(cycle_times), 2),
                "max_ms": round(max(cycle_times), 2),
                "mean_ms": round(statistics.mean(cycle_times), 2),
                "median_ms": round(statistics.median(cycle_times), 2),
                "stdev_ms": round(statistics.stdev(cycle_times), 2) if len(cycle_times) > 1 else 0,
            },
            "processing_time": {
                "min_ms": round(min(processing_times), 2),
                "max_ms": round(max(processing_times), 2),
                "mean_ms": round(statistics.mean(processing_times), 2),
                "median_ms": round(statistics.median(processing_times), 2),
                "stdev_ms": round(statistics.stdev(processing_times), 2)
                if len(processing_times) > 1
                else 0,
            },
            "detections": {
                "total": sum(detection_counts),
                "min_per_frame": min(detection_counts),
                "max_per_frame": max(detection_counts),
                "mean_per_frame": round(statistics.mean(detection_counts), 2),
            },
        }

        if self._profile_samples:
            stats["profile"] = {
                "samples": self._profile_samples,
                "avg_stage_ms": {
                    stage: round(total / self._profile_samples, 3)
                    for stage, total in sorted(self._profile_stage_totals.items())
                },
            }

        return stats

    def save_results(self, output_dir="benchmark_results"):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"ocr_benchmark_{timestamp}.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "benchmark_duration_seconds": self.duration,
            "vision_mode": cfg["vision_mode"],
            "statistics": self.get_statistics(),
            "detailed_results": self.results,
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        log.info("Results saved to: %s", filename)
        return filename

    def print_summary(self):
        stats = self.get_statistics()
        if not stats:
            log.info("No results to display.")
            return

        log.info("%s", "=" * 60)
        log.info("OCR BENCHMARK SUMMARY")
        log.info("%s", "=" * 60)
        log.info("Mode: %s", cfg["vision_mode"])
        log.info("Benchmark Duration: %s seconds", stats["duration_seconds"])
        log.info("Completed Frames: %s", stats["total_frames_completed"])
        log.info("Triggered Frames: %s", stats["total_frames_triggered"])
        log.info("Failed Frames: %s", stats["failed_frames"])
        log.info("FPS (Completed): %s frames/sec", stats["frames_per_second"])
        log.info("CYCLE TIME (Trigger -> callback complete):")
        log.info("  Min:    %s ms", stats["cycle_time"]["min_ms"])
        log.info("  Max:    %s ms", stats["cycle_time"]["max_ms"])
        log.info("  Mean:   %s ms", stats["cycle_time"]["mean_ms"])
        log.info("  Median: %s ms", stats["cycle_time"]["median_ms"])
        log.info("  StdDev: %s ms", stats["cycle_time"]["stdev_ms"])
        log.info("PROCESSING TIME (OCR inference):")
        log.info("  Min:    %s ms", stats["processing_time"]["min_ms"])
        log.info("  Max:    %s ms", stats["processing_time"]["max_ms"])
        log.info("  Mean:   %s ms", stats["processing_time"]["mean_ms"])
        log.info("  Median: %s ms", stats["processing_time"]["median_ms"])
        log.info("  StdDev: %s ms", stats["processing_time"]["stdev_ms"])
        log.info("DETECTIONS:")
        log.info("  Total:          %s", stats["detections"]["total"])
        log.info("  Min per frame:  %s", stats["detections"]["min_per_frame"])
        log.info("  Max per frame:  %s", stats["detections"]["max_per_frame"])
        log.info("  Mean per frame: %s", stats["detections"]["mean_per_frame"])
        if "profile" in stats:
            log.info("OCR STAGE PROFILE (%s samples):", stats["profile"]["samples"])
            for stage, avg_ms in stats["profile"]["avg_stage_ms"].items():
                log.info("  %s: %s ms", stage, avg_ms)
        log.info("%s", "=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Run OCR benchmark")
    parser.add_argument("--duration", type=int, default=30, help="Benchmark duration in seconds")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory to write benchmark json reports",
    )
    parser.add_argument(
        "--flush-wait",
        type=float,
        default=2.0,
        help="Seconds to wait after trigger loop for worker completion",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=1,
        help="Maximum in-flight async OCR jobs",
    )
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    if cfg["vision_mode"] != "ocr":
        print("WARNING: config vision_mode is not 'ocr'.")
        print("Set vision_mode: ocr in config/default.yaml to benchmark OCR.")
        sys.exit(1)

    camera = Camera(0)
    benchmark = OCRBenchmark(
        duration_seconds=args.duration,
        flush_wait_seconds=args.flush_wait,
        max_inflight=max(1, args.max_inflight),
    )

    try:
        benchmark.run_benchmark(camera)
        benchmark.print_summary()
        benchmark.save_results(args.output_dir)
    except KeyboardInterrupt:
        log.warning("Benchmark interrupted by user.")
        benchmark.print_summary()
        benchmark.save_results(args.output_dir)
    finally:
        camera.release()


if __name__ == "__main__":
    main()
