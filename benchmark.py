import time
import threading
import statistics
from camera import Camera
from vision import run_vision, ocr_instance
from pathlib import Path
import json
from datetime import datetime


class OCRBenchmark:
    """
    Benchmark class to test OCR cycle performance over 30 seconds.
    Measures cycle time, processing time, detection counts, and statistics.
    """

    def __init__(self, duration_seconds=30):
        self.duration = duration_seconds
        self.results = []
        self.lock = threading.Lock()
        self.running = False
        self.start_time = None

    def run_benchmark(self, camera):
        """
        Run OCR benchmark for specified duration.
        Triggers OCR on each frame and records cycle times.
        """
        print(f"Starting OCR Benchmark ({self.duration} seconds)...")
        print("Processing frames...\n")

        self.running = True
        self.start_time = time.perf_counter()
        frame_count = 0

        while self.running:
            elapsed = time.perf_counter() - self.start_time

            if elapsed >= self.duration:
                self.running = False
                break

            frame = camera.get_frame()
            if frame is None:
                continue

            frame_count += 1
            trigger_time = time.perf_counter()

            # Run OCR synchronously for benchmarking
            result = ocr_instance.run(frame)
            cycle_time_ms = (time.perf_counter() - trigger_time) * 1000

            with self.lock:
                self.results.append({
                    "frame_num": frame_count,
                    "cycle_time_ms": round(cycle_time_ms, 2),
                    "processing_time_ms": result.get("processing_time_ms", 0),
                    "detection_count": len(result.get("detections", []))
                })

            # Print progress every 10 frames
            if frame_count % 10 == 0:
                print(f"  Frame {frame_count}: {cycle_time_ms:.2f}ms cycle, "
                      f"{len(result.get('detections', []))} average detections")

        print(f"\nBenchmark complete! Processed {frame_count} frames in {elapsed:.1f}s\n")
        return self.results

    def get_statistics(self):
        """Calculate and return performance statistics"""
        if not self.results:
            return None

        cycle_times = [r["cycle_time_ms"] for r in self.results]
        processing_times = [r["processing_time_ms"] for r in self.results]
        detection_counts = [r["detection_count"] for r in self.results]

        stats = {
            "total_frames": len(self.results),
            "duration_seconds": self.duration,
            "frames_per_second": round(len(self.results) / self.duration, 2),
            "cycle_time": {
                "min_ms": round(min(cycle_times), 2),
                "max_ms": round(max(cycle_times), 2),
                "mean_ms": round(statistics.mean(cycle_times), 2),
                "median_ms": round(statistics.median(cycle_times), 2),
                "stdev_ms": round(statistics.stdev(cycle_times), 2) if len(cycle_times) > 1 else 0
            },
            "processing_time": {
                "min_ms": round(min(processing_times), 2),
                "max_ms": round(max(processing_times), 2),
                "mean_ms": round(statistics.mean(processing_times), 2),
                "median_ms": round(statistics.median(processing_times), 2),
                "stdev_ms": round(statistics.stdev(processing_times), 2) if len(processing_times) > 1 else 0
            },
            "detections": {
                "total": sum(detection_counts),
                "min": min(detection_counts),
                "max": max(detection_counts),
                "mean": round(statistics.mean(detection_counts), 2)
            }
        }

        return stats

    def save_results(self, output_dir="benchmark_results"):
        """
        Save benchmark results to a JSON file in the specified directory.
        Creates directory if it doesn't exist.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"ocr_benchmark_{timestamp}.json"

        stats = self.get_statistics()

        report = {
            "timestamp": datetime.now().isoformat(),
            "benchmark_duration_seconds": self.duration,
            "statistics": stats,
            "detailed_results": self.results
        }

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Results saved to: {filename}")
        return filename

    def print_summary(self):
        """Print a summary of the benchmark results"""
        stats = self.get_statistics()

        if not stats:
            print("No results to display.")
            return

        print("\n" + "="*60)
        print("OCR BENCHMARK SUMMARY")
        print("="*60)
        print(f"Duration: {stats['duration_seconds']} seconds")
        print(f"Total Frames Processed: {stats['total_frames']}")
        print(f"Frames Per Second: {stats['frames_per_second']}")
        print()
        print("CYCLE TIME (Full OCR cycle including overhead):")
        print(f"  Min:    {stats['cycle_time']['min_ms']} ms")
        print(f"  Max:    {stats['cycle_time']['max_ms']} ms")
        print(f"  Mean:   {stats['cycle_time']['mean_ms']} ms")
        print(f"  Median: {stats['cycle_time']['median_ms']} ms")
        print(f"  StdDev: {stats['cycle_time']['stdev_ms']} ms")
        print()
        print("PROCESSING TIME (OCR inference only):")
        print(f"  Min:    {stats['processing_time']['min_ms']} ms")
        print(f"  Max:    {stats['processing_time']['max_ms']} ms")
        print(f"  Mean:   {stats['processing_time']['mean_ms']} ms")
        print(f"  Median: {stats['processing_time']['median_ms']} ms")
        print(f"  StdDev: {stats['processing_time']['stdev_ms']} ms")
        print()
        print("DETECTIONS:")
        print(f"  Total:  {stats['detections']['total']}")
        print(f"  Min per frame:  {stats['detections']['min']}")
        print(f"  Max per frame:  {stats['detections']['max']}")
        print(f"  Mean per frame: {stats['detections']['mean']}")
        print("="*60 + "\n")


def main():
    """Run the benchmark"""
    camera = Camera(0)
    benchmark = OCRBenchmark(duration_seconds=30)

    try:
        benchmark.run_benchmark(camera)
        benchmark.print_summary()
        benchmark.save_results()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        benchmark.print_summary()
        benchmark.save_results()
    finally:
        camera.release()


if __name__ == "__main__":
    main()

