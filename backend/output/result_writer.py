"""
Result Writer

Handles saving vision processing results to disk in JSONL format, organized by date.
"""

import json
import threading
from datetime import datetime
from pathlib import Path

from backend.core.config_loader import cfg


_write_lock = threading.Lock()

# Pre-resolve output directory at module load time
_result_dir = cfg.get("output", {}).get("result_dir", "data/results")
_output_dir = Path(_result_dir)
if not _output_dir.is_absolute():
    _output_dir = Path(__file__).resolve().parents[2] / _output_dir


def save_result(result: dict):
    """
    Saves a vision processing result to a daily JSONL file.
    """
    now = datetime.now()

    with _write_lock:
        result["timestamp"] = now.isoformat()

        _output_dir.mkdir(parents=True, exist_ok=True)

        daily_file = _output_dir / f"{now.date().isoformat()}.jsonl"
        with daily_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False))
            handle.write("\n")

