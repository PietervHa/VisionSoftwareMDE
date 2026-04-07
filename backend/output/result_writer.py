import json
import threading
from datetime import datetime
from pathlib import Path

from backend.core.config_loader import cfg


_write_lock = threading.Lock()


def save_result(result: dict):
    now = datetime.now()

    with _write_lock:
        result["timestamp"] = now.isoformat()

        result_dir = cfg.get("output", {}).get("result_dir", "data/results")
        output_dir = Path(result_dir)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).resolve().parents[2] / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        daily_file = output_dir / f"{now.date().isoformat()}.jsonl"
        with daily_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False))
            handle.write("\n")

