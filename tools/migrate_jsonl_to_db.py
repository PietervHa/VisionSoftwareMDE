"""
Migrate JSONL result history into the SQL database.

One-time migration: reads every data/results/*.jsonl file and inserts each
line as a row via backend.core.db.insert_result(), so existing inspection
history isn't lost when switching from file-based storage to the database.

Usage:
    python tools/migrate_jsonl_to_db.py
    python tools/migrate_jsonl_to_db.py --dry-run
    python tools/migrate_jsonl_to_db.py --force   # re-run even if the table already has rows

Safe to run from a fresh checkout: it only reads data/results/*.jsonl and
writes to the database configured in config.yaml. It refuses to run twice
by default (use --force to override) since this would duplicate rows —
there's no unique constraint on a result row to de-duplicate against.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `backend.*` importable regardless of how this script is invoked
# (running it directly puts only tools/ on sys.path, not the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select
from tqdm import tqdm

from backend.core import db
from backend.core.config_loader import cfg

BATCH_SIZE = 1000


def _result_dir() -> Path:
    result_dir = cfg.get("output", {}).get("result_dir", "data/results")
    path = Path(result_dir)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path


def _existing_row_count() -> int:
    with db.engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(db.results_table)).scalar_one()


def migrate(dry_run: bool = False, force: bool = False) -> None:
    db.init_db()

    existing = _existing_row_count()
    if existing and not dry_run and not force:
        print(f"'results' table already has {existing} rows — refusing to run again (use --force to override).")
        sys.exit(1)
    if existing and force:
        print(f"Proceeding despite {existing} existing rows (--force).")

    result_dir = _result_dir()
    files = sorted(result_dir.glob("*.jsonl"))
    if not files:
        print(f"No .jsonl files found in {result_dir}")
        return

    total_inserted = 0
    total_skipped = 0
    batch: list[dict] = []

    def flush(batch: list[dict]) -> None:
        if not batch or dry_run:
            return
        with db.engine.begin() as conn:
            rows = [db._row_from_result(r) for r in batch]
            conn.execute(db.results_table.insert(), rows)

    for file_path in tqdm(files, desc="Files"):
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    total_skipped += 1
                    tqdm.write(f"Skipping malformed line {line_number} in {file_path.name}")
                    continue

                batch.append(record)
                total_inserted += 1

                if len(batch) >= BATCH_SIZE:
                    flush(batch)
                    batch = []

    flush(batch)

    verb = "would insert" if dry_run else "inserted"
    print(f"\nDone. {verb} {total_inserted} rows, skipped {total_skipped} malformed lines.")
    if not dry_run:
        print(f"'results' table now has {_existing_row_count()} rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Parse and count without writing to the database")
    parser.add_argument("--force", action="store_true", help="Run even if the results table already has rows")
    args = parser.parse_args()

    migrate(dry_run=args.dry_run, force=args.force)