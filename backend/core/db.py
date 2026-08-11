"""
Database Layer

SQL storage for vision inspection results, via SQLAlchemy Core. The point
of using SQLAlchemy instead of raw sqlite3/pyodbc calls is portability:
this module works unchanged against SQLite today and against MSSQL later —
only `database.url` in config.yaml needs to change.

    config.yaml:
        database:
          url: "sqlite:///data/results.db"
          # later, once a server is available:
          # url: "mssql+pyodbc://user:password@server/db?driver=ODBC+Driver+17+for+SQL+Server"

Schema design: different detection backends (OCR, template matching,
classifier, YOLO, Roboflow) each log their own extra fields
(searched_word, match_score, best_reference, label, count, ...). Rather
than giving every possible field its own nullable column — which means a
schema migration every time a backend changes — this module keeps a fixed
set of columns for whatever the analytics dashboard and exports need to
filter/aggregate on, and stores everything else as a JSON blob in
`details`. Nothing is lost; nothing needs to change when a backend adds a
new field.

NOTE for the MSSQL migration later: this has only been tested against
SQLite. Two things worth double-checking once a real MSSQL server is
available:
  - `pyodbc` (or `pymssql`) needs to be installed, plus the matching ODBC
    driver on the machine running this.
  - The `DateTime` columns here map to MSSQL's plain DATETIME type
    (~3.33ms precision). The JSONL history has microsecond timestamps; if
    that precision matters, switch to `mssql.DATETIME2` for those columns.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    create_engine,
    select,
)
from sqlalchemy.engine import Engine

from backend.core.config_loader import cfg

log = logging.getLogger(__name__)

# Fields that get their own column. Everything else on a result dict is
# folded into the `details` JSON blob — see module docstring.
_CORE_FIELDS = {
    "timestamp",
    "status",
    "mode",
    "confidence_threshold",
    "processing_time_ms",
    "cycle_time_ms",
    "error",
}

metadata = MetaData()

results_table = Table(
    "results",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, nullable=False),
    Column("status", String(10), nullable=False),
    Column("mode", String(50), nullable=True),
    Column("confidence_threshold", Float, nullable=True),
    Column("processing_time_ms", Float, nullable=True),
    Column("cycle_time_ms", Float, nullable=True),
    Column("error", Text, nullable=True),
    Column("details", Text, nullable=True),  # JSON-encoded, dialect-portable
    Index("ix_results_timestamp", "timestamp"),
)


def _resolve_sqlite_url(url: str) -> str:
    """Resolve relative sqlite:/// paths against the repo root, like every other path in this project."""
    prefix = "sqlite:///"
    if not url.startswith(prefix) or url.startswith("sqlite:////"):
        return url  # not sqlite, or already an absolute path
    relative_part = url[len(prefix):]
    repo_root = Path(__file__).resolve().parents[2]
    absolute_path = (repo_root / relative_part).resolve()
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    return f"{prefix}{absolute_path}"


def _build_engine() -> Engine:
    url = cfg.get("database", {}).get("url", "sqlite:///data/results.db")
    url = _resolve_sqlite_url(url)
    log.info("Database engine: %s", url.split("://")[0] + "://...")
    return create_engine(url, future=True)


engine: Engine = _build_engine()


def init_db() -> None:
    """Create the results table if it doesn't exist yet. Safe to call on every startup."""
    metadata.create_all(engine)


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            log.warning("Could not parse timestamp %r, using current time", value)
    return datetime.now()


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_from_result(result: dict) -> dict:
    """Build the DB row dict for one result. Shared by insert_result() and the migration script's bulk insert."""
    details = {k: v for k, v in result.items() if k not in _CORE_FIELDS}
    return {
        "timestamp": _parse_timestamp(result.get("timestamp")),
        "status": result.get("status") or "NOK",
        "mode": result.get("mode"),
        "confidence_threshold": _safe_float(result.get("confidence_threshold")),
        "processing_time_ms": _safe_float(result.get("processing_time_ms")),
        "cycle_time_ms": _safe_float(result.get("cycle_time_ms")),
        "error": result.get("error"),
        "details": json.dumps(details, ensure_ascii=False) if details else None,
    }


def insert_result(result: dict) -> None:
    """Insert one inspection result. Same `result` dict shape as the JSONL writer expects."""
    row = _row_from_result(result)
    with engine.begin() as conn:
        conn.execute(results_table.insert().values(**row))


def _date_bounds(start: date, end: date) -> tuple[datetime, datetime]:
    return datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.max.time())


def query_summary(start: date, end: date) -> list[dict]:
    """
    Lightweight query for the analytics dashboard: timestamp/status/
    processing_time_ms only, no `details` blob. Used on every dashboard
    refresh, so keep it cheap.
    """
    start_dt, end_dt = _date_bounds(start, end)
    cols = results_table.c
    stmt = (
        select(cols.timestamp, cols.status, cols.processing_time_ms)
        .where(and_(cols.timestamp >= start_dt, cols.timestamp <= end_dt))
        .order_by(cols.timestamp)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [dict(r) for r in rows]


def query_full(start: date, end: date) -> list[dict]:
    """
    Full query for exports: every core column plus `details` parsed back
    out to individual keys, so an exported row looks like the original
    JSONL record again.
    """
    start_dt, end_dt = _date_bounds(start, end)
    cols = results_table.c
    stmt = (
        select(results_table)
        .where(and_(cols.timestamp >= start_dt, cols.timestamp <= end_dt))
        .order_by(cols.timestamp)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()

    results = []
    for row in rows:
        item = {k: v for k, v in dict(row).items() if k not in ("id", "details")}
        if isinstance(item.get("timestamp"), datetime):
            item["timestamp"] = item["timestamp"].isoformat()
        details_raw = row.get("details")
        if details_raw:
            try:
                item.update(json.loads(details_raw))
            except (TypeError, json.JSONDecodeError):
                pass
        results.append(item)
    return results