"""
Automated Q trigger for long-running vision tests.

The script can be used in two ways:

1. Standalone:
   python tools/q_trigger.py

2. From the main application:
   start_q_trigger(...)

The standalone mode sends a global Q key press at a configurable interval.
The integrated mode uses the same behavior but can optionally skip a trigger
when the application's vision worker is already busy.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from typing import Callable, Optional

import keyboard

log = logging.getLogger(__name__)

DEFAULT_DELAY_S = 30.0
DEFAULT_INTERVAL_S = 1.0
DEFAULT_MAX_HOURS = 8.0


def run_q_trigger(
    stop_event: threading.Event,
    *,
    delay_s: float = DEFAULT_DELAY_S,
    interval_s: float = DEFAULT_INTERVAL_S,
    max_hours: float = DEFAULT_MAX_HOURS,
    can_trigger: Optional[Callable[[], bool]] = None,
) -> None:
    """Send Q presses until stopped or the maximum duration is reached."""
    delay_s = max(0.0, float(delay_s))
    interval_s = max(0.05, float(interval_s))
    max_hours = min(max(0.0, float(max_hours)), 8.0)
    max_duration_s = max_hours * 3600.0

    log.info(
        "Automated Q trigger enabled: first trigger in %.1fs, interval %.2fs, max duration %.2fh",
        delay_s,
        interval_s,
        max_hours,
    )

    if stop_event.wait(delay_s):
        log.info("Automated Q trigger stopped before first trigger")
        return

    started_at = time.monotonic()
    trigger_count = 0

    while not stop_event.is_set():
        elapsed = time.monotonic() - started_at
        if elapsed >= max_duration_s:
            break

        if can_trigger is None or can_trigger():
            try:
                keyboard.press("q")
                trigger_count += 1
                log.info("Automated Q trigger #%d", trigger_count)
            except Exception:
                log.exception("Failed to send automated Q trigger")
        else:
            log.debug("Skipping automated Q trigger because vision is busy")

        # Keep the interval measured between trigger attempts rather than
        # adding it on top of the keypress/logging overhead.
        if stop_event.wait(interval_s):
            return

    log.info(
        "Automated Q trigger finished after %.2fh (%d Q presses)",
        min((time.monotonic() - started_at) / 3600.0, max_hours),
        trigger_count,
    )


def start_q_trigger(
    *,
    delay_s: float = DEFAULT_DELAY_S,
    interval_s: float = DEFAULT_INTERVAL_S,
    max_hours: float = DEFAULT_MAX_HOURS,
    can_trigger: Optional[Callable[[], bool]] = None,
) -> tuple[threading.Thread, threading.Event]:
    """Start the automated Q trigger in a daemon thread."""
    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_q_trigger,
        kwargs={
            "stop_event": stop_event,
            "delay_s": delay_s,
            "interval_s": interval_s,
            "max_hours": max_hours,
            "can_trigger": can_trigger,
        },
        daemon=True,
        name="automated_q_trigger",
    )
    thread.start()
    return thread, stop_event


def main() -> None:
    parser = argparse.ArgumentParser(description="Send Q triggers for a long-running vision test.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_S, help="Seconds before the first Q press")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S, help="Seconds between Q presses")
    parser.add_argument("--hours", type=float, default=DEFAULT_MAX_HOURS, help="Maximum runtime, capped at 8 hours")
    args = parser.parse_args()

    stop_event = threading.Event()

    try:
        run_q_trigger(
            stop_event,
            delay_s=args.delay,
            interval_s=args.interval,
            max_hours=args.hours,
        )
    except KeyboardInterrupt:
        stop_event.set()
        log.info("Automated Q trigger interrupted")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()