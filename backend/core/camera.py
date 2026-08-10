"""
Camera Management Module

Provides the Camera class for asynchronous frame capture, supporting various
rotations and flipping based on configuration and application state.

Recovery behaviour
-------------------
The background update thread continuously monitors read health. When frame
reads start failing (e.g. the camera is unplugged), it detects this quickly
(FAILURE_THRESHOLD consecutive failures) and starts attempting to reopen the
capture device. Reopen attempts use a short exponential backoff so that:
  - a camera that comes back quickly (replugged) is picked up almost
    instantly (first retry after RECONNECT_MIN_INTERVAL seconds), and
  - a camera that stays absent for a while doesn't cause the thread to
    hammer the OS/driver with repeated open attempts (retry interval is
    capped at RECONNECT_MAX_INTERVAL seconds).
A manual reconnect() can also be requested from any thread (e.g. a "retry"
button in the UI); it's honored on the capture thread itself so there's no
cross-thread access to the underlying cv2.VideoCapture object.
"""

import cv2
import threading
import time
import sys
from backend.core.config_loader import cfg
from backend.utils.logger import get_logger

log = get_logger(__name__)
TARGET = 1 / 30  # Target time per frame for ~30 FPS

# Consecutive failed reads before the update thread treats the camera as
# down and starts attempting to reopen it. Kept small so recovery reacts
# quickly, while still ignoring a single transient dropped frame.
FAILURE_THRESHOLD = 5  # ~0.15s at target FPS

# Backoff bounds (in seconds) between reopen attempts while the camera is
# down. Starts fast so a quick replug is picked up almost immediately, and
# backs off so a camera that stays unplugged doesn't get hammered with
# open attempts.
RECONNECT_MIN_INTERVAL = 0.25
RECONNECT_MAX_INTERVAL = 2.0


class Camera:
    """
    Asynchronous camera interface that captures frames in a background thread.
    """
    def __init__(self, index=0, app_state=None):
        self.app_state = app_state
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self._flip = cfg["camera"]["flip"]

        # Connection health, readable from any thread (plain bool reads/
        # writes are atomic under the GIL, so no extra lock is needed here).
        self.connected = False
        # Set from any thread to request an immediate reconnect attempt on
        # the capture thread, bypassing the current backoff wait.
        self._force_reconnect = threading.Event()

        self.cap = None
        self._open_capture()

        t = threading.Thread(target=self._update, daemon=True)
        t.start()

    def _open_capture(self):
        """
        Opens (or reopens) the underlying VideoCapture device using the
        configured index/resolution. Shared by __init__ and the automatic
        reconnect logic in _update() so there is one place that knows how
        to stand the camera back up. Updates self.connected to reflect the
        outcome.
        """
        cam_cfg = cfg["camera"]
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as exc:
                log.warning("Error releasing camera before reopen: %s", exc)

        self.cap = cv2.VideoCapture(cam_cfg["index"], backend)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])

        self.connected = bool(self.cap.isOpened())

        if self.connected:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info(
                "Camera opened successfully: index=%s resolution=%sx%s",
                cam_cfg["index"],
                width,
                height,
            )
        else:
            log.error("Camera failed to open: index=%s", cam_cfg["index"])

        return self.connected

    def _attempt_reconnect(self, failures_so_far):
        """
        Wraps _open_capture() for use from the recovery paths below, so
        exceptions during a reopen attempt (e.g. driver hiccup) can never
        kill the update thread.
        """
        log.info(
            "Attempting to reconnect camera (after %d failed reads)...",
            failures_so_far,
        )
        try:
            return self._open_capture()
        except Exception as exc:
            log.error("Camera reconnect attempt failed: %s", exc, exc_info=True)
            self.connected = False
            return False

    def _update(self):
        """
        Internal worker thread that continuously reads frames from the camera.

        Both the read and any post-processing are guarded so an unexpected
        exception (e.g. the device disappearing mid-read) can't silently
        kill this thread and leave the app serving a stale frame forever.
        On sustained read failures, the capture device is automatically
        reopened using a fast-detect + backoff strategy (see module
        docstring) so the live feed recovers as soon as the camera is
        available again.
        """
        consecutive_failures = 0
        next_reconnect_attempt = 0.0
        backoff = RECONNECT_MIN_INTERVAL

        while self.running:
            t0 = time.perf_counter()

            # Manual reconnect request (e.g. a "retry" button in the UI),
            # honored immediately regardless of the current failure count
            # or backoff timer.
            if self._force_reconnect.is_set():
                self._force_reconnect.clear()
                self._attempt_reconnect(consecutive_failures)
                consecutive_failures = 0
                backoff = RECONNECT_MIN_INTERVAL
                next_reconnect_attempt = 0.0

            try:
                ret, frame = self.cap.read()
            except Exception as exc:
                log.error("Camera read raised an exception: %s", exc, exc_info=True)
                ret, frame = False, None

            if ret and frame is not None:
                if consecutive_failures > 0:
                    log.info(
                        "Camera recovered after %d failed reads",
                        consecutive_failures,
                    )
                consecutive_failures = 0
                backoff = RECONNECT_MIN_INTERVAL
                self.connected = True
                try:
                    # Flip the frame (1 = horizontal, 0 = vertical, -1 = both)
                    frame = cv2.flip(frame, self._flip)
                    rotation = self.app_state.get_camera_rotation() if self.app_state else 0
                    if rotation == 1:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif rotation == 2:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif rotation == 3:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    # rotation == 0: no rotation
                    with self.lock:
                        self.latest_frame = frame
                except Exception as exc:
                    log.error("Frame post-processing failed: %s", exc, exc_info=True)
            else:
                consecutive_failures += 1
                if consecutive_failures == 1:
                    log.warning("Camera frame read failed")
                    self.connected = False

                if consecutive_failures >= FAILURE_THRESHOLD:
                    now = time.monotonic()
                    if now >= next_reconnect_attempt:
                        success = self._attempt_reconnect(consecutive_failures)
                        if success:
                            backoff = RECONNECT_MIN_INTERVAL
                        else:
                            backoff = min(backoff * 2, RECONNECT_MAX_INTERVAL)
                        next_reconnect_attempt = now + backoff

            elapsed = time.perf_counter() - t0
            remaining = TARGET - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def get_frame(self):
        """
        Retrieves a thread-safe copy of the latest captured frame.
        """
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def is_connected(self):
        """
        Returns whether the last read from the camera succeeded. Useful for
        surfacing a "camera disconnected / reconnecting..." state in the UI.
        """
        return self.connected

    def reconnect(self):
        """
        Requests an immediate reconnect attempt, bypassing the current
        backoff wait. Safe to call from any thread (e.g. a manual "retry"
        button) since the actual VideoCapture access still only ever
        happens on the camera's own update thread.
        """
        self._force_reconnect.set()

    def release(self):
        """
        Stops the update thread and releases the camera hardware.
        """
        log.info("Camera release called")
        self.running = False
        self.cap.release()