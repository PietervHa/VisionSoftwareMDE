"""
Camera Management Module

Provides the Camera class for asynchronous frame capture, supporting various 
rotations and flipping based on configuration and application state.
"""

import cv2
import threading
import time
import sys
from backend.core.config_loader import cfg
from backend.utils.logger import get_logger

log = get_logger(__name__)
TARGET = 1 / 30  # Target time per frame for ~30 FPS

# Consecutive failed reads (roughly 1s at target FPS) before the update
# thread attempts to reopen the capture device on its own.
RECONNECT_AFTER_FAILURES = 30

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

        self.cap = None
        self._open_capture()

        t = threading.Thread(target=self._update, daemon=True)
        t.start()

    def _open_capture(self):
        """
        Opens (or reopens) the underlying VideoCapture device using the
        configured index/resolution. Shared by __init__ and the automatic
        reconnect logic in _update() so there is one place that knows how
        to stand the camera back up.
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

        if self.cap.isOpened():
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

    def _update(self):
        """
        Internal worker thread that continuously reads frames from the camera.

        Both the read and any post-processing are guarded so an unexpected
        exception (e.g. the device disappearing mid-read) can't silently
        kill this thread and leave the app serving a stale frame forever.
        After a sustained run of failed reads, the capture device is
        automatically reopened.
        """
        consecutive_failures = 0

        while self.running:
            t0 = time.perf_counter()
            try:
                ret, frame = self.cap.read()
            except Exception as exc:
                log.error("Camera read raised an exception: %s", exc, exc_info=True)
                ret, frame = False, None

            if ret and frame is not None:
                consecutive_failures = 0
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
                if consecutive_failures >= RECONNECT_AFTER_FAILURES:
                    log.error(
                        "Camera unresponsive after %d consecutive failed reads; attempting to reopen",
                        consecutive_failures,
                    )
                    try:
                        self._open_capture()
                    except Exception as exc:
                        log.error("Camera reopen attempt failed: %s", exc, exc_info=True)
                    consecutive_failures = 0

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


    def release(self):
        """
        Stops the update thread and releases the camera hardware.
        """
        log.info("Camera release called")
        self.running = False
        self.cap.release()