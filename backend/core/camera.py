import cv2
import threading
import time
import sys
from backend.core.config_loader import cfg
from backend.utils.logger import get_logger

log = get_logger(__name__)
TARGET = 1 / 30  # Target time per frame for ~30 FPS

class Camera:
    def __init__(self, index=0):  # <- change index
        cam_cfg = cfg["camera"]
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.app_state = app_state
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

        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self._flip = cam_cfg["flip"]

        t = threading.Thread(target=self._update, daemon=True)
        t.start()


    def _update(self):
        while self.running:
            t0 = time.perf_counter()
            ret, frame = self.cap.read()
            if ret:
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
            else:
                log.warning("Camera frame read failed")
            elapsed = time.perf_counter() - t0
            remaining = TARGET - elapsed
            if remaining > 0:
                time.sleep(remaining)


    def get_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()


    def release(self):
        log.info("Camera release called")
        self.running = False
        self.cap.release()