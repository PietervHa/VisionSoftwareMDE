import cv2
import threading
import time
from backend.core.config_loader import cfg
from backend.utils.logger import get_logger

log = get_logger(__name__)

class Camera:
    def __init__(self, index=0, app_state=None):  # <- change index
        cam_cfg = cfg["camera"]
        self.app_state = app_state
        self.cap = cv2.VideoCapture(cam_cfg["index"], cv2.CAP_DSHOW)
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

        t = threading.Thread(target=self._update, daemon=True)
        t.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Flip the frame (1 = horizontal, 0 = vertical, -1 = both)
                frame = cv2.flip(frame, cfg["camera"]["flip"])
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
            time.sleep(1/30)

    def get_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def release(self):
        log.info("Camera release called")
        self.running = False
        self.cap.release()