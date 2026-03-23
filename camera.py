import cv2
import threading
import time
from config_loader import cfg
from utils.logger import get_logger

log = get_logger(__name__)

class Camera:
    def __init__(self, index=0):  # <- change index
        cam_cfg = cfg["camera"]
        self.cap = cv2.VideoCapture(cam_cfg["index"], cv2.CAP_DSHOW)

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
                with self.lock:
                    self.latest_frame = frame
            else:
                log.warning("Camera frame read failed")
            time.sleep(0.005)

    def get_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def release(self):
        log.info("Camera release called")
        self.running = False
        self.cap.release()