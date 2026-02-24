import cv2
import threading
import time

class Camera:
    def __init__(self, index=1):  # <- change index
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True

        t = threading.Thread(target=self._update, daemon=True)
        t.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            time.sleep(0.005)

    def get_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def release(self):
        self.running = False
        self.cap.release()