import cv2
import threading
import time

class Camera:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True

        t = threading.Thread(target=self._update, daemon=True)
        t.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # verwijder de # in de lijn hieronder om de camera te flippen hierdoor word het spiegelbeeld
                #frame = cv2.flip(frame, 1)  # 1 = horizontal flip (mirror) 0 = verticalflip -1 = both
                with self.lock:
                    self.latest_frame = frame
            time.sleep(0.005)  # prevent CPU burn

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def release(self):
        self.running = False
        self.cap.release()