from __future__ import annotations
import time
from pathlib import Path
import cv2
from backend.core.camera import Camera
from backend.core.config_loader import cfg


class DatasetCapture:
    def __init__(self, save_dir: str = "data/dataset"):
        self.save_dir = Path(save_dir)
        if not self.save_dir.is_absolute():
            self.save_dir = Path(__file__).resolve().parents[1] / self.save_dir

        self.ok_dir = self.save_dir / "ok"
        self.defective_dir = self.save_dir / "defective"
        self.ok_dir.mkdir(parents=True, exist_ok=True)
        self.defective_dir.mkdir(parents=True, exist_ok=True)

        self.camera_index = int(cfg["camera"]["index"])
        self.counters = {"ok": 0, "defective": 0}
        self.rotation_steps = 0  # 0,1,2,3 => 0/90/180/270 clockwise

    def _save_frame(self, frame, label: str) -> Path:
        timestamp_ms = int(time.time() * 1000)
        target_dir = self.ok_dir if label == "ok" else self.defective_dir
        filename = f"{label}_{timestamp_ms}.jpg"
        output_path = target_dir / filename
        cv2.imwrite(str(output_path), frame)
        self.counters[label] += 1
        return output_path

    def _apply_rotation(self, frame):
        if self.rotation_steps == 1:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation_steps == 2:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.rotation_steps == 3:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def run(self):
        camera = Camera(app_state=None)
        if not camera.cap.isOpened():
            print(f"Failed to open camera index {self.camera_index}")
            return

        window_name = "Dataset Capture | 1=OK  2=Defective  W=Rotate  Q=Quit"
        empty_reads = 0

        try:
            while True:
                frame = camera.get_frame()
                if frame is None:
                    # Camera updates frames in a background thread; allow a short warm-up.
                    empty_reads += 1
                    if empty_reads > 60:
                        print("Failed to read frame from camera.")
                        break
                    time.sleep(0.02)
                    continue
                empty_reads = 0

                display_frame = self._apply_rotation(frame)
                overlay = display_frame.copy()
                counts_text = f"OK: {self.counters['ok']} | Defective: {self.counters['defective']}"
                rotation_text = f"Rotation: {self.rotation_steps * 90} deg"
                cv2.putText(
                    overlay,
                    counts_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    rotation_text,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(window_name, overlay)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("1"):
                    self._save_frame(display_frame, "ok")
                    print(f"Saved OK: {self.counters['ok']} images")
                elif key == ord("2"):
                    self._save_frame(display_frame, "defective")
                    print(f"Saved Defective: {self.counters['defective']} images")
                elif key == ord("w"):
                    self.rotation_steps = (self.rotation_steps + 1) % 4
                    print(f"Rotation set to {self.rotation_steps * 90} degrees")
                elif key == ord("q"):
                    break

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()
            print(
                "Capture complete. Final totals - OK: {ok}, Defective: {defective}".format(
                    ok=self.counters["ok"],
                    defective=self.counters["defective"],
                )
            )


def main():
    print("Dataset Capture Tool")
    print("Press '1' to save an OK image, '2' to save a Defective image, 'w' to rotate 90 degrees, and 'q' to quit.")
    capture = DatasetCapture(save_dir="data/dataset")
    capture.run()


if __name__ == "__main__":
    main()

