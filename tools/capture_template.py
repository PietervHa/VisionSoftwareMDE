import time
from pathlib import Path
import cv2
import numpy as np
from backend.core.camera import Camera
from backend.core.config_loader import cfg


def _resolve_output_dir() -> Path:
    od_cfg = cfg.get("object_detection", {})
    template_cfg = od_cfg.get("template", {}) if isinstance(od_cfg.get("template"), dict) else {}
    references = template_cfg.get("references", [])

    # Derive output dir from first reference path, fall back to models/templates
    if references:
        output_dir = Path(references[0]).parent
    else:
        output_dir = Path("models/templates")

    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[1] / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _draw_overlay(frame: np.ndarray, frozen: bool) -> np.ndarray:
    display = frame.copy()
    cv2.putText(
        display,
        "SPACE: freeze  |  Q/ESC: quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if frozen:
        cv2.putText(
            display,
            "ENTER: select ROI  |  SPACE: unfreeze",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            "FROZEN",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return display


def main() -> None:
    output_dir = _resolve_output_dir()
    camera = Camera()
    window_name = "Template Capture"

    frozen = False
    frozen_frame = None

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 744, 480)

        while True:
            if not frozen:
                frame = camera.get_frame()
                if frame is not None:
                    frozen_frame = frame.copy()
                else:
                    time.sleep(0.01)

            if frozen_frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            display = _draw_overlay(frozen_frame, frozen)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord(" "):
                frozen = not frozen
                continue

            if frozen and key in (13, 10):
                x, y, w, h = cv2.selectROI(
                    "Select ROI",
                    frozen_frame,
                    fromCenter=False,
                    showCrosshair=True,
                )

                if w > 0 and h > 0:
                    crop = frozen_frame[y : y + h, x : x + w]

                    timestamp = int(time.time() * 1000)
                    base_name = f"template_{timestamp}"
                    base_path = output_dir / f"{base_name}.jpg"
                    cv2.imwrite(str(base_path), crop)
                    print(f"Saved: {base_path}  ({crop.shape[1]}x{crop.shape[0]} px)")

                    for scale in [0.85, 1.15]:
                        new_w = max(1, int(round(crop.shape[1] * scale)))
                        new_h = max(1, int(round(crop.shape[0] * scale)))
                        scaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        scale_str = str(scale).replace(".", "_")
                        scale_path = output_dir / f"{base_name}_scale{scale_str}.jpg"
                        cv2.imwrite(str(scale_path), scaled)
                        print(f"Saved scale variant: {scale_path}  ({new_w}x{new_h} px)")

                    print("\nAdd to config/default.yaml under object_detection.template.references:")
                    print(f"  - data/templates/{base_name}.jpg")
                    print(f"  - data/templates/{base_name}_scale0_85.jpg")
                    print(f"  - data/templates/{base_name}_scale1_15.jpg\n")

                frozen = False
                cv2.destroyWindow("Select ROI")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

