"""
Standalone smoke test for the Camera recovery logic in camera.py.

Stubs out cv2 and the backend.* imports so the module can be exercised
without OpenCV or the real project, then simulates a camera being
unplugged and replugged to verify:
  1. Disconnects are detected quickly (FAILURE_THRESHOLD reads).
  2. The feed recovers automatically once the device is "replugged".
  3. is_connected() accurately reflects state throughout.
  4. reconnect() forces an immediate retry.

Run with: python3 test_camera_recovery.py
"""
import sys
import types
import time
import unittest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Fake cv2 + backend.* modules so camera.py can be imported standalone
# ---------------------------------------------------------------------------

class FakeDevice:
    """Shared, controllable state simulating a physical camera."""
    def __init__(self):
        self.present = True
        self.frame_counter = 0


DEVICE = FakeDevice()


class FakeVideoCapture:
    def __init__(self, index, backend):
        self.index = index
        self.backend = backend
        # isOpened() reflects device presence *at construction time*,
        # matching real cv2 behaviour (opening while unplugged fails).
        self._opened = DEVICE.present

    def set(self, prop, value):
        return True

    def get(self, prop):
        return 640 if prop == FAKE_CV2.CAP_PROP_FRAME_WIDTH else 480

    def isOpened(self):
        return self._opened

    def read(self):
        if not DEVICE.present or not self._opened:
            return False, None
        DEVICE.frame_counter += 1
        return True, {"frame_id": DEVICE.frame_counter}

    def release(self):
        self._opened = False


FAKE_CV2 = types.SimpleNamespace(
    VideoCapture=FakeVideoCapture,
    CAP_DSHOW=0,
    CAP_ANY=0,
    CAP_PROP_BUFFERSIZE=1,
    CAP_PROP_FRAME_WIDTH=3,
    CAP_PROP_FRAME_HEIGHT=4,
    flip=lambda frame, code: frame,
    rotate=lambda frame, code: frame,
    ROTATE_90_CLOCKWISE=0,
    ROTATE_180=1,
    ROTATE_90_COUNTERCLOCKWISE=2,
)

sys.modules["cv2"] = FAKE_CV2

backend_pkg = types.ModuleType("backend")
backend_core = types.ModuleType("backend.core")
backend_utils = types.ModuleType("backend.utils")
config_loader_mod = types.ModuleType("backend.core.config_loader")
logger_mod = types.ModuleType("backend.utils.logger")

config_loader_mod.cfg = {
    "camera": {"flip": 1, "index": 0, "width": 640, "height": 480}
}
logger_mod.get_logger = lambda name: MagicMock()

sys.modules["backend"] = backend_pkg
sys.modules["backend.core"] = backend_core
sys.modules["backend.utils"] = backend_utils
sys.modules["backend.core.config_loader"] = config_loader_mod
sys.modules["backend.utils.logger"] = logger_mod

sys.path.insert(0, "/home/claude/work")
import camera as camera_module  # noqa: E402


def wait_until(predicate, timeout=3.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class CameraRecoveryTests(unittest.TestCase):
    def setUp(self):
        DEVICE.present = True
        DEVICE.frame_counter = 0
        self.cam = camera_module.Camera(app_state=None)
        self.assertTrue(
            wait_until(lambda: self.cam.get_frame() is not None),
            "camera never produced an initial frame",
        )

    def tearDown(self):
        self.cam.release()

    def test_initially_connected_and_streaming(self):
        self.assertTrue(self.cam.is_connected())
        self.assertIsNotNone(self.cam.get_frame())

    def test_detects_unplug_quickly(self):
        DEVICE.present = False
        detected = wait_until(lambda: not self.cam.is_connected(), timeout=1.0)
        self.assertTrue(detected, "disconnect was not detected in time")

    def test_recovers_after_replug(self):
        DEVICE.present = False
        self.assertTrue(wait_until(lambda: not self.cam.is_connected(), timeout=1.0))

        last_frame_before = self.cam.get_frame()

        # Simulate the user plugging the camera back in.
        DEVICE.present = True

        recovered = wait_until(lambda: self.cam.is_connected(), timeout=3.0)
        self.assertTrue(recovered, "camera did not recover after replug")

        # New frames should resume flowing.
        got_new_frame = wait_until(
            lambda: self.cam.get_frame() != last_frame_before, timeout=1.0
        )
        self.assertTrue(got_new_frame, "live feed did not resume after recovery")

    def test_manual_reconnect_forces_immediate_retry(self):
        DEVICE.present = False
        self.assertTrue(wait_until(lambda: not self.cam.is_connected(), timeout=1.0))

        # Camera comes back, but we don't want to wait for the backoff
        # timer - simulate pressing a "retry" button in the UI.
        DEVICE.present = True
        self.cam.reconnect()

        recovered = wait_until(lambda: self.cam.is_connected(), timeout=0.5)
        self.assertTrue(recovered, "manual reconnect() did not recover promptly")

    def test_stays_down_while_unplugged_no_crash(self):
        DEVICE.present = False
        self.assertTrue(wait_until(lambda: not self.cam.is_connected(), timeout=1.0))
        # Let several backoff cycles pass; thread must survive and keep
        # reporting disconnected rather than raising/dying.
        time.sleep(1.0)
        self.assertFalse(self.cam.is_connected())


if __name__ == "__main__":
    unittest.main(verbosity=2)