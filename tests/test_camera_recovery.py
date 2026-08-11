"""
Standalone smoke test for the Camera recovery logic in camera.py.

Stubs out cv2 and the backend.* imports so the module can be exercised
without OpenCV or the real project. Includes a FakeVideoCapture that can
simulate all three disconnect failure modes seen in practice:
  - fail fast: read() returns (False, None)
  - hang: read() blocks (reproducing the DirectShow-on-Windows hang)
  - freeze: read() keeps returning (True, <same frame as before>)
    (some DirectShow drivers re-serve the last buffered frame instead
    of failing once the device is gone - this is what real hardware
    testing surfaced: the app never noticed the disconnect at all)

Run with: python3 test_camera_recovery.py
"""
import sys
import types
import time
import threading
import unittest
import numpy as np
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Fake cv2 + backend.* modules so camera.py can be imported standalone
# ---------------------------------------------------------------------------

class FakeDevice:
    """Shared, controllable state simulating a physical camera."""
    def __init__(self):
        self.present = True
        self.frame_counter = 0
        self.present_event = threading.Event()
        self.present_event.set()
        # If True, read() blocks (like a hung DirectShow call) while the
        # device is absent, instead of returning (False, None).
        self.hang_on_disconnect = False
        # If True, read() keeps returning (True, <last frame>) while the
        # device is absent, instead of failing or blocking.
        self.freeze_on_disconnect = False
        # Simulates real camera warm-up: each freshly opened VideoCapture
        # returns (False, None) for this many seconds after being opened,
        # before it starts delivering real frames.
        self.warmup_seconds = 0.0

    def set_present(self, value):
        self.present = value
        if value:
            self.present_event.set()
        else:
            self.present_event.clear()


DEVICE = FakeDevice()


class FakeVideoCapture:
    def __init__(self, index, backend):
        self.index = index
        self.backend = backend
        self._opened = DEVICE.present
        self._released = False
        self._last_frame = None
        self._created_at = time.monotonic()

    def set(self, prop, value):
        return True

    def get(self, prop):
        return 640 if prop == FAKE_CV2.CAP_PROP_FRAME_WIDTH else 480

    def isOpened(self):
        return self._opened

    def read(self):
        if self._released:
            return False, None

        if not DEVICE.present:
            if DEVICE.freeze_on_disconnect:
                # Driver keeps re-serving the last frame it ever had,
                # forever, instead of failing.
                if self._last_frame is not None:
                    return True, self._last_frame
                return False, None
            if DEVICE.hang_on_disconnect:
                # Simulate a DirectShow-style hang: block until the device
                # comes back (or this capture is released out from under us).
                while not DEVICE.present_event.wait(timeout=0.05):
                    if self._released:
                        return False, None
                if self._released:
                    return False, None
            else:
                return False, None

        if not self._opened:
            return False, None

        if time.monotonic() - self._created_at < DEVICE.warmup_seconds:
            return False, None

        DEVICE.frame_counter += 1
        frame = np.full((4, 4, 3), DEVICE.frame_counter % 256, dtype=np.uint8)
        self._last_frame = frame
        return True, frame

    def release(self):
        self._released = True
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
        DEVICE.present_event.set()
        DEVICE.frame_counter = 0
        DEVICE.hang_on_disconnect = False
        DEVICE.freeze_on_disconnect = False
        DEVICE.warmup_seconds = 0.0
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

    def test_recovers_from_fast_failing_disconnect(self):
        DEVICE.hang_on_disconnect = False
        DEVICE.set_present(False)
        self.assertTrue(wait_until(lambda: not self.cam.is_connected(), timeout=2.0))

        DEVICE.set_present(True)
        self.assertTrue(
            wait_until(lambda: self.cam.is_connected(), timeout=3.0),
            "camera did not recover after replug (fast-fail read)",
        )

    def test_recovers_from_hanging_read(self):
        """
        Reproduces the real-world symptom: read() blocks instead of
        returning False once the device disappears. This is the case the
        old single-thread implementation could NOT recover from, since it
        never got control back from the blocked read() call.
        """
        DEVICE.hang_on_disconnect = True
        DEVICE.set_present(False)

        # The old worker is now stuck inside read(). The supervisor must
        # notice via staleness (not via a returned failure) and mark us
        # disconnected within roughly STALE_TIMEOUT.
        self.assertTrue(
            wait_until(lambda: not self.cam.is_connected(), timeout=2.5),
            "supervisor never detected the stalled/hanging camera",
        )

        # Device comes back while the old worker may still be blocked;
        # a freshly spawned worker (via the supervisor's reconnect
        # attempts) should pick up frames again.
        DEVICE.set_present(True)
        self.assertTrue(
            wait_until(lambda: self.cam.is_connected(), timeout=4.0),
            "camera did not recover after replug (hanging read)",
        )

        last_frame = self.cam.get_frame()
        self.assertTrue(
            wait_until(lambda: not np.array_equal(self.cam.get_frame(), last_frame), timeout=1.0),
            "live feed did not resume producing new frames after recovery",
        )

    def test_manual_reconnect_forces_immediate_retry(self):
        DEVICE.hang_on_disconnect = True
        DEVICE.set_present(False)
        self.assertTrue(wait_until(lambda: not self.cam.is_connected(), timeout=2.5))

        DEVICE.set_present(True)
        self.cam.reconnect()

        self.assertTrue(
            wait_until(lambda: self.cam.is_connected(), timeout=1.5),
            "manual reconnect() did not recover promptly",
        )

    def test_recovers_from_frozen_frame(self):
        """
        Regression test for the real-world symptom in the second log:
        the driver keeps returning ret=True with the exact same buffered
        frame after disconnect, instead of failing or hanging. Content-
        based staleness must catch this even though reads keep "succeeding".
        """
        # Speed up the test by shrinking FROZEN_TIMEOUT for this run only.
        original_frozen_timeout = camera_module.FROZEN_TIMEOUT
        camera_module.FROZEN_TIMEOUT = 0.5
        try:
            DEVICE.freeze_on_disconnect = True
            DEVICE.set_present(False)

            self.assertTrue(
                wait_until(lambda: not self.cam.is_connected(), timeout=2.0),
                "supervisor never detected the frozen (identical) frame feed",
            )

            DEVICE.set_present(True)
            self.assertTrue(
                wait_until(lambda: self.cam.is_connected(), timeout=3.0),
                "camera did not recover after replug (frozen-frame case)",
            )

            last_frame = self.cam.get_frame()
            self.assertTrue(
                wait_until(
                    lambda: not np.array_equal(self.cam.get_frame(), last_frame),
                    timeout=1.0,
                ),
                "live feed did not resume producing new frames after recovery",
            )
        finally:
            camera_module.FROZEN_TIMEOUT = original_frozen_timeout

    def test_survives_realistic_camera_warmup(self):
        """
        Regression test for the startup race: a freshly (re)started worker
        needs a moment before it delivers its first frame (real cameras
        have autoexposure/init delay). The supervisor must NOT treat that
        as a stall and tear the worker down before it gets a chance -
        which previously caused the camera to never stabilize at all.
        """
        self.cam.release()  # tear down the instant-warmup camera from setUp

        DEVICE.warmup_seconds = 0.4  # plausible real-world camera init delay
        DEVICE.frame_counter = 0
        cam = camera_module.Camera(app_state=None)
        try:
            self.assertTrue(
                wait_until(lambda: cam.is_connected(), timeout=3.0),
                "camera never became connected despite a normal warm-up delay",
            )
            # Once connected, it must STAY connected - no thrashing/flapping.
            still_connected = True
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if not cam.is_connected():
                    still_connected = False
                    break
                time.sleep(0.02)
            self.assertTrue(
                still_connected,
                "camera flapped disconnected shortly after connecting "
                "(supervisor is thrashing the worker)",
            )
        finally:
            cam.release()

    def test_stays_down_while_unplugged_no_crash(self):
        DEVICE.hang_on_disconnect = True
        DEVICE.set_present(False)
        self.assertTrue(wait_until(lambda: not self.cam.is_connected(), timeout=2.5))
        time.sleep(1.5)  # let a few backoff cycles pass
        self.assertFalse(self.cam.is_connected())


if __name__ == "__main__":
    unittest.main(verbosity=2)