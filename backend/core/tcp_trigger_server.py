"""Standalone TCP trigger server for PLC-driven vision cycles.

This module is intentionally self-contained so it can be wired into an
application without changing existing backend files. It listens for incoming
trigger bytes from a PLC, runs the provided vision function, and sends back
OK/NOK responses based on the result.
"""

import socket
import threading
import time

from backend.core.config_loader import cfg
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class TCPTriggerServer:
    """
    TCP server that listens for triggers from external devices (e.g., PLCs).
    
    When a trigger byte is received, it captures a frame, runs vision 
    processing, and returns an OK/NOK response.
    """
    def __init__(self, camera, run_vision_fn, process_result_fn):
        self.camera = camera
        self.run_vision_fn = run_vision_fn
        self.process_result_fn = process_result_fn

        trig_cfg = cfg.get("trigger", {})
        self.host = trig_cfg.get("host", "0.0.0.0")
        self.port = trig_cfg.get("port", 5001)
        self.timeout_s = trig_cfg.get("timeout_s", 5.0)
        self.enabled = trig_cfg.get("enabled", False)
        self.trigger_byte = int(trig_cfg.get("trigger_byte", "0x01"), 16)
        self.response_ok = trig_cfg.get("response_ok", "OK\n").encode()
        self.response_nok = trig_cfg.get("response_nok", "NOK\n").encode()

        self._server_thread = None
        self._stop_event = threading.Event()

    def start(self):
        """
        Starts the TCP server in a background thread if enabled in configuration.
        """
        if not self.enabled:
            logger.info("TCP trigger disabled, skipping")
            return

        self._stop_event = threading.Event()
        self._server_thread = threading.Thread(target=self._serve, daemon=True)
        self._server_thread.start()
        logger.info("TCP trigger server started on %s:%s", self.host, self.port)

    def stop(self):
        """
        Signals the server thread to stop and closes the server.
        """
        self._stop_event.set()
        logger.info("TCP trigger server stopped")

    def _serve(self):
        """
        Main server loop that accepts incoming TCP connections.
        """
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((self.host, self.port))
            server_sock.listen(1)
            server_sock.settimeout(1.0)

            while not self._stop_event.is_set():
                try:
                    conn, addr = server_sock.accept()
                    logger.info("PLC connected: %s", addr)
                    threading.Thread(
                        target=self._handle_connection,
                        args=(conn, addr),
                        daemon=True,
                    ).start()
                except socket.timeout:
                    continue
                except Exception as exc:
                    if not self._stop_event.is_set():
                        logger.error("TCP accept error: %s", exc)
        finally:
            try:
                server_sock.close()
            except Exception:
                pass

    def _handle_connection(self, conn, addr):
        """
        Handles an individual PLC connection, listening for trigger bytes.
        """
        try:
            conn.settimeout(None)
            while True:
                try:
                    data = conn.recv(1)
                except Exception:
                    break

                if not data:
                    break

                received_byte = data[0]
                logger.debug("Trigger received: byte=0x%02X from %s", received_byte, addr)

                if received_byte != self.trigger_byte:
                    logger.warning(
                        "Unexpected trigger byte 0x%02X (expected 0x%02X), triggering anyway",
                        received_byte,
                        self.trigger_byte,
                    )

                response = self._trigger_vision_and_wait()
                try:
                    conn.sendall(response)
                    logger.debug("Response sent: %s to %s", response.decode().strip(), addr)
                except Exception as send_exc:
                    logger.error("Failed to send response to %s: %s", addr, send_exc)
                    break
        finally:
            try:
                conn.close()
            finally:
                logger.info("PLC disconnected: %s", addr)

    def _build_response(self, result: dict) -> bytes:
        """
        Builds the byte response sent back to the PLC for a completed vision
        cycle.

        For "ocr" and "object_detection" modes this is the configured OK/NOK
        byte string. For "ocread" no OK/NOK judgement is made here at all -
        the raw recognized text is sent instead (newline-terminated) so the
        PLC can run its own comparison against the expected value.
        """
        if result.get("mode") == "ocread" and not result.get("error"):
            text = result.get("text")
            if not text:
                # Fall back to reconstructing from detections if "text" is
                # missing for some reason (e.g. a non-standard OCR engine).
                text = " ".join(
                    str(d.get("text", ""))
                    for d in result.get("detections", [])
                    if isinstance(d, dict)
                ).strip()
            return (str(text) + "\n").encode("utf-8", errors="replace")

        return self.response_ok if result.get("status") == "OK" else self.response_nok

    def _trigger_vision_and_wait(self) -> bytes:
        """
        Triggers vision processing and waits for the result to return a response.
        A missing camera frame is treated as NOK for the PLC, but is recorded
        separately as a camera-unavailable failure rather than being passed into
        the vision pipeline.
        """
        frame = self.camera.get_frame()
        trigger_time = time.perf_counter()

        if frame is None:
            logger.warning(
                "No frame available for TCP trigger; recording camera-unavailable NOK"
            )

            result = {
                "status": "NOK",
                "confidence": 0.0,
                "detections": [],
                "failure_reason": "camera_unavailable",
            }

            try:
                self.process_result_fn(result, trigger_time)
            except Exception as exc:
                logger.error(
                    "Failed to process camera-unavailable NOK result: %s",
                    exc,
                    exc_info=True,
                )

            return self.response_nok

        result_holder = []
        done_event = threading.Event()

        def callback(result):
            result_holder.append(result)
            done_event.set()

        try:
            self.run_vision_fn(frame, callback=callback)
        except Exception as exc:
            logger.error(
                "Failed to start vision processing for TCP trigger: %s",
                exc,
                exc_info=True,
            )

            result = {
                "status": "NOK",
                "confidence": 0.0,
                "detections": [],
                "failure_reason": "vision_start_failed",
            }

            try:
                self.process_result_fn(result, trigger_time)
            except Exception as process_exc:
                logger.error(
                    "Failed to process vision-start-failed NOK result: %s",
                    process_exc,
                    exc_info=True,
                )

            return self.response_nok

        done_event.wait(timeout=self.timeout_s)

        if not result_holder:
            logger.warning(
                "Vision timed out after %ss; recording timeout NOK",
                self.timeout_s,
            )

            result = {
                "status": "NOK",
                "confidence": 0.0,
                "detections": [],
                "failure_reason": "vision_timeout",
            }

            try:
                self.process_result_fn(result, trigger_time)
            except Exception as exc:
                logger.error(
                    "Failed to process timeout NOK result: %s",
                    exc,
                    exc_info=True,
                )

            return self.response_nok

        result = result_holder[0]

        self.process_result_fn(result, trigger_time)

        return self._build_response(result)

    def is_enabled(self) -> bool:
        return self.enabled

