"""Unit tests for the ESP32 serial / Wi-Fi link, using fakes (no hardware)."""

from types import SimpleNamespace
import time
import unittest
from unittest import mock

import motor_interface as mi


def port(device: str, description: str, vid: int | None) -> SimpleNamespace:
    return SimpleNamespace(device=device, description=description, vid=vid)


class FakeSerial:
    """Minimal stand-in for serial.Serial that replays scripted lines."""

    def __init__(self, replies: list[bytes]) -> None:
        self.replies = list(replies)
        self.written: list[bytes] = []
        self.is_open = True

    def readline(self) -> bytes:
        return self.replies.pop(0) if self.replies else b""

    @property
    def in_waiting(self) -> int:
        return len(self.replies)

    def write(self, data: bytes) -> None:
        self.written.append(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False


class FindPortTests(unittest.TestCase):
    def tearDown(self) -> None:
        mi.close_serial()

    def test_picks_esp32_usb_chip_and_skips_bluetooth(self) -> None:
        ports = [
            port("COM3", "Standard Serial over Bluetooth link (COM3)", None),
            port("COM4", "Standard Serial over Bluetooth link (COM4)", None),
            port("COM7", "Silicon Labs CP210x USB to UART Bridge (COM7)", 0x10C4),
        ]
        with mock.patch("serial.tools.list_ports.comports", return_value=ports):
            self.assertEqual(mi.find_esp32_port(), "COM7")

    def test_bluetooth_only_raises_clear_error(self) -> None:
        ports = [port("COM3", "Standard Serial over Bluetooth link (COM3)", None)]
        with mock.patch("serial.tools.list_ports.comports", return_value=ports):
            with self.assertRaisesRegex(RuntimeError, "No ESP32 USB serial port"):
                mi.find_esp32_port()

    def test_two_boards_require_explicit_port(self) -> None:
        ports = [port("COM5", "CH340 (COM5)", 0x1A86), port("COM7", "CP210x (COM7)", 0x10C4)]
        with mock.patch("serial.tools.list_ports.comports", return_value=ports):
            with self.assertRaisesRegex(RuntimeError, "Several"):
                mi.find_esp32_port()


class ConnectTests(unittest.TestCase):
    def tearDown(self) -> None:
        mi.close_serial()

    def test_waits_for_ready_then_sends_newline_commands(self) -> None:
        fake = FakeSerial([b"boot noise\r\n", b"READY stepper jog controller\r\n"])
        with mock.patch("serial.Serial", return_value=fake):
            self.assertEqual(mi.connect("COM7"), "COM7")
        mi.send_motor_command(-2.5, 4, simulation_mode=False)
        self.assertEqual(fake.written[-1], b"PAN:-2.50,TILT:4.00\n")

    def test_board_that_never_answers_raises_and_closes(self) -> None:
        fake = FakeSerial([])
        with mock.patch("serial.Serial", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "did not answer"):
                mi.connect("COM7", ready_timeout_s=0.05)
        self.assertFalse(fake.is_open)

    def test_read_messages_returns_firmware_lines(self) -> None:
        fake = FakeSerial([b"READY\r\n"])
        with mock.patch("serial.Serial", return_value=fake):
            mi.connect("COM7")
        fake.replies = [b"LIMIT pan reached\r\n", b"\r\n", b"POS pan 45.0 deg\r\n"]
        self.assertEqual(mi.read_messages(), ["LIMIT pan reached", "POS pan 45.0 deg"])

    def test_simulation_never_opens_a_port(self) -> None:
        with mock.patch("serial.Serial") as opened:
            mi.send_motor_command(1, 1, simulation_mode=True, log_simulation=False)
        opened.assert_not_called()


class WifiConnectTests(unittest.TestCase):
    def tearDown(self) -> None:
        mi.close_serial()

    def test_wifi_connect_sends_pan_tilt_over_websocket(self) -> None:
        class FakeWs:
            def __init__(self, *args, **kwargs) -> None:
                self.sent: list[str] = []
                self._closed = False

            def connect(self, url: str, timeout: float = 5.0) -> None:
                self.url = url

            def settimeout(self, _: float) -> None:
                pass

            def recv(self) -> str:
                time.sleep(0.02)
                if self._closed:
                    raise OSError("closed")
                return "STATUS stopped 30 stopped 30 off"

            def send(self, payload: str) -> None:
                self.sent.append(payload)

            def close(self) -> None:
                self._closed = True

        fake = FakeWs()
        with mock.patch.object(mi, "_import_websocket") as importer:
            importer.return_value = SimpleNamespace(
                WebSocket=lambda enable_multithread=True: fake
            )
            label = mi.connect(link="wifi", wifi_host="192.168.4.1", ready_timeout_s=1.0)
        self.assertEqual(label, "wifi:192.168.4.1")
        mi.send_motor_command(-1.25, 2.5, simulation_mode=False)
        self.assertIn("PAN:-1.25,TILT:2.50", fake.sent)


if __name__ == "__main__":
    unittest.main()
