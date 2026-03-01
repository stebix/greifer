"""Shared test fixtures — fakes for hardware and network boundaries."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class StopStreaming(Exception):
    """Raised by FakeDevice to terminate _stream_loop after exhausting states."""


@dataclass
class FakeState:
    """Mimics the pyspacemouse state object returned by device.read()."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    t: float = 0.0


class FakeDevice:
    """Yields a scripted sequence of states, then raises StopStreaming.

    Usage::

        device = FakeDevice([
            FakeState(x=100),
            FakeState(y=200),
        ])
        # device.read() → FakeState(x=100)
        # device.read() → FakeState(y=200)
        # device.read() → raises StopStreaming
    """

    def __init__(self, states: list[FakeState]) -> None:
        self._states = iter(states)

    def read(self) -> FakeState:
        try:
            return next(self._states)
        except StopIteration:
            raise StopStreaming("FakeDevice exhausted")


class FakeClient:
    """Records all messages passed to send_message().

    Usage::

        client = FakeClient()
        client.send_message(msg)
        assert len(client.messages) == 1
    """

    messages: list = field(default_factory=list)

    def __init__(self) -> None:
        self.messages: list = []

    def is_connected(self) -> bool:
        return True

    def send_message(self, msg: object) -> None:
        self.messages.append(msg)


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped QApplication for widget tests (offscreen)."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return app
