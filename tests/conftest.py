"""Shared test fixtures — fakes for hardware and network boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field


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

    def send_message(self, msg: object) -> None:
        self.messages.append(msg)
