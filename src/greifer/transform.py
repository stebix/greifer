"""Pure transform computation extracted from the streaming loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy import ndarray

from greifer.math import build_rotation_matrix


MOTION_THRESHOLD: float = 1e-6


@dataclass(frozen=True, slots=True)
class Sensitivity:
    """Per-axis scale factors for the 6 degrees of freedom."""

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

    @classmethod
    def uniform(cls, trans: float, rot: float) -> Sensitivity:
        """All translation axes share *trans*; all rotation axes share *rot*."""
        return cls(x=trans, y=trans, z=trans, roll=rot, pitch=rot, yaw=rot)

    def for_axis(self, axis: Axis) -> float:
        """Look up the scale factor for a given Axis enum member."""
        return getattr(self, axis.value)


def compute_increments(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    sensitivity: Sensitivity,
) -> tuple[float, float, float, float, float, float] | None:
    """Scale raw device state and apply motion threshold.

    Returns the scaled (dx, dy, dz, rx, ry, rz) tuple, or None if the
    total motion falls below the threshold.
    """
    dx = x * sensitivity.x
    dy = y * sensitivity.y
    dz = z * sensitivity.z

    rx = roll * sensitivity.roll
    ry = pitch * sensitivity.pitch
    rz = yaw * sensitivity.yaw

    total_motion = (
        abs(dx) + abs(dy) + abs(dz)
        + abs(rx) + abs(ry) + abs(rz)
    )

    if total_motion < MOTION_THRESHOLD:
        return None

    return (dx, dy, dz, rx, ry, rz)


class Axis(StrEnum):
    """Degrees of freedom that can be individually locked."""

    X = "x"
    Y = "y"
    Z = "z"
    ROLL = "roll"
    PITCH = "pitch"
    YAW = "yaw"

    @property
    def index(self) -> int:
        """Positional index in the (x, y, z, roll, pitch, yaw) tuple."""
        return _AXIS_ORDER.index(self)


_AXIS_ORDER: tuple[Axis, ...] = tuple(Axis)


class DofLockFilter:
    """Zeros locked axes in a 6-DOF increment tuple.

    Sits between ``compute_increments`` and ``TransformAccumulator.update``
    in the pipeline, leaving both untouched.
    """

    def __init__(self, locked: set[Axis] | None = None) -> None:
        self._locked: set[Axis] = set(locked) if locked else set()

    def lock(self, *axes: Axis) -> None:
        """Lock one or more axes (idempotent)."""
        self._locked.update(axes)

    def unlock(self, *axes: Axis) -> None:
        """Unlock one or more axes (idempotent)."""
        self._locked.difference_update(axes)

    def toggle(self, axis: Axis) -> None:
        """Toggle a single axis lock."""
        self._locked.symmetric_difference_update({axis})

    @property
    def locked(self) -> frozenset[Axis]:
        """Read-only snapshot of currently locked axes."""
        return frozenset(self._locked)

    def apply(
        self,
        dx: float,
        dy: float,
        dz: float,
        rx: float,
        ry: float,
        rz: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Return the increment tuple with locked axes zeroed."""
        vals = [dx, dy, dz, rx, ry, rz]
        for axis in self._locked:
            vals[axis.index] = 0.0
        return (vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])


class TransformAccumulator:
    """Accumulates incremental 4x4 transforms with periodic reorthogonalization.

    Pure computation — no I/O, no side effects beyond internal state.
    """

    def __init__(self, reorthogonalize_interval: int = 1000) -> None:
        self.matrix: ndarray = np.eye(4)
        self._iteration: int = 0
        self._reorth_interval = reorthogonalize_interval

    @property
    def rotation(self) -> ndarray:
        """The 3x3 rotation submatrix."""
        return self.matrix[:3, :3]

    @property
    def translation(self) -> ndarray:
        """The translation vector."""
        return self.matrix[:3, 3]

    def update(
        self,
        dx: float,
        dy: float,
        dz: float,
        rx: float,
        ry: float,
        rz: float,
    ) -> ndarray:
        """Apply an incremental transform. Returns the cumulative matrix."""
        dT = np.eye(4)
        dT[:3, 3] = [dx, dy, dz]
        dT[:3, :3] = build_rotation_matrix(rx, ry, rz)

        self.matrix = dT @ self.matrix
        self._iteration += 1

        if self._iteration % self._reorth_interval == 0:
            self.reorthogonalize()

        return self.matrix

    def reorthogonalize(self) -> None:
        """SVD polar-factor correction on the rotation submatrix."""
        U, _, Vt = np.linalg.svd(self.matrix[:3, :3])
        self.matrix[:3, :3] = U @ Vt
