"""Pure transform computation extracted from the streaming loop."""

import numpy as np
from numpy import ndarray

from greifer.math import build_rotation_matrix


MOTION_THRESHOLD: float = 1e-6


def compute_increments(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    trans_scale: float,
    rot_scale: float,
) -> tuple[float, float, float, float, float, float] | None:
    """Scale raw device state and apply motion threshold.

    Returns the scaled (dx, dy, dz, rx, ry, rz) tuple, or None if the
    total motion falls below the threshold.
    """
    dx = x * trans_scale
    dy = y * trans_scale
    dz = z * trans_scale

    rx = roll * rot_scale
    ry = pitch * rot_scale
    rz = yaw * rot_scale

    total_motion = (
        abs(dx) + abs(dy) + abs(dz)
        + abs(rx) + abs(ry) + abs(rz)
    )

    if total_motion < MOTION_THRESHOLD:
        return None

    return (dx, dy, dz, rx, ry, rz)


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
