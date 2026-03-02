"""Multi-target management for addressing multiple 3D Slicer nodes."""

from __future__ import annotations

from collections.abc import Sequence

from numpy import ndarray

from greifer.transform import TransformAccumulator


class TargetManager:
    """Manages multiple named targets, each with independent transform state.

    The first name in *names* becomes the initial active target.
    """

    def __init__(
        self,
        names: Sequence[str],
        *,
        reorthogonalize_interval: int = 1000,
    ) -> None:
        if not names:
            raise ValueError("At least one target name is required")
        self._reorth_interval = reorthogonalize_interval
        self._targets: dict[str, TransformAccumulator] = {
            name: TransformAccumulator(reorthogonalize_interval)
            for name in names
        }
        self._active: str = names[0]

    @property
    def active_name(self) -> str:
        return self._active

    @property
    def active_accumulator(self) -> TransformAccumulator:
        return self._targets[self._active]

    @property
    def names(self) -> list[str]:
        """All registered target names in insertion order."""
        return list(self._targets)

    def switch_to(self, name: str) -> ndarray:
        """Switch active target. Returns the current matrix of the new target.

        Raises ``KeyError`` if *name* is not registered.
        """
        if name not in self._targets:
            raise KeyError(f"Unknown target: {name!r}")
        self._active = name
        return self._targets[name].matrix

    def add_target(self, name: str) -> None:
        """Register a new target (idempotent — no-op if already present)."""
        if name not in self._targets:
            self._targets[name] = TransformAccumulator(self._reorth_interval)

    def update(
        self,
        dx: float,
        dy: float,
        dz: float,
        rx: float,
        ry: float,
        rz: float,
    ) -> ndarray:
        """Apply an incremental transform to the active target.

        Returns the cumulative matrix.
        """
        return self._targets[self._active].update(dx, dy, dz, rx, ry, rz)
