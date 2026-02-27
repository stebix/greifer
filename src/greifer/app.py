import multiprocessing
import queue as queue_module
import time

from multiprocessing.queues import Queue

import numpy as np

import pyspacemouse
import pyigtl

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from greifer.math import build_rotation_matrix


# ── Configuration ──────────────────────────────────────────────
SLICER_HOST: str = "127.0.0.1"
SLICER_PORT: int = 18944
DEVICE_NAME: str = "SpaceMouseTransform"  # Must match your Slicer transform node name

# Sensitivity tuning — adjust these to taste
TRANS_SCALE: float = 0.005    # mm per unit of SpaceMouse translation
ROT_SCALE: float = 0.001    # radians per unit of SpaceMouse rotation

UPDATE_HZ: int = 500     # How often to push updates

ENABLE_VISUALIZATION: bool = True
# ───────────────────────────────────────────────────────────────




def _stream_loop(
    device: pyspacemouse.SpaceMouseDevice,
    client: pyigtl.OpenIGTLinkClient,
    vis_queue: Queue | None,
    dt: float,
) -> None:
    """Hot path: read SpaceMouse, compute transform, send to Slicer."""
    T_cumulative = np.eye(4)

    while True:
        state = device.read()

        if vis_queue is not None:
            try:
                vis_queue.put_nowait((
                    state.t,
                    state.x, state.y, state.z,
                    state.roll, state.pitch,
                    state.yaw,
                ))
            except queue_module.Full:
                pass

        dx = state.x * TRANS_SCALE
        dy = state.y * TRANS_SCALE
        dz = state.z * TRANS_SCALE

        rx = state.roll * ROT_SCALE
        ry = state.pitch * ROT_SCALE
        rz = state.yaw * ROT_SCALE

        total_motion = (
            abs(dx) + abs(dy) + abs(dz)
            + abs(rx) + abs(ry) + abs(rz)
        )

        if total_motion < 1e-6:
            time.sleep(dt)
            continue

        dT = np.eye(4)
        dT[:3, 3] = [dx, dy, dz]
        dT[:3, :3] = build_rotation_matrix(rx, ry, rz)

        T_cumulative = dT @ T_cumulative

        transform_msg = pyigtl.TransformMessage(
            T_cumulative, device_name=DEVICE_NAME
        )
        client.send_message(transform_msg)

        time.sleep(dt)


def main() -> None:
    console = Console()

    # ── Settings overview panel ───────────────────────────────────
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Update rate", f"{UPDATE_HZ} Hz")
    table.add_row(
        "Sensitivity",
        f"trans {TRANS_SCALE} · rot {ROT_SCALE}",
    )
    table.add_row(
        "Visualization",
        "enabled" if ENABLE_VISUALIZATION else "disabled",
    )
    table.add_row()
    table.add_row("SpaceMouse", DEVICE_NAME)
    table.add_row("3D Slicer", f"{SLICER_HOST}:{SLICER_PORT}")
    console.print(Panel(table, title="greifer", expand=False))

    # ── Connect to 3D Slicer ─────────────────────────────────────
    with console.status(
        f"Connecting to 3D Slicer at {SLICER_HOST}:{SLICER_PORT} …"
    ):
        client = pyigtl.OpenIGTLinkClient(
            host=SLICER_HOST, port=SLICER_PORT
        )
        time.sleep(1)
    console.print("✓ Connected to 3D Slicer", style="green")

    # ── Spawn visualization process ──────────────────────────────
    vis_queue: Queue | None = None
    vis_process: multiprocessing.Process | None = None

    if ENABLE_VISUALIZATION:

        from greifer.visualization import run_visualization
        vis_queue = multiprocessing.Queue(maxsize=600)
        vis_process = multiprocessing.Process(
            target=run_visualization, args=(vis_queue,), daemon=True
        )
        vis_process.start()

    # ── Open SpaceMouse & stream ──────────────────────────────────
    dt = 1.0 / UPDATE_HZ

    with pyspacemouse.open() as device:
        console.print("✓ SpaceMouse opened", style="green")

        try:
            with console.status("Streaming to 3D Slicer …"):
                _stream_loop(device, client, vis_queue, dt)

        except KeyboardInterrupt:
            pass

        finally:
            client.stop()
            if vis_queue is not None:
                vis_queue.cancel_join_thread()
                try:
                    vis_queue.put_nowait(None)
                except (BrokenPipeError, OSError, queue_module.Full):
                    pass
                vis_queue.close()
            if vis_process is not None:
                vis_process.join(timeout=3)
                if vis_process.is_alive():
                    vis_process.terminate()
            console.print("✓ Shut down cleanly.", style="green")
