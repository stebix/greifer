import multiprocessing
import queue as queue_module
import time

from multiprocessing.queues import Queue

import pyspacemouse
import pyigtl

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from greifer.transform import TransformAccumulator, compute_increments


# ── Configuration ──────────────────────────────────────────────
SLICER_HOST: str = "127.0.0.1"
SLICER_PORT: int = 18944
DEVICE_NAME: str = "SpaceMouseTransform"  # Must match your Slicer transform node name

# Sensitivity tuning — adjust these to taste
TRANS_SCALE: float = 0.005    # mm per unit of SpaceMouse translation
ROT_SCALE: float = 0.001    # radians per unit of SpaceMouse rotation

UPDATE_HZ: int = 500     # How often to push updates
REORTHOGONALIZE_INTERVAL: int = 1000  # Re-orthogonalize rotation matrix every N iterations

ENABLE_VISUALIZATION: bool = True

VIS_HZ: int = 60                           # Visualization target rate
VIS_STRIDE: int = UPDATE_HZ // VIS_HZ      # Enqueue every Nth sample (=8)
# ───────────────────────────────────────────────────────────────




def _stream_loop(
    device: pyspacemouse.SpaceMouseDevice,
    client: pyigtl.OpenIGTLinkClient,
    vis_queue: Queue | None,
    dt: float,
    vis_stride: int = 1,
) -> None:
    """Hot path: read SpaceMouse, compute transform, send to Slicer."""
    accumulator = TransformAccumulator(
        reorthogonalize_interval=REORTHOGONALIZE_INTERVAL
    )
    next_tick = time.monotonic()
    vis_counter = 0

    while True:
        state = device.read()

        if vis_queue is not None:
            vis_counter += 1
            if vis_counter >= vis_stride:
                vis_counter = 0
                try:
                    vis_queue.put_nowait((
                        state.t,
                        state.x, state.y, state.z,
                        state.roll, state.pitch,
                        state.yaw,
                    ))
                except queue_module.Full:
                    pass

        increments = compute_increments(
            state.x, state.y, state.z,
            state.roll, state.pitch, state.yaw,
            trans_scale=TRANS_SCALE,
            rot_scale=ROT_SCALE,
        )

        if increments is not None:
            matrix = accumulator.update(*increments)

            transform_msg = pyigtl.TransformMessage(
                matrix, device_name=DEVICE_NAME
            )
            client.send_message(transform_msg)

        next_tick += dt
        sleep_remaining = next_tick - time.monotonic()
        if sleep_remaining > 0:
            time.sleep(sleep_remaining)
        else:
            next_tick = time.monotonic()


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
        f"enabled (1:{VIS_STRIDE} decimation)" if ENABLE_VISUALIZATION else "disabled",
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
                _stream_loop(device, client, vis_queue, dt, vis_stride=VIS_STRIDE)

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
