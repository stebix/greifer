import time
import numpy as np

import pyspacemouse
import pyigtl

from greifer.math import build_rotation_matrix


# ── Configuration ──────────────────────────────────────────────
SLICER_HOST: str = "127.0.0.1"
SLICER_PORT: int = 18944
DEVICE_NAME: str = "SpaceMouseTransform"  # Must match your Slicer transform node name

# Sensitivity tuning — adjust these to taste
TRANS_SCALE: float = 0.005    # mm per unit of SpaceMouse translation
ROT_SCALE: float = 0.001    # radians per unit of SpaceMouse rotation

UPDATE_HZ: int = 60     # How often to push updates
# ───────────────────────────────────────────────────────────────




def main() -> None:
    print('Hello from greifer!')

    print(
        pyspacemouse.get_connected_devices(),
    )


    print(f'Connecting to Slicer at {SLICER_HOST}:{SLICER_PORT} ...')
    client = pyigtl.OpenIGTLinkClient(host=SLICER_HOST, port=SLICER_PORT)
    time.sleep(1)  # Give connection time to establish
    print('Connected.\n')


    # basic transformation
    T_cumulative = np.eye(4)
    dt = 1.0 / UPDATE_HZ

    with pyspacemouse.open() as device:
        print('Device opened successfully!')
        while True:
            state = device.read()

            dx = state.x * TRANS_SCALE
            dy = state.y * TRANS_SCALE
            dz = state.z * TRANS_SCALE

            rx = state.roll * ROT_SCALE
            ry = state.pitch * ROT_SCALE
            rz = state.yaw * ROT_SCALE

            total_motion = abs(dx) + abs(dy) + abs(dz) + abs(rx) + abs(ry) + abs(rz)

            if total_motion < 1e-6:
                time.sleep(dt)
                continue

            dT = np.eye(4)

            translation = [dx, dy, dz]
            dT[:3, 3] = translation
            dR = build_rotation_matrix(rx, ry, rz)
            dT[:3, :3] = dR

            T_cumulative = dT @ T_cumulative

            transform_msg = pyigtl.TransformMessage(T_cumulative, device_name=DEVICE_NAME)
            client.send_message(transform_msg)
           
            time.sleep(dt)



if __name__ == '__main__':
    main()