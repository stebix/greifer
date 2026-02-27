import time
import numpy as np

import pyspacemouse
import pyigtl


# ── Configuration ──────────────────────────────────────────────
SLICER_HOST = "127.0.0.1"
SLICER_PORT = 18944
DEVICE_NAME = "SpaceMouseTransform"  # Must match your Slicer transform node name

# Sensitivity tuning — adjust these to taste
TRANS_SCALE = 0.005    # mm per unit of SpaceMouse translation
ROT_SCALE   = 0.002  # radians per unit of SpaceMouse rotation

UPDATE_HZ   = 60     # How often to push updates
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

            if abs(state.x) < 1e-6:
                time.sleep(dt)
                continue

            dT = np.eye(4)

            translation = [state.x * TRANS_SCALE, 0.0, 0.0]
            dT[:3, 3] = translation

            T_cumulative = dT @ T_cumulative

            transform_msg = pyigtl.TransformMessage(T_cumulative, device_name=DEVICE_NAME)
            client.send_message(transform_msg)
           
            time.sleep(dt)



if __name__ == '__main__':
    main()