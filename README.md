# greifer

Synopsis: A compatibility layer to connect 3Dconnexion SpaceMouse devices to 3D Slicer via OpenIGTLink.

Problem Statement: 3Dconnexion's SpaceMouse devices are human interface devices (HID) that provide 6-DOF input  widely used in 3D applications. However, they lack native support for 3D Slicer, an open-source platform for medical image computing. This project aims to bridge that gap by creating a Python-based compatibility layer that translates SpaceMouse inputs into OpenIGTLink messages that Slicer can understand.

## Installation

```bash
git clone https://github.com/stebix/greifer.git
cd greifer
uv pip install -e .
```

## Usage

Inside 3DSlicer, we first require the additional modules:
- OpenIGTLinkIF -> https://github.com/openigtlink/SlicerOpenIGTLink
- SlicerIGT -> https://github.com/SlicerIGT/SlicerIGT
The tools can also be installed from the Slicer Extension Manager.

Then, we need to create a new "Linear Transform" node in Slicer and name it "SpaceMouseTransform" (or change the `DEVICE_NAME` variable in `greifer/__init__.py` to match your chosen name). This node will receive the transformations from the SpaceMouse.

Finally, run the `greifer` module:

```bash
python -m greifer
```
