# Harden Transform Implementation Plan

## Goal

Add a "Harden Transform" feature that bakes the active target's accumulated rigid transform into Slicer's volume spatial metadata (origin + direction cosines), then resets the transform node to identity. Since greifer only produces rigid transforms, no voxel resampling is needed — the operation is a header-only update on the Slicer side.

The feature spans two systems: **greifer** (signal + reset) and **3D Slicer** (apply transform to volume header). Communication uses a `pyigtl.StringMessage` carrying the target name, followed by an identity `TransformMessage` to visually reset the transform node.

---

## Key Design Decisions

### Command format: `("harden", None)`

Existing commands are all 2-tuples: `("toggle", Axis)`, `("sensitivity", Sensitivity)`, `("switch_target", str)`. The visualization process does not know the active target name, so we use `("harden", None)` as a 2-tuple with a `None` payload. This keeps command destructuring uniform. The stream loop ignores `cmd[1]` for harden commands and reads `target_manager.active_name` directly — which is the authoritative source of truth.

### Message ordering: StringMessage → reset → identity TransformMessage

The StringMessage tells Slicer "bake the current transform into the volume now." The identity TransformMessage resets the visual transform node. If these arrive in the wrong order, Slicer would bake an identity transform (no-op). The stream loop must:
1. Send `StringMessage(string=target_name, device_name="GreiferHarden")`
2. Reset the accumulator via `target_manager.harden_active()`
3. Send `TransformMessage(identity, device_name=target_name)`

### Slicer-side script as standalone module

The listener script lives in `slicer/greifer_harden_listener.py` at the repository root — not inside `src/greifer/` since it runs inside 3D Slicer's embedded Python, not greifer's virtualenv. It observes incoming IGTL text nodes and calls `vtkSlicerTransformLogic.hardenTransform()` on volumes under the named transform.

Usage: `Slicer.exe --python-script slicer/greifer_harden_listener.py` or `exec(open(...).read())` in `.slicerrc.py`.

### Button placement in visualization sidebar

A new `HardenPanel` widget between `TargetPanel` and `DofLockPanel`. Follows the existing pattern: self-contained `QWidget` subclass with its own `cmd_queue` interaction.

---

## Changes By File

### Modified: `src/greifer/transform.py`

Add `reset()` method to `TransformAccumulator`:

```python
def reset(self) -> None:
    """Reset the accumulated transform to identity."""
    self.matrix = np.eye(4)
    self._iteration = 0
```

### Modified: `src/greifer/target.py`

Add `harden_active()` method to `TargetManager`:

```python
def harden_active(self) -> None:
    """Reset the active target's accumulator to identity."""
    self._targets[self._active].reset()
```

### Modified: `src/greifer/app.py`

Add harden command handler in `_stream_loop` (after `switch_target` elif, before `except queue_module.Empty`):

```python
elif cmd[0] == "harden":
    name = target_manager.active_name
    # 1. Tell Slicer to bake the transform
    client.send_message(
        pyigtl.StringMessage(string=name, device_name="GreiferHarden")
    )
    # 2. Reset greifer's accumulator
    target_manager.harden_active()
    # 3. Send identity so Slicer's transform node resets
    client.send_message(
        pyigtl.TransformMessage(
            target_manager.active_accumulator.matrix,
            device_name=name,
        )
    )
    log.info("Hardened transform for %s", name)
```

### Modified: `src/greifer/visualization.py`

**Update `Command` type** (line 33):

```python
type Command = (
    tuple[str, Axis]
    | tuple[str, Sensitivity]
    | tuple[str, str]
    | tuple[str, None]       # ("harden", None)
)
```

**New `HardenPanel` widget** (between `TargetPanel` and `DofLockPanel`):

```python
class HardenPanel(QWidget):
    """Button to harden the active target's transform."""

    def __init__(self, cmd_queue: Queue[Command], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cmd_queue = cmd_queue
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QLabel("Transform")
        header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(header)

        self._btn = QPushButton("Harden")
        self._btn.setToolTip("Bake the current transform into the volume and reset to identity.")
        self._btn.clicked.connect(self._on_harden)
        layout.addWidget(self._btn)

    def _on_harden(self) -> None:
        try:
            self._cmd_queue.put_nowait(("harden", None))
        except (Full, BrokenPipeError, OSError):
            pass
```

**Update `_build_right_panel`** to insert `HardenPanel`:

```
Right Panel (fixed width=220px)
├── TargetPanel      (dropdown)
├── HardenPanel      (NEW — "Harden" button)
├── DofLockPanel     (toggle buttons)
└── SensitivityPanel (sliders)
```

### New: `slicer/greifer_harden_listener.py`

Standalone script for 3D Slicer's Python environment:

```python
"""Greifer Harden Listener for 3D Slicer.

Listens for OpenIGTLink StringMessages with device name "GreiferHarden"
and hardens the named transform into all volumes that reference it.

Usage:
    Slicer.exe --python-script greifer_harden_listener.py

Or add to ~/.slicerrc.py:
    exec(open("/path/to/greifer_harden_listener.py").read())
"""
import slicer
import vtk

@vtk.calldata_type(vtk.VTK_OBJECT)
def _on_node_added(caller, event, calldata):
    node = calldata
    if node is None or not node.IsA("vtkMRMLTextNode"):
        return
    if node.GetName() != "GreiferHarden":
        return

    transform_name = node.GetText()
    if not transform_name:
        return

    transform_node = slicer.util.getFirstNodeByName(
        transform_name, className="vtkMRMLLinearTransformNode"
    )
    if transform_node is None:
        print(f"[greifer] Transform node '{transform_name}' not found")
        slicer.mrmlScene.RemoveNode(node)
        return

    # Harden all volumes under this transform
    hardened = 0
    for i in range(slicer.mrmlScene.GetNumberOfNodes()):
        n = slicer.mrmlScene.GetNthNode(i)
        if n and n.IsA("vtkMRMLVolumeNode"):
            if n.GetParentTransformNode() == transform_node:
                logic = slicer.vtkSlicerTransformLogic()
                logic.hardenTransform(n)
                hardened += 1

    if hardened:
        print(f"[greifer] Hardened '{transform_name}' into {hardened} volume(s)")
    else:
        print(f"[greifer] No volumes found under '{transform_name}'")

    slicer.mrmlScene.RemoveNode(node)

def setup_harden_listener():
    tag = slicer.mrmlScene.AddObserver(
        slicer.vtkMRMLScene.NodeAddedEvent, _on_node_added
    )
    print(f"[greifer] Harden listener active (observer tag: {tag})")
    return tag

_greifer_harden_tag = setup_harden_listener()
```

Key: the `@vtk.calldata_type(vtk.VTK_OBJECT)` decorator is required for the `calldata` parameter to work in Slicer's Python observer callbacks.

### Tests

**`tests/test_accumulator.py`** — `TestReset`:
- `test_reset_restores_identity` — update then reset gives identity
- `test_reset_clears_iteration_counter` — `_iteration` goes to 0
- `test_reset_then_update_accumulates_from_identity` — clean slate after reset

**`tests/test_target.py`** — `TestHardenActive`:
- `test_harden_resets_active_to_identity`
- `test_harden_does_not_affect_inactive` — only active target resets
- `test_harden_then_update_accumulates_from_identity`

**`tests/test_integration.py`** — `TestHardenCommand`:
- `test_harden_sends_string_then_identity` — verifies both messages sent in order
- `test_harden_resets_accumulator` — accumulator is identity after harden
- `test_harden_targets_active_only` — inactive targets unaffected

**`tests/test_visualization.py`** — `TestHardenPanel`:
- `test_has_harden_button`
- `test_clicking_enqueues_harden_command` — click → `("harden", None)` in queue
- `test_full_queue_does_not_crash`

---

## Implementation Order

| Step | Files | Depends on | Notes |
|------|-------|------------|-------|
| 1 | `transform.py` + `test_accumulator.py` | — | Leaf: `reset()` method |
| 2 | `target.py` + `test_target.py` | Step 1 | `harden_active()` delegates to `reset()` |
| 3 | `app.py` + `test_integration.py` | Step 2 | Core wiring: cmd handler, StringMessage + identity |
| 4 | `visualization.py` + `test_visualization.py` | Step 3 | `HardenPanel` UI button |
| 5 | `slicer/greifer_harden_listener.py` | — | Independent: runs inside Slicer |

Steps 1–3 are the backend (testable without UI). Step 4 adds the button. Step 5 is the Slicer receiver.

---

## Verification

1. Run `pytest` — all existing + new tests pass
2. Launch greifer with `--targets Femur Tibia`
3. Move SpaceMouse to accumulate a transform on Femur
4. In Slicer (with listener script loaded), verify the volume moves
5. Click "Harden" in the visualization sidebar
6. Verify: Slicer console prints `[greifer] Hardened 'Femur' into 1 volume(s)`
7. Verify: volume position is preserved but the transform node is now identity
8. Move SpaceMouse again — transform accumulates from zero

---

## Out of Scope

- **Per-target harden**: Hardening an inactive target by name. Currently hardens only the active target.
- **Undo harden**: No mechanism to reverse. The previous transform is lost once baked.
- **Harden confirmation dialog**: Button press is immediate — no "are you sure?" prompt.
- **Bidirectional acknowledgment**: Slicer does not send success/failure back. Reset is optimistic.
- **Slicer script packaging**: Standalone file, not a pip-installable Slicer extension.
- **Harden all targets**: Batch-hardening every target at once.
