# Merge Segmentation Implementation Plan

## Context

After positioning a graft volume with the SpaceMouse and hardening its transform, the user needs to merge the graft's labelmap voxels into a host labelmap. This is the final step in the workflow: **position → harden → merge**. The merge resamples the graft onto the host's voxel grid (nearest-neighbor), overwrites non-zero voxels, and removes the graft from the Slicer scene. All resampling runs Slicer-side via SimpleITK — greifer only sends a signal.

---

## Key Design Decisions

### Host volume via `--host-volume` CLI argument

The host labelmap is fixed for the session. A new `--host-volume NAME` argument identifies it. When omitted, the merge button is disabled (greyed out) and merge commands log a warning.

### Graft = active target

The graft is always `target_manager.active_name`. No extra selection needed — the user positions it, so it's already the active target.

### Command format: `("merge", None)`

Same 2-tuple pattern as harden. The stream loop reads the active target name and host volume from its own scope.

### Message format: `StringMessage("graft\nhost", device_name="GreiferMerge")`

Both names in a single newline-delimited payload. One message, one atomic operation on the Slicer side.

### Auto-harden before merge

The Slicer merge handler checks if the graft still has a parent transform and hardens it automatically. This handles the case where the user clicks Merge without clicking Harden first.

### Merge button in existing HardenPanel

Add a "Merge" button below the existing "Harden" button under the same "Transform" header. Keep `self._btn` for the harden button (existing tests reference it), add `self._merge_btn` for merge.

### No greifer-side state change after merge

TargetManager keeps the target. The Slicer volume is gone but the transform node persists (harmless — SpaceMouse input just sends transforms to a node with nothing under it).

---

## Changes By File

### Modified: `src/greifer/app.py`

**New CLI argument** in `_parse_args()`:

```python
parser.add_argument(
    "--host-volume",
    default=None,
    metavar="NAME",
    help=(
        "Slicer labelmap volume to merge grafts into. "
        "Required for the merge command to work."
    ),
)
```

**New `host_volume` parameter** on `_stream_loop`:

```python
def _stream_loop(
    ...,
    host_volume: str | None = None,
) -> None:
```

**New command handler** (after `"harden"` elif):

```python
elif cmd[0] == "merge":
    if host_volume is None:
        log.warning("Merge ignored — no --host-volume specified")
    else:
        graft_name = target_manager.active_name
        client.send_message(
            pyigtl.StringMessage(
                string=f"{graft_name}\n{host_volume}",
                device_name="GreiferMerge",
            )
        )
        log.info("Merge requested: %s -> %s", graft_name, host_volume)
```

**`main()` plumbing**: read `args.host_volume`, add to settings table, pass to `_stream_loop` and `run_visualization` (via kwargs).

### Modified: `src/greifer/visualization.py`

**`HardenPanel`** — add `host_volume` param and merge button:

```python
class HardenPanel(QWidget):
    """Buttons to harden and merge the active target's transform."""

    def __init__(
        self,
        cmd_queue: Queue[Command],
        host_volume: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cmd_queue = cmd_queue
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QLabel("Transform")
        header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(header)

        self._btn = QPushButton("Harden")
        self._btn.setToolTip(
            "Bake the current transform into the volume and reset to identity."
        )
        self._btn.clicked.connect(self._on_harden)
        layout.addWidget(self._btn)

        self._merge_btn = QPushButton("Merge")
        self._merge_btn.setToolTip(
            "Merge the graft into the host volume and remove the graft from the scene."
        )
        self._merge_btn.clicked.connect(self._on_merge)
        self._merge_btn.setEnabled(host_volume is not None)
        layout.addWidget(self._merge_btn)

    def _on_harden(self) -> None:
        try:
            self._cmd_queue.put_nowait(("harden", None))
        except (Full, BrokenPipeError, OSError):
            pass

    def _on_merge(self) -> None:
        try:
            self._cmd_queue.put_nowait(("merge", None))
        except (Full, BrokenPipeError, OSError):
            pass
```

**`_build_right_panel`** — add `host_volume` param, pass to `HardenPanel`:

```python
def _build_right_panel(
    cmd_queue, sensitivity, target_names,
    host_volume: str | None = None,
) -> tuple[QWidget, DofLockPanel]:
    ...
    layout.addWidget(HardenPanel(cmd_queue, host_volume=host_volume))
    ...
```

**`run_visualization`** — add `host_volume` param, pass to `_build_right_panel`.

**`Command` type** — already covers `tuple[str, None]`, no change needed.

### Modified: `slicer/greifer_harden_listener.py`

Refactor existing `_on_node_added` into a dispatcher that calls `_handle_harden` (existing logic, extracted) or `_handle_merge` (new):

```python
@vtk.calldata_type(vtk.VTK_OBJECT)
def _on_node_added(caller, event, calldata):
    node = calldata
    if node is None or not node.IsA("vtkMRMLTextNode"):
        return
    name = node.GetName()
    if name == "GreiferHarden":
        _handle_harden(node)
    elif name == "GreiferMerge":
        _handle_merge(node)
```

**`_handle_merge`** — the core merge logic:

1. Parse `"graft_name\nhost_name"` from text node
2. Find both `vtkMRMLLabelMapVolumeNode` by name
3. Auto-harden graft if it still has a parent transform
4. `sitkUtils.PullVolumeFromSlicer` → SimpleITK images
5. `sitk.Resample(graft, host, identity, sitkNearestNeighbor, 0, graft.GetPixelID())` — resample graft onto host grid
6. Overwrite: `host_arr[graft_arr != 0] = graft_arr[graft_arr != 0]`
7. `sitkUtils.PushVolumeToSlicer(result, host_node)` — update host in-place
8. `slicer.mrmlScene.RemoveNode(graft_node)` — delete graft from scene
9. Cleanup text node

Update module docstring to reflect both commands. Import `SimpleITK`/`sitkUtils` with `_HAS_SITK` guard.

---

## Tests

**`tests/test_integration.py`** — `TestMergeCommand`:
- `test_merge_sends_string_message` — verifies `StringMessage` with `"Graft\nSkull"` payload, device_name `"GreiferMerge"`
- `test_merge_payload_contains_active_and_host` — switch to B, merge → payload is `"B\nHost"`
- `test_merge_without_host_volume_logs_warning` — `host_volume=None` → warning logged, no messages sent
- `test_merge_does_not_change_target_manager_state` — accumulator matrix unchanged after merge

**`tests/test_visualization.py`** — `TestMergeButton`:
- `test_merge_button_exists` — button text is "Merge"
- `test_merge_button_enabled_with_host` — enabled when `host_volume="Skull"`
- `test_merge_button_disabled_without_host` — disabled when `host_volume=None`
- `test_merge_button_disabled_by_default` — disabled when no kwarg
- `test_clicking_merge_enqueues_command` — click → `("merge", None)` in queue
- `test_harden_button_still_works` — adding merge doesn't break harden
- `test_full_queue_does_not_crash_merge`

Existing `TestHardenPanel` tests pass unchanged — they reference `panel._btn` which remains the harden button.

---

## Implementation Order

| Step | Files | Depends on | Notes |
|------|-------|------------|-------|
| 1 | `app.py` | — | CLI arg + `host_volume` param + merge handler + main() plumbing |
| 2 | `test_integration.py` | Step 1 | `TestMergeCommand` (4 tests) |
| 3 | `visualization.py` | — | Merge button in HardenPanel + signature threading |
| 4 | `test_visualization.py` | Step 3 | `TestMergeButton` (7 tests) |
| 5 | `slicer/greifer_harden_listener.py` | — | Refactor dispatch + `_handle_merge` |
| 6 | Run `pytest` | All | Full suite green |

---

## Verification

1. `pytest` — all existing + new tests pass
2. `greifer --targets Graft --host-volume Skull` — settings panel shows host volume
3. Merge button is enabled; without `--host-volume` it is greyed out
4. Position graft → Harden → Merge
5. Slicer console: `[greifer] Merged 'Graft' into 'Skull' (N voxels overwritten)`
6. Graft disappears from scene; host contains graft labels at positioned location
7. Skip Harden, go directly to Merge — auto-harden kicks in

---

## Out of Scope

- **Undo merge**: The previous host state is lost once overwritten
- **Merge confirmation dialog**: Button press is immediate
- **Bidirectional acknowledgment**: Slicer does not send success/failure back
- **Union/additive merge mode**: Only replace/overwrite is supported
- **SegmentationNode support**: Only LabelMapVolumeNode; segmentations need conversion first
- **Batch merge**: Merging multiple grafts at once
