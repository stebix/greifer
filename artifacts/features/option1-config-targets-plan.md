# Option 1 Implementation Plan: Config-Driven Multi-Target Support

## Goal

Replace the single hardcoded `DEVICE_NAME` with a system that supports multiple named targets, each with independent transform state, and allows the user to switch between them at runtime.

## Key Design Decisions

### Per-target transform state

Each target gets its own `TransformAccumulator`. Switching targets means switching which accumulator receives increments. This preserves the accumulated position of inactive targets — switching from Femur to Tibia and back doesn't lose Femur's position.

### TargetManager as the central abstraction

A `TargetManager` class encapsulates the target list, the active selection, and the per-target accumulators. The stream loop interacts with it instead of a bare `TransformAccumulator` + `DEVICE_NAME` constant.

**Option 2 compatibility**: `TargetManager` exposes an `add_target(name)` method so that a future IGTL listener can dynamically register targets discovered via bidirectional communication. The internal target dict is mutable by design.

### Configuration via CLI arguments

Introduce a minimal `argparse` setup for `--targets`. This avoids the overhead of a config file system while unblocking this feature. Default: `["SpaceMouseTransform"]` for full backward compatibility.

We won't add a config file now, but the design doesn't preclude one later — a config file parser would simply feed the same data into `TargetManager`.

### Switching via visualization UI

A `QComboBox` dropdown in the right sidebar, placed above the DOF locks panel (target selection is the primary control — it determines which structure the space mouse moves). Dropdown changes emit a `("switch_target", name)` command through the existing `cmd_queue`.

**Option 2 compatibility**: The same `("switch_target", name)` command can later be emitted by an IGTL listener thread — the stream loop doesn't care who sent it. If bidirectional communication adds a new target, a `("add_target", name)` command updates both the TargetManager and signals the visualization process to refresh its dropdown.

### Immediate matrix send on switch

When the active target changes, the stream loop immediately sends the new target's current matrix to Slicer. Without this, if the space mouse is at rest, Slicer wouldn't receive the switched target's transform until the mouse moves (because `compute_increments` returns `None` for zero motion).

---

## Changes By File

### New: `src/greifer/target.py`

```python
class TargetManager:
    """Manages multiple named targets, each with independent transform state."""

    def __init__(
        self,
        names: Sequence[str],
        *,
        reorthogonalize_interval: int = 1000,
    ) -> None:
        # dict preserves insertion order (first name = initial active target)
        self._targets: dict[str, TransformAccumulator] = {
            name: TransformAccumulator(reorthogonalize_interval)
            for name in names
        }
        self._active: str = names[0]

    @property
    def active_name(self) -> str: ...

    @property
    def active_accumulator(self) -> TransformAccumulator: ...

    @property
    def names(self) -> list[str]: ...

    def switch_to(self, name: str) -> ndarray:
        """Switch active target. Returns current matrix of new target.
        Raises KeyError if name not in target list."""
        ...

    def add_target(self, name: str) -> None:
        """Register a new target (idempotent). For future Option 2 use."""
        ...

    def update(self, dx, dy, dz, rx, ry, rz) -> ndarray:
        """Delegate to the active target's accumulator."""
        ...
```

### Modified: `src/greifer/app.py`

**Configuration section:**
- Remove `DEVICE_NAME` constant
- Add `argparse` setup in `main()` with `--targets` (nargs="+", default=["SpaceMouseTransform"])

**`_stream_loop` signature:**
- Replace the implicit `DEVICE_NAME` closure with an explicit `target_manager: TargetManager` parameter
- Remove the standalone `accumulator` local variable

**Command handling (inside `_stream_loop`):**
```python
# existing commands...
elif cmd[0] == "switch_target":
    matrix = target_manager.switch_to(cmd[1])
    # Send current matrix immediately so Slicer updates even if mouse is at rest
    client.send_message(
        pyigtl.TransformMessage(matrix, device_name=target_manager.active_name)
    )
```

**Hot-path transform send:**
```python
# Before:
matrix = accumulator.update(*increments)
transform_msg = pyigtl.TransformMessage(matrix, device_name=DEVICE_NAME)

# After:
matrix = target_manager.update(*increments)
transform_msg = pyigtl.TransformMessage(matrix, device_name=target_manager.active_name)
```

**`main()` changes:**
- Parse args, construct `TargetManager` from parsed target names
- Pass target names to `run_visualization`
- Update settings panel to show target list instead of single device name

### Modified: `src/greifer/visualization.py`

**`run_visualization` signature:**
```python
def run_visualization(
    data_queue: Queue[Sample | None],
    cmd_queue: Queue[Command],
    initial_sensitivity: Sensitivity | None = None,
    target_names: Sequence[str] = ("SpaceMouseTransform",),
) -> None:
```

**New `TargetPanel` widget** (placed at top of right sidebar):
- `QComboBox` populated with `target_names`
- On selection change: `cmd_queue.put_nowait(("switch_target", selected_name))`
- Styled consistently with existing panels (bold header label, dark background)
- If only one target: panel still visible but dropdown is effectively inert (no behavioral difference, but the user sees what's targeted)

**`Command` type update:**
```python
type Command = (
    tuple[str, Axis]
    | tuple[str, Sensitivity]
    | tuple[str, str]  # ("switch_target", name)
)
```

**Right sidebar layout change:**
```
Right Panel (fixed width=220px)
├── TargetPanel (NEW — dropdown for target selection)
├── DofLockPanel (unchanged)
└── SensitivityPanel (unchanged)
```

### New: `tests/test_target.py`

Test cases for `TargetManager`:
- Construction with one and multiple targets
- `active_name` defaults to first in list
- `switch_to` changes active target and returns its matrix
- `switch_to` unknown name raises `KeyError`
- `update` delegates to active accumulator only (inactive targets unaffected)
- `add_target` registers new target with identity matrix
- `add_target` is idempotent (adding existing name is a no-op)
- `names` returns all registered targets in insertion order

### Modified: `tests/test_integration.py`

- Update `_stream_loop` calls to pass a `TargetManager` instead of relying on the `DEVICE_NAME` closure
- Add test: switch_target command changes which device_name appears in sent messages
- Add test: switch_target triggers immediate matrix send

---

## Implementation Order

1. **`target.py`** + **`test_target.py`** — TargetManager in isolation, fully tested
2. **`app.py`** stream loop refactor — swap accumulator for TargetManager, add command handling
3. **`test_integration.py`** — update existing tests, add switch_target tests
4. **`visualization.py`** — add TargetPanel, wire to cmd_queue
5. **`app.py` `main()`** — argparse + wiring (pass target names to visualization)

Steps 1-3 are functional without UI changes (targets switchable via tests/commands). Steps 4-5 add the user-facing controls.

---

## Out of Scope (noted for future work)

- **Reset button**: Reset a target's accumulated transform to identity. Useful but orthogonal to multi-target switching.
- **Config file** (TOML/YAML): Would replace or supplement CLI args. Not needed for initial delivery.
- **Per-target sensitivity**: Each structure could have its own sensitivity profile. YAGNI for now.
- **Per-target DOF locks**: Same rationale — shared locks are sufficient initially.
- **Option 2 (bidirectional IGTL)**: The `add_target` method and `cmd_queue` command pattern are the integration seams. A future IGTL listener thread would push `("switch_target", name)` or `("add_target", name)` commands into the same queue.
