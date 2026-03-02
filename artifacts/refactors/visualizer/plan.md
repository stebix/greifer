# `_Visualizer` Refactoring Plan

## Context

`_Visualizer` in `visualization.py:55–155` encapsulates the real-time plot state
for 6-DOF SpaceMouse data. It owns three responsibilities: data buffering
(queue → deques), curve rendering (deques → pyqtgraph), and lock-indicator
toggling. The class is ~100 lines and generally clean, but its constructor
conflates data-state setup with Qt widget construction, which forces tests into
the `object.__new__()` escape hatch. The code review in `artifacts/gutter/`
already flagged two low-hanging issues (`assert` → `ValueError`, `range(len())`
→ `enumerate`).

The goal is to raise quality through targeted structural and idiomatic
improvements—no new abstractions for their own sake.

---

## Changes

### 1. ~~Inject plot artifacts instead of constructing them in `__init__`~~ DONE

**Problem.** `_Visualizer.__init__` receives a bare `GraphicsLayoutWidget` and
internally creates all `PlotItem`s, `PlotDataItem` curves, `TextItem` labels,
and x-axis linkage. This means the class cannot be instantiated without a live
Qt session, which is why both `_make_drain_target()` and `_make_lock_target()`
in the tests bypass the constructor entirely with `object.__new__`.

**Change.** Extract plot construction into a module-level factory function
(`_build_plot_grid`). `_Visualizer.__init__` accepts the finished products:

```python
def _build_plot_grid(
    win: pg.GraphicsLayoutWidget,
) -> tuple[list[pg.PlotItem], list[pg.PlotDataItem], list[pg.TextItem], pg.PlotItem]:
    """Create the 6-panel plot grid. Returns (plots, curves, labels, anchor_plot)."""
    ...

class _Visualizer:
    def __init__(
        self,
        queue: Queue[Sample | None],
        app: QApplication,
        *,
        plots: list[pg.PlotItem],
        curves: list[pg.PlotDataItem],
        lock_labels: list[pg.TextItem],
        anchor_plot: pg.PlotItem,
    ) -> None:
        ...
```

- `run_visualization()` calls `_build_plot_grid(graph_widget)` then passes the
  results to `_Visualizer(...)`.
- Tests can now construct `_Visualizer` normally with mocks for the plot objects
  instead of using `object.__new__`.
- No new classes or indirection—just a plain function that returns a tuple.

**Test impact.** `_make_drain_target` and `_make_lock_target` become thin
wrappers that pass `MagicMock` lists into a normal `__init__` call. The
`object.__new__` hack is eliminated.

### 2. ~~Promote `Sample` to a `NamedTuple`~~ DONE

**Problem.** `type Sample = tuple[float, float, float, float, float, float, float]`
is a bare positional alias — nothing documents which index is time, which is x,
etc. The producer in `app.py:161` packs the tuple by position; the consumer in
`_drain_queue` unpacks by index. The data layout is implicit convention.

**Change.** Replace the type alias with a `NamedTuple`:

```python
class Sample(NamedTuple):
    t: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
```

`NamedTuple` is a tuple subclass, so all existing positional code (indexing,
iteration, unpacking) continues to work without changes. The producer in
`app.py` packs positionally and doesn't need updating. `_drain_queue` can
optionally switch to named access (`sample.t`, `sample.x`, ...) for clarity,
but isn't forced to.

This also subsumes the old "derive buffer count from CHANNELS" item — the
buffer count becomes `len(Sample._fields)` instead of the magic number 7.

**Not included: connecting Channel styling to Sample fields.** The `Channel`
NamedTuple already serves as the styling metadata (name, grid position, color),
and the link to `Sample` fields is implicit via `enumerate(CHANNELS)` ordering.
Making this explicit (e.g. adding a `field: str` to `Channel`) would add
indirection for a coupling that is stable and unlikely to change. The cost
outweighs the benefit.

### 3. ~~Use `enumerate` in `update()`~~ DONE

**Problem.** `for i in range(len(CHANNELS)):` with `self._bufs[i + 1]` is
non-idiomatic and splits the channel/index relationship across two expressions.

**Change.**

```python
for i, _ch in enumerate(CHANNELS):
    arr = np.array(self._bufs[i + 1])
    self._curves[i].setData(t_arr, arr)
```

Already flagged in the existing code review. Minimal change.

### 4. ~~Replace `assert` with `ValueError`~~ DONE

**Problem.** `assert first_plot is not None` at line 99 uses `assert` for
a runtime invariant. If `CHANNELS` were empty, this produces a confusing
`AssertionError` instead of a descriptive error.

**Change.**

```python
if first_plot is None:
    raise ValueError("CHANNELS must not be empty")
```

### 5. ~~Centralize `AXIS_TO_INDEX` mapping~~ DONE

**Problem.** The mapping `{Axis.X: 0, Axis.Y: 1, ..., Axis.YAW: 5}` is
defined twice: in `run_visualization()` (line 488) and in
`DofLockFilter._INDEX` (transform.py:90). Both map Axis enum → positional
index in the 6-DOF tuple.

**Change.** Add a single canonical mapping as a class attribute on `Axis`:

```python
# In transform.py
class Axis(StrEnum):
    X = "x"
    Y = "y"
    Z = "z"
    ROLL = "roll"
    PITCH = "pitch"
    YAW = "yaw"

    @property
    def index(self) -> int:
        """Positional index in the (x, y, z, roll, pitch, yaw) tuple."""
        return _AXIS_ORDER.index(self)

_AXIS_ORDER: tuple[Axis, ...] = tuple(Axis)
```

Then both `DofLockFilter._INDEX` and `run_visualization` use `axis.index`
instead of maintaining parallel dicts. `DofLockFilter._INDEX` can be removed
entirely.

### 6. ~~Import queue exceptions directly~~ DONE

**Problem.** `import queue as queue_module` is an unusual alias used to avoid
shadowing the `queue` parameter name. The module is then used only for
`queue_module.Empty` and `queue_module.Full`.

**Change.** `from queue import Empty, Full` and use the exception names
directly. Clearer, avoids the naming awkwardness.

### 7. ~~Tighten type annotations on queues~~ DONE

**Problem.** Every queue parameter is annotated as bare `multiprocessing.Queue`
— the type system can't distinguish the data queue (`Sample | None`) from the
command queue (`tuple[str, Axis] | tuple[str, Sensitivity]`).

A subtlety: `multiprocessing.Queue` is not a class — it's a bound method on
`BaseContext` that acts as a factory function. This means
`multiprocessing.Queue[T]` is invalid both at runtime (not subscriptable) and
to type checkers (they see a callable, not a generic type).
`from __future__ import annotations` would only defer evaluation, not fix the
type-checker's view.

The actual class lives at `multiprocessing.queues.Queue`, which *does* support
generic subscripting (via `__class_getitem__`) and is recognized by typeshed.

**Change.** Import the real class and use it in annotations:

```python
from multiprocessing.queues import Queue

# In _Visualizer
def __init__(
    self,
    queue: Queue[Sample | None],
    app: QApplication,
    ...
) -> None:

# In DofLockPanel / SensitivityPanel
def __init__(
    self,
    cmd_queue: Queue[tuple[str, Axis | Sensitivity]],
    ...
) -> None:

# In run_visualization
def run_visualization(
    data_queue: Queue[Sample | None],
    cmd_queue: Queue[tuple[str, Axis | Sensitivity]],
    ...
) -> None:
```

No `from __future__ import annotations` needed. The existing hints are
otherwise solid (all methods have return types, internal collections are
annotated inline).

### 8. ~~Decompose `run_visualization()`~~ DONE

**Problem.** The entry-point function (lines 448–500) is a 53-line monolith
that handles: Qt app creation, main window layout, right-panel assembly,
visualizer instantiation, lock-button wiring, and timer setup. Not complex
per se, but the single flat block makes it harder to scan.

**Change.** Extract two small helpers alongside the existing `_build_plot_grid`
from change 1:

```python
def _build_right_panel(
    cmd_queue: Queue[tuple[str, Axis | Sensitivity]],
    sensitivity: Sensitivity,
) -> tuple[QWidget, DofLockPanel]:
    """Assemble the lock + sensitivity sidebar."""
    ...

def run_visualization(...) -> None:
    ...
    graph_widget = pg.GraphicsLayoutWidget()
    plots, curves, labels, anchor = _build_plot_grid(graph_widget)
    right_panel, lock_panel = _build_right_panel(cmd_queue, initial_sensitivity)
    ...
```

Each helper is a pure layout builder with no state. `run_visualization` becomes
a short wiring function.

---

## Ordering and Dependencies

| Step | Change                      | Depends on | Risk   |
| ---- | --------------------------- | ---------- | ------ |
| 1    | Inject plot artifacts       | —          | Medium |
| 2    | `Sample` → `NamedTuple`     | —          | Low    |
| 3    | `enumerate` in `update()`   | —          | Low    |
| 4    | `assert` → `ValueError`    | —          | Low    |
| 5    | Centralize axis index       | —          | Low    |
| 6    | Import `Empty, Full`        | —          | Low    |
| 7    | Parameterize queue types    | 6          | Low    |
| 8    | Decompose entry function    | 1          | Low    |

Steps 2–6 are independent and can land in any order. Step 7 is cleanest after
step 6 (since `from __future__ import annotations` interacts with the import
section). Step 1 is the structural change; step 8 builds on it. Recommended
execution order: 2–6 as quick wins, then 1, then 7, then 8.

## Out of scope

- **New classes or abstractions** (e.g. a `BufferBank` class). The data layer
  is 30 lines—wrapping it adds indirection without value.
- **Panel refactors** (`DofLockPanel`, `SensitivityPanel`). These are already
  well-structured and independently testable. They can be a separate cycle.
- **Connecting `Channel` to `Sample` fields.** The implicit ordering convention
  is stable and well-understood. Formalizing it adds coupling without payoff
  given the fixed 6-DOF domain.
