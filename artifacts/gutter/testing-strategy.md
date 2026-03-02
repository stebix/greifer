# Testing Strategy for greifer

## Testability Assessment

The codebase has three distinct testability tiers:

| Module | Testability | Why |
|--------|------------|-----|
| `math.py` | **Excellent** | Pure function, no side effects, deterministic |
| `app.py` `_stream_loop` | **Poor as-is** | Monolithic loop coupling device I/O, computation, network, and IPC in one function |
| `visualization.py` | **Moderate** | Queue-draining logic is testable; rendering requires a Qt event loop |

The core problem is that `_stream_loop` is a 60-line `while True` loop that
interleaves pure computation (scaling, matrix building, accumulation,
reorthogonalization) with three I/O concerns (device read, queue put, network
send). This makes it impossible to unit test the computation without mocking all
three I/O channels.

## Recommended Refactoring (Prerequisite)

Extract the transform accumulation into a small, stateful, pure class:

```python
class TransformAccumulator:
    """Accumulates incremental 4x4 transforms with periodic reorthogonalization."""

    def __init__(self, reorthogonalize_interval: int = 1000) -> None:
        self.matrix: np.ndarray = np.eye(4)
        self._iteration: int = 0
        self._reorth_interval = reorthogonalize_interval

    def update(self, dx, dy, dz, rx, ry, rz) -> np.ndarray:
        """Apply an incremental transform. Returns the new cumulative matrix."""
        ...

    def reorthogonalize(self) -> None:
        """SVD polar-factor correction on the rotation submatrix."""
        ...
```

This single extraction unlocks all the high-value unit tests without touching
the I/O flow. The `_stream_loop` becomes a thin orchestrator that calls
`accumulator.update(...)` and passes the result to the network client.

Similarly, extract the scaling + threshold logic:

```python
def compute_increments(state, trans_scale, rot_scale) -> tuple[float, ...] | None:
    """Scale raw device state. Returns None if below motion threshold."""
    ...
```

## Test Pyramid

```
              ┌──────────┐
              │ Manual /  │  ← What you have now (Slicer visual check)
              │ E2E smoke │     Keep as a release gate, don't automate yet
              └─────┬─────┘
                    │
            ┌───────┴────────┐
            │  Integration   │  ← Mock device + mock client, run real loop
            │  (3-5 tests)   │     for N iterations, assert output properties
            └───────┬────────┘
                    │
      ┌─────────────┴─────────────┐
      │       Unit Tests          │  ← The bulk of the value
      │       (20-30 tests)       │
      └───────────────────────────┘
```

## Layer 1: Unit Tests — `math.py`

Easiest wins and the highest confidence-per-line-of-test-code.

| Test Case | What it verifies |
|-----------|-----------------|
| `test_identity_at_zero_angles` | `build_rotation_matrix(0, 0, 0)` returns `np.eye(3)` |
| `test_90deg_x_rotation` | Known rotation: `[0, 0, 1]` maps to `[0, -1, 0]` |
| `test_90deg_y_rotation` | Known rotation: `[1, 0, 0]` maps to `[0, 0, -1]` |
| `test_90deg_z_rotation` | Known rotation: `[1, 0, 0]` maps to `[0, 1, 0]` |
| `test_result_is_orthonormal` | `R @ R.T ≈ I` and `det(R) ≈ 1` for random angles |
| `test_composition_order_is_zyx` | Verify `Rz @ Ry @ Rx` by comparing to manual composition |
| `test_small_angle_approximation` | For tiny angles, `R ≈ I + skew(rx, ry, rz)` |
| `test_inverse_is_transpose` | `R(-a) ≈ R(a).T` for valid rotation matrices |

Use `np.testing.assert_allclose` with `atol=1e-12` throughout.

## Layer 2: Unit Tests — `TransformAccumulator` (after extraction)

| Test Case | What it verifies |
|-----------|-----------------|
| `test_initial_state_is_identity` | `.matrix` starts as `np.eye(4)` |
| `test_pure_translation_accumulates` | Two `(dx=1,0,0,0,0,0)` updates → translation `[2,0,0]` |
| `test_pure_rotation_accumulates` | Two equal yaw increments → double the yaw |
| `test_zero_motion_returns_identity` | `update(0,0,0,0,0,0)` leaves matrix unchanged |
| `test_reorthogonalization_triggers` | After N updates, rotation submatrix is still orthonormal |
| `test_reorthogonalization_preserves_translation` | SVD correction doesn't corrupt the `[:3, 3]` column |
| `test_drift_without_reorthogonalization` | After 10k noisy updates *without* reorth, `det(R)` drifts from 1.0 — proving the feature is needed |
| `test_accumulation_order` | `dT @ T_cumulative` (left-multiply) gives world-frame increments |

## Layer 3: Unit Tests — `compute_increments` (after extraction)

| Test Case | What it verifies |
|-----------|-----------------|
| `test_scaling_applied_correctly` | Raw `(100, 200, 300, ...)` × `TRANS_SCALE` yields correct mm values |
| `test_below_threshold_returns_none` | Tiny input → `None` (skipped) |
| `test_exactly_at_threshold` | Boundary condition |

## Layer 4: Unit Tests — `_Visualizer._drain_queue`

Test with a real `multiprocessing.Queue` or a `queue.SimpleQueue` fake:

| Test Case | What it verifies |
|-----------|-----------------|
| `test_drain_empty_queue` | Returns `True`, no crash |
| `test_drain_single_sample` | Appends to all 7 buffers, time offset computed |
| `test_drain_sentinel_returns_false` | `None` in queue → returns `False` |
| `test_negative_timestamp_skipped` | Samples with `t < 0` are discarded |
| `test_time_offset_subtracted` | Second sample's time is relative to first |

## Layer 5: Integration Tests

These test the wired-together system with fakes for the external boundaries.

### Fake objects needed

```python
@dataclass
class FakeState:
    x: float = 0.0; y: float = 0.0; z: float = 0.0
    roll: float = 0.0; pitch: float = 0.0; yaw: float = 0.0
    t: float = 0.0

class FakeDevice:
    """Yields a scripted sequence of states, then raises KeyboardInterrupt."""
    def __init__(self, states: list[FakeState]): ...
    def read(self) -> FakeState: ...

class FakeClient:
    """Records all TransformMessages sent."""
    def __init__(self): self.messages = []
    def send_message(self, msg): self.messages.append(msg)
```

| Test Case | What it verifies |
|-----------|-----------------|
| `test_stream_loop_sends_transforms` | Run 10 iterations with known inputs, verify messages sent to `FakeClient` contain expected matrices |
| `test_stream_loop_skips_zero_motion` | All-zero states → no messages sent |
| `test_stream_loop_fills_vis_queue` | Verify queue receives samples when provided |
| `test_stream_loop_tolerates_full_queue` | Full queue → no crash, loop continues |
| `test_round_trip_identity` | 1000 random small increments followed by their exact inverses → `T_cumulative ≈ I` |

## What NOT to Test

- **Don't automate PyQt rendering tests.** The visualization is a diagnostic
  tool, not a deliverable. Keep manual inspection for now. `pytest-qt` adds
  heavy CI complexity for low-value assertions.
- **Don't mock `pyigtl.OpenIGTLinkClient` internals.** Test at the
  `send_message` boundary, not inside the protocol.
- **Don't test `main()` end-to-end.** It's pure orchestration wiring — the
  individual pieces are covered. A smoke test that imports `greifer` without
  crashing is sufficient.
- **Don't test `rich` console output.** It's cosmetic.

## Tooling

Add to `[dependency-groups] dev` in `pyproject.toml`:

```toml
dev = [
    "ruff>=0.15.4",
    "pytest>=8.0",
    "pytest-cov>=6.0",
    "hypothesis>=6.0",     # Property-based testing for math.py
]
```

**Why Hypothesis?** Rotation matrices have strong algebraic invariants
(orthonormality, determinant = 1, inverse = transpose). Property-based tests
catch edge cases that hand-picked angles miss — e.g., gimbal lock near ±π/2.

## Directory Layout

```
tests/
├── conftest.py           # Shared fixtures (FakeDevice, FakeClient, etc.)
├── test_math.py          # Layer 1: rotation matrix tests
├── test_accumulator.py   # Layer 2: TransformAccumulator tests
├── test_increments.py    # Layer 3: scaling + threshold tests
├── test_visualization.py # Layer 4: queue drain logic tests
└── test_integration.py   # Layer 5: stream loop integration tests
```

## CI Integration

Extend `.github/workflows/ci.yaml`:

```yaml
- name: Run tests
  run: uv run pytest --tb=short --cov=greifer --cov-report=term-missing
```

## Prioritized Implementation Order

1. **`test_math.py`** — immediate, zero refactoring needed, high confidence gain
2. **Extract `TransformAccumulator`** — small refactoring, unlocks the core tests
3. **`test_accumulator.py`** — the highest-value tests in the suite (correctness
   of the thing Slicer receives)
4. **`test_integration.py`** with fakes — validates the wiring
5. **`test_visualization.py`** drain logic — moderate value, easy to write
6. **Hypothesis property tests** for `math.py` — catches subtle edge cases

## Key Insight

The codebase is small (~330 lines) but the computation buried inside
`_stream_loop` is what actually matters for correctness — it's what drives
Slicer. Extracting it into `TransformAccumulator` is a ~30-line refactor that
turns an untestable hot loop into a thoroughly testable state machine. That
single change gives you 80% of the testing value.
