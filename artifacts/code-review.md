# greifer — Code Review

Review of the current codebase state (post stream-loop extraction).

## Correctness

### Rotation matrix drift (`app.py:80`)

`T_cumulative = dT @ T_cumulative` accumulates floating-point error. Over long
sessions the 3×3 rotation submatrix drifts from orthonormal, introducing
skew/shear into the Slicer transform. Periodic re-orthogonalization (e.g. SVD
polar factor every ~1000 iterations) would fix this cheaply:

```python
if iteration % 1000 == 0:
    U, _, Vt = np.linalg.svd(T_cumulative[:3, :3])
    T_cumulative[:3, :3] = U @ Vt
```

### Sleep-based timing drifts (`app.py:87`)

`time.sleep(dt)` does not account for time spent in the loop body. At 500 Hz
(`dt = 0.002s`) this means the effective rate is always lower than `UPDATE_HZ`.
A monotonic clock target loop fixes this:

```python
next_tick = time.monotonic()
while True:
    # ... work ...
    next_tick += dt
    time.sleep(max(0, next_tick - time.monotonic()))
```

## Robustness

### No error handling for the Slicer connection (`app.py:115-118`)

If Slicer isn't running, `OpenIGTLinkClient(...)` will silently fail or throw.
The `time.sleep(1)` is a hope-based readiness check. Similarly, `send_message`
inside the loop has no handling for a dropped connection — a network hiccup will
crash the app with no recovery hint.

Suggestion: catch the connection error and print a clear diagnostic, and wrap
`send_message` with a try/except that logs and optionally reconnects.

## Design

### Hardcoded config with no override mechanism (`app.py:19-31`)

All tunable parameters (`SLICER_HOST`, `SLICER_PORT`, `TRANS_SCALE`, etc.) are
module-level constants with no way to change them without editing source. Even a
lightweight `argparse` for host, port, device name, and scale factors would make
the tool significantly more usable.

### `jaxtyping` dependency for a single annotation (`math.py:6`)

`jaxtyping` is a non-trivial runtime dependency used only for the return type of
`build_rotation_matrix`. It also forces a file-level `# ruff: noqa: F722`
suppression. Replacing with `np.ndarray` (or `numpy.typing.NDArray`) would drop
the dependency and the noqa.

### Queue rate mismatch (`app.py:50`, `visualization.py:17`)

The producer enqueues at 500 Hz; the consumer renders at 60 Hz. The queue
(`maxsize=600`) fills in ~1.2 seconds, after which samples are silently dropped
via `put_nowait` / `except Full: pass`. This works but means the visualization
always shows a sampled subset of the data. If intentional, a comment would
clarify; if not, consider downsampling on the producer side (e.g. enqueue every
8th sample to match 60 Hz).

### `pyproject.toml` description (`pyproject.toml:4`)

Still reads `"Add your description here"`.

## Minor / Stylistic

### `import queue as queue_module` (`app.py:2`, `visualization.py:4`)

Unusual alias. `from queue import Full, Empty` and using the exception names
directly is more idiomatic and avoids the naming awkwardness.

### `assert first_plot is not None` (`visualization.py:74`)

Uses `assert` for runtime logic. If `CHANNELS` were ever empty this produces a
confusing `AssertionError`. Minor given `CHANNELS` is a module constant, but a
`ValueError` with a message would be clearer.

### `for i in range(len(CHANNELS))` (`visualization.py:106`)

Idiomatic Python: `for i, ch in enumerate(CHANNELS)`.

## What's Done Well

- Clean module separation: `math.py` (pure computation), `visualization.py`
  (isolated process), `app.py` (orchestration).
- The `_stream_loop` extraction keeps the hot path flat and readable at 2 indent
  levels, while `main()` handles setup/teardown.
- Multiprocessing architecture with a daemon visualization process keeps the
  streaming loop decoupled from rendering.
- Graceful shutdown (sentinel value, `join` with timeout, `terminate` fallback)
  is thorough and correct.
- `_Visualizer` class cleanly encapsulates mutable plot state.
- `Channel` as a `NamedTuple` with declarative `CHANNELS` tuple is a nice
  pattern for the plot grid layout.
