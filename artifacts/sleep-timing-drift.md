# Sleep-Based Timing Drift

Issue reference: `app.py:87` (code-review.md), currently at `app.py:75` and `app.py:94`.

## Problem

`time.sleep(dt)` sleeps for `dt` seconds **after** all loop-body work has
completed. It does not account for the time the loop body itself consumed.

Each iteration's actual period is:

```
T_actual = T_work + T_sleep
```

where `T_work` includes:

1. `device.read()` — USB poll
2. `vis_queue.put_nowait()` — serialization + IPC
3. Matrix math — rotation build, cumulative multiply, occasional SVD
4. `client.send_message()` — TCP send to Slicer

At a target of 500 Hz the budget per iteration is **2 ms**. Even if `T_work`
averages only 0.3–0.5 ms, the effective rate drops to:

```
1 / (0.002 + 0.0004) ≈ 416 Hz  →  ~17 % undershoot
```

### Aggravating factors

- **SVD spikes**: Re-orthogonalization every 1000 iterations is more expensive
  than a normal iteration, creating periodic rate dips.
- **TCP backpressure**: `send_message` can stall when the socket buffer is full.
- **OS scheduling jitter**: `time.sleep` granularity on Windows is typically
  1–15 ms, meaning a 2 ms sleep may actually sleep 3–16 ms.

### Consequences

- Transform update rate to 3D Slicer is **lower and less stable** than
  configured.
- Sensitivity tuning (`TRANS_SCALE`, `ROT_SCALE`) implicitly depends on a
  stable rate — variable timing means the same physical input produces slightly
  different displacement depending on system load.

## Mitigation

Replace the naive `time.sleep(dt)` with a **monotonic-clock target loop**.

### Implementation sketch

```python
next_tick = time.monotonic()

while True:
    # ... loop body ...

    next_tick += dt
    sleep_remaining = next_tick - time.monotonic()
    if sleep_remaining > 0:
        time.sleep(sleep_remaining)
    else:
        # Overran the budget — reset to avoid cascading catch-up
        next_tick = time.monotonic()
```

### What this achieves

1. **Self-correcting cadence** — If an iteration takes 0.5 ms of work, it
   sleeps only 1.5 ms, keeping the tick-to-tick period at exactly `dt` on
   average.
2. **Overrun handling** — The `else` branch prevents a burst of zero-sleep
   iterations trying to "catch up" after a long stall. It resets the baseline
   instead.
3. **No new dependencies** — Uses only `time.monotonic()`, already imported.

### Required changes (all in `_stream_loop`)

- Add `next_tick = time.monotonic()` before the `while True`.
- Remove the early `time.sleep(dt)` + `continue` on the idle branch (line 75)
  so both paths fall through to a single sleep point.
- Replace `time.sleep(dt)` at the bottom (line 94) with the tick-advance logic.
