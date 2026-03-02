# structured-targetting feature description

In this project, we want to do a feature extension. Currently we only adress a node in 3DSlicer via the single hardcoded `DEVICE_NAME` parameter in the @src/greifer/app.py file. The node targeted with this `DEVICE_NAME` symbolizes a specific structure that we want to manipulate via `greifer` and the space mouse. In the intended usage of the app, we may have multiple structures that we want to position separately via `greifer`.

To achieve this goal, we want to think about mechanisms to target different nodes in 3DSlicer. 
Options are:

- strict name based matching: we could specifiy a list of node names in the `greifer` config, and then we would have to make sure that the nodes in 3DSlicer are named accordingly.

- bidirectional communication: we could have a mechanism to send messages from `greifer` to 3DSlicer and vice versa, so that we can select the target node in 3DSlicer and then send its name to `greifer` to set it as the current target. --> Can you research this possibility with pyIGTLink?

---

## Analysis & Research Results

### Current State

`DEVICE_NAME` is a single hardcoded constant (`"SpaceMouseTransform"`) in `app.py:30`. It flows directly into `pyigtl.TransformMessage(matrix, device_name=DEVICE_NAME)` at line 182. There is no mechanism to change it at runtime or at startup without editing source. Communication is strictly unidirectional — greifer sends transform messages to Slicer, nothing comes back.

### Option 1: Strict Name-Based Matching (Config-Driven)

Greifer holds a list of known target names (e.g. from a config file or CLI args) and lets the user cycle between them locally (e.g. via a keyboard shortcut or a dropdown in the visualization UI).

**Pros:**
- Simple to implement — just swap which `device_name` string goes into `TransformMessage`
- No dependency on Slicer-side scripting
- Deterministic — the user explicitly declares what structures exist

**Cons:**
- Names must be manually synchronized between greifer config and Slicer scene
- Adding/removing structures requires editing config and restarting (or a reload mechanism)
- Fragile — a typo in the config silently sends transforms to a nonexistent node

### Option 2: Bidirectional Communication via pyIGTLink

**Research result: pyigtl fully supports receiving messages.**

The `OpenIGTLinkClient` (which greifer already uses) inherits from `OpenIGTLinkBase`, which provides:

- `get_latest_messages()` — non-blocking, returns all pending incoming messages
- `wait_for_message(device_name, timeout)` — blocking wait for a message from a specific device

pyigtl supports `StringMessage`, which is ideal — Slicer could send a string like `"Femur"` and greifer would switch its active target.

**How it would work in practice:**

1. **Slicer side**: A small Python script (run in Slicer's Python console or as a module) uses `OpenIGTLinkIF` to send a `StringMessage` when the user selects a node. Slicer already has this infrastructure — `slicer.vtkMRMLIGTLConnectorNode` can push messages.
2. **Greifer side**: A lightweight listener (separate thread or polled in the existing `_stream_loop` via `get_latest_messages()`) watches for incoming `StringMessage` objects and updates the active `device_name`.
3. **UX flow**: User clicks a transform node in Slicer -> Slicer script fires a `StringMessage` with the node name -> greifer picks it up -> subsequent `TransformMessage` objects are addressed to that node.

**Pros:**
- No manual name synchronization — Slicer is the source of truth
- Dynamic — add/remove structures in the Slicer scene without touching greifer
- Natural UX — the user works in Slicer where they already see the 3D scene

**Cons:**
- Requires a companion script running in Slicer (small, but still a deployment concern)
- Slightly more complex architecture (message listener thread/polling)
- Depends on Slicer's OpenIGTLinkIF being configured for bidirectional traffic (connector node needs to be set to send AND receive)

### Recommended Approach: Layered Design

These two options are not mutually exclusive. A solid design could combine both:

1. **Config-driven target list as the foundation** — greifer knows which structures it *can* target (with a sensible default). This also enables the visualization UI to show a target selector.
2. **Bidirectional IGTL as an optional enhancement** — when Slicer sends a selection message, greifer switches the active target *if* the name matches a known target (or auto-adds it to the list).

This layered approach gives resilience (config works without bidirectional setup) and convenience (bidirectional when available).