# Legacy Archive

This directory holds code and data that are no longer part of the active BTC-1H runtime or research frame.

Rules:

- Archived files are retained for reference only.
- New code must not import from `archive/`.
- Anything worth reviving should be ported into the current architecture rather than imported from archived modules.
- Protected operational organs remain outside this directory and stay in active source paths.

Archive layout:

- `archive/legacy_decision/`: removed or superseded decision-layer code kept for reference.
- `archive/legacy_replay/`: legacy live-style replay harness, scripts, configs, and scenarios.
- `archive/legacy_probability/`: old probability engines or probability research retained for historical context.
- `archive/legacy_research/`: retired research utilities that are outside the active HMM/probability frame.
- `archive/legacy_docs/`: archived docs that describe superseded architecture.
