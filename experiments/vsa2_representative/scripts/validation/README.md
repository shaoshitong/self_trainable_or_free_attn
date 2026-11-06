# H100 baseline validation worker

These portable scripts validate the frozen representative-attention baseline identified by SOURCE_MANIFEST.yaml. The coordinator supplies resource policy, compiler locations and task paths through environment variables; no access configuration is included.

The source closure places these files under `_control/` in an immutable source snapshot. `job.sh` creates a per-job copy, verifies source and pinned TK headers, runs the CPU reference suite, builds the TK extension with ptxas resource reporting, runs generic and TK GPU suites, and only then captures the original 16K benchmark. `check_suites.py` contains the original baseline's expected test-family counts; update those counts for later expanded suites. It rejects missing success markers and all-skipped runs.

Legacy backward values are estimates from separately timed F+B minus F. The captured F+B interval is retained. The baseline scope is fine ABC attention with frozen selection, not the full routed/coarse/gated operator. A native tier-A row is not a semantic baseline for ABC.

Execution identities and raw build output stay with the coordinator. Share only sanitized relative compiler/test/performance feedback with coding assistants.
