# Native VSA2: three representative-side wins ported and measured on H100

Source `2c67a73`, job 809484, H100 80GB HBM3. Build and import passed; `tests/test_vsa2.py` **499 passed / 276
skipped** and `tests/test_vsa.py` **143 passed / 28 skipped** on the same build that produced these numbers.

Three changes from the representative campaign apply to native VSA2. Each is an environment toggle, off by
default when this was measured:

| toggle | what it does | rep result it came from |
|---|---|---|
| `ECHOSPARSE_VSA_COARSE_META_CACHE` | reuses the shape-only coarse CSR metadata across calls | `RESULTS.md` §13 (CFG39, -3.2% forward) |
| `ECHOSPARSE_VSA_FUSED_COLS` | builds the router's kv->q CSR in one kernel instead of three | `RESULTS.md` §26 (CFG46, -1.3% forward) |
| `ECHOSPARSE_VSA_TK_BWD_STORE` | tier-A backward stores dK/dV from their single owner CTA instead of reduce-adding into zeroed fp32 accumulators | `RESULTS.md` §1 (CFG37, -2.1% F+B) |

## Method

`benchmarks/vsa2_native_bench.py`, four processes per shape in ABBA order (default, arm, arm, default), CUDA
events, warmup 10, 30 iterations, 3 batches; ratios come from inside one run only. Every `ECHOSPARSE_VSA_*`
variable is stripped from the children's environment except the arm's own. A baseline-vs-baseline arm gives the
A/A displacement. Seven shapes (L 2048-32768, H 8/16, D 64/128) plus the bench's own partial-`blk_cnt` variants;
14 (arm, shape) rows per arm, all verdict OK.

## Result

| arm | forward ratio (median, min) | backward ratio (median, min) | F+B ratio (median, min) |
|---|---|---|---|
| `COARSE_META_CACHE` | **0.931**, 0.537 | 0.997, 0.961 | **0.972**, 0.712 |
| `FUSED_COLS` | 0.994, 0.977 | 1.002, 0.945 | 0.991, 0.916 |
| `TK_BWD_STORE` | 1.001, 0.941 | **0.956**, 0.911 | **0.968**, 0.919 |

- **`COARSE_META_CACHE` pays most where the operator is small**: 0.537 on the forward at L2048/H16/D64 and 0.630 at
  L4096/H8/D128, converging to ~0.96-0.98 at L16384 and above, where the metadata is a smaller share of the call.
- **`TK_BWD_STORE` is a steady backward win**, 3.4-8.9% across every shape, and 2-8% on a complete autograd call.
- **`FUSED_COLS` is neutral here** (0.99-1.00 on every row), unlike the -1.3% it gave the representative operator:
  native's router is a smaller share of the call, so removing two launches out of fourteen does not resolve.

## Error

- **forward `o` and the selection are bit-identical** to the default on every shape, and the A/A arm is
  bit-identical to itself.
- **`dk`, `dv`, `dgate` are bit-identical** to the default (0 differing elements of 16-33M): the store path writes
  its tile instead of reduce-adding, so it also makes those gradients deterministic.
- **`dq` is not bit-identical, and does not need to be**: it is a cross-CTA fp32 reduce-add. Its `rel_l2` against
  the default is at most **1.58e-5**, while the same error between two default runs is at most **1.51e-5** — the
  arm is inside the operator's own run-to-run noise, which is the only meaningful bar here.
- Padded slots past `blk_cnt` are exactly zero in `o` and every gradient on every shape.

## What is not claimed

No hardware counters (`ncu` is not permitted on this machine), so these are wall-clock ratios only. `FUSED_COLS`
is left available but off by default: it is bit-exact and slightly cheaper in kernel count, but this build does
not resolve a win. Nothing here changes a selection, a bitmap or a forward output.
