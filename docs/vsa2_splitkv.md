# VSA2 balanced split-KV scheduling (experimental, H100)

`SplitKV` is an opt-in schedule for the fine branch of `vsa2_attention` / `vsa_attention`. It changes which CTA
works on which part of a block's selected list; the selected block pairs, the operator, the coarse branch and the
gate are unchanged, and outputs and gradients stay within the existing test tolerances. The default is unsplit,
and on the TK backend the default backward options are 0 (the original wait and walk algorithm). On H100 it helps when a few
selected lists are much longer than the rest: long query rows (forward) or hot kv columns (backward). On uniform
masks and on the real routed captures measured so far it does not help, so it is not a general speedup.

## Usage

Continuing the README example (`qb`, `kb`, `vb`, `gb`, `lay`):

```python
from echosparse import vsa

# forward-only: long query rows (rare stragglers)
o = vsa.vsa2_attention(qb, kb, vb, gb, lay.blk_cnt, R=16, topk=64,
                       split_kv=vsa.SplitKV(chunk=128, fwd=True, bwd=False))
# backward-only: hot kv columns
o = vsa.vsa2_attention(qb, kb, vb, gb, lay.blk_cnt, R=16, topk=64,
                       split_kv=vsa.SplitKV(chunk=64, fwd=False, bwd=True))
```

`SplitKV(chunk=64, fwd=True, bwd=True, max_scratch_bytes=1 << 30)`:

- `chunk`: a list with more than `chunk` blocks runs as `ceil(n / chunk)` balanced chunk tasks; shorter lists
  keep the direct path. 1 ≤ chunk ≤ 2^30.
- `fwd` / `bwd`: split the forward (q→kv lists) and the backward (kv→q lists) independently.
- `max_scratch_bytes`: cap on the forward fp32 partials (and the generic backward partials) per stage. Rows past
  the cap run unsplit; results do not change. With no host readback the partial buffer is sized from the host
  bound `min(max_scratch_bytes // slot_bytes, 2·ceil(nnz/chunk))` slots, so it is allocated even when nothing
  splits.

It works with `backend="tk"` (H100 production path) and `backend="generic"`, and with
`fine_block_sparse_attention`.

## Automatic policy (`SplitKV.auto()`)

```python
o = vsa.vsa2_attention(qb, kb, vb, gb, lay.blk_cnt, R=16, topk=64, split_kv="auto")   # = vsa.SplitKV.auto()
split = vsa.SplitKV.auto(chunk=64, policy=vsa.SplitPolicy(rel_mean=4.0, tail_frac=0.05, fwd_waves=1.0,
                                                          bwd_waves=8.0))
print(vsa.describe_split(qb, sel, split, backend="tk"))   # diagnostic: the plans it would build (synchronizes)
```

The GPU planner decides from the selection's CSR row lengths, separately per direction: q2k lengths for the
forward (and the generic variant-2 Q-owner pass), k2q lengths for the KV-owner backward. Both can be active at
once; there is no global hot-Q / hot-KV switch. A list of n blocks is eligible when

```
n >= max(min_chunks·chunk, chunk + 1, rel_mean · (its head's mean list length), tail_frac · W / resident)
```

with W the direction's total edges and resident = SMs × CTAs/SM of the consuming kernel (a fixed table per
backend and D, e.g. TK forward 132 × 4 = 528, TK backward D128 132 × 2 = 264; `resident_ctas` overrides). If
more than `waves × resident` lists are eligible (`fwd_waves` 1, `bwd_waves` 8), the GPU already has that much
long work to overlap and nothing is split in that direction. Ineligible, over-budget and over-cap lists run
direct and whole; no edge is dropped or duplicated. The decision never leaves the GPU (no `.item()`, no copy or
synchronization in the operator), but it is not free: each split direction still runs the planner (three small
kernels and two cumsums), a merge launch and a task grid of up to `rows + slots` CTAs, and the task kernels
read one 16-byte task record per CTA even when every list is direct (no separate all-direct fast path).
`SplitKV.auto` caps the forward partials at 64 MiB and the chunk tasks of a plan at `max_task_waves × resident`.

These are heuristic defaults from a few H100 measurements, not tuned optima.

Observed on H100 in a 51-row matrix (B1/H20/D128/K64 unless noted; complete F+B, medians of 10, identical inputs,
selected pairs and NNZ in every arm; O and all gradients within the original tolerances, and the actual plans and
per-head split counts checked, in every row). Split lists are the actual plans recorded by `plan_stats`, forward /
KV-owner backward. Manual C64 splits both directions:

| Case | Split lists F / B | TK F+B (ms) | Manual C64 | Auto C64 |
|---|---|---:|---:|---:|
| hot KV 32K | 0 / 640 | 19.850 | 14.368 | 14.286 (−28.0%) |
| hot KV 76.8K | 0 / 1,500 | 58.448 | 51.059 | 51.065 (−12.6%) |
| hot KV 122.88K | 0 / 0 (2,389 eligible > budget) | 100.209 | 94.879 | 100.379 (+0.2%) |
| hot KV 220K | 0 / 0 (none above the work threshold) | 191.941 | 190.039 | 192.357 (+0.2%) |
| straggler Q 76.8K | 20 / 0 | 50.478 | 50.258 | 49.377 (−2.2%) |
| straggler Q 122.88K | 20 / 0 | 96.660 | 95.838 | 94.760 (−2.0%) |
| straggler Q 220K | 20 / 0 | 202.329 | 200.149 | 198.649 (−1.8%) |
| extreme hot-Q 32K | 0 / 0 (1,280 eligible > budget) | 13.662 | 13.754 | 14.256 (+4.3%) |
| uniform 32K | 0 / 0 | 13.565 | 14.041 | 13.781 (+1.6%) |
| random-input router 32K, routed | — | 14.550 | 14.914 | 14.604 (+0.4%) |
| diagnostic: extreme hot-Q, H2/K4 | 8 / 0 | 1.202 | 0.689 | 0.701 (−41.7%) |

Auto keeps the manual gains where the rules select lists (hot KV at 32K and 76.8K, the forward of a rare long
straggler, splitting only that direction), and it leaves the 122.88K and 220K hot-KV gains (−5.3% and −1.0%
manual) unused: the budget and the work threshold are deliberately conservative. Where it splits nothing it still
runs the planner and the task kernels, and measures 0.2% (long hot KV) to 4.3% (extreme hot-Q, together with
run-to-run variation) over TK; it is not zero-overhead. The H2/K4 row only shows the mechanism, not a model setting.

Real Wan 1.3B captures (setup as in the section below; the sampled fp32 O and route-identity checks also passed),
TK → auto F+B:

| Layout | Split lists F / B | Frozen (ms) | Routed (ms) |
|---|---|---|---|
| seq | 25 / 1 | 8.177 → 8.485 (+3.8%) | 9.904 → 8.727 (−11.9%) |
| cube | 0 / 11 | 10.060 → 10.099 (+0.4%) | 10.711 → 11.344 (+5.9%) |

The routed seq improvement comes from one batch whose TK baseline (9.904 ms) is far above the earlier routed
measurement (8.937 ms) while the same selection frozen regresses, so it is not a stable gain. On the real
captures there is no consistent win in either direction, and it is not a general model speedup. This does not
justify a Wan 1.4 sweep, and none was run.

Validation: at 0309dfb, a fresh H100 compile (77 s) passed all 1,448 tests with 304 design skips, including the 32
policy and auto tests. af5fed3 then added `SplitPolicy` finite and integer validation with no native change; 56
invalid cases and the valid defaults were checked on the actual class (extracted by AST, standard library only),
not by a further CUDA run.

## How it works

**Indexed chunk worklist.** `split_plan` (csrc/vsa/split.cu) runs on the GPU with no host synchronization:
count chunks per list, a cumsum, mark the lists that fit under the slot cap, a second cumsum, then fill a task
array of `{row, edge_begin, edge_end, slot}`. Split chunks come first (heavy first), then one direct task per
remaining list. Kernels run a 1-D grid over the task array; CTAs past the device-side task count exit. A task
reads its kv (or q) blocks from `[edge_begin, edge_end)` of the existing CSR lists; nothing is copied.

**Forward: partial output and LSE merge.** A direct task writes O and the LSE as before. A chunk task writes its
fp32 normalized partial O_i and its log2 LSE_i (TK: by TMA through the idle K/V shared memory). The merge
writes, per query row, `O = Σ 2^(LSE_i − M) O_i / Σ 2^(LSE_i − M)` with `M = max LSE_i`, in chunk order
(deterministic), and the global LSE in natural log, as the unsplit kernels store it:
`LSE = ln 2 · (M + log2 Σ 2^(LSE_i − M))`. Fusion, the backward's delta and its use of the global LSE are
unchanged. On TK the merge keeps padded query rows finite, as TK's backward requires.

**Backward: inverse-CSR KV-owner reduce-add.** The KV-owner backward walks the k2q (inverse CSR) list of each kv
block. Split kv blocks become chunk tasks over contiguous ranges of that list. On TK every task reduce-adds its
dK/dV (and, as before, dQ) into the existing fp32 accumulators with TMA. No partial slots are needed, and the
coarse gradients are still added once, in `finalize`. The fp32 summation order of a split block's dK/dV across
chunks is not fixed, like TK's dQ already. The generic backend writes per-chunk fp32 partials and merges them in
a fixed order instead.

## Results (H100, B1/H20/D128/K64, R16, bf16)

Medians of 10 timed runs after warmup; F / B / F+B are the complete operator with identical inputs, identical
selected pairs and identical NNZ in every arm. "Frozen": the selection is computed once and passed in (router
excluded). "Routed": the PSP router and exact global top-k run inside every forward. Every result row passed the
existing numerical checks. Synthetic masks are scheduling experiments, not claims about router output.

| Case | Tokens | TK F / B / F+B (ms) | Split | Split F / B / F+B (ms) | F+B change |
|---|---:|---|---|---|---:|
| uniform | 32,768 | 3.230 / 10.292 / 13.430 | C64 both | 3.380 / 12.179 / 13.956 | +3.9% |
| uniform | 32,768 | | C64 bwd-only | 3.406 / 10.752 / 14.089 | +4.9% |
| hot KV | 32,768 | 3.263 / 16.703 / 19.721 | C64 both | 3.297 / 11.112 / 14.587 | −26.0% |
| hot KV | 32,768 | | C64 bwd-only | 3.359 / 11.154 / 14.617 | −25.9% |
| hot KV | 220,000 | 25.476 / 168.384 / 192.044 | C64 both | 25.712 / 166.363 / 190.305 | −0.9% |
| hot KV | 220,000 | | C64 bwd-only | 25.920 / 166.387 / 190.079 | −1.0% |
| straggler Q | 220,000 | 40.099 / 162.804 / 202.302 | C128 fwd-only | 36.019 / 162.642 / 198.474 | −1.9% |
| straggler Q | 220,000 | | C64 both | 36.010 / 164.708 / 200.395 | −0.9% |

Earlier hot-KV F+B runs at other lengths: 76,800 tokens 58.419 → 51.112 ms (−12.5%), 122,880 tokens
100.288 → 95.021 ms (−5.3%).

- **Hot KV.** One head's hot kv blocks each walk hundreds of query blocks. Unsplit, those long CTAs set the tail
  of the TK backward; split into chunks, they spread over the GPU. The relative gain shrinks with length. At
  220K the fine backward is 161.38 → 159.40 ms while the coarse backward is only 2.22 ms (independent phase
  timings), so the coarse branch does not explain the dilution. The counters below show the split changes the
  fine backward's DRAM traffic and L2 hit rate a lot at 32K and hardly at all at 220K. Longer sequences already
  provide many waves of long work over a working set far larger than L2; that remains an interpretation.
- **Long query rows.** Splitting the forward helps only when there are few long rows relative to the GPU. With
  H20/K64 the extreme hot-Q case already has 1,280 heavy query tasks and the forward barely changes. A labelled
  diagnostic with H2/K4 has 8, and there the forward gains about 65%; that is not a typical model setting. The
  rare straggler at 220K (one full-length query row per head) gains 10.2% forward (40.099 → 36.019 ms with C128
  forward-only), 1.9% F+B. Splitting the backward there as well (C64 both) makes the backward slower.
- **Uniform.** Planning and merging cost more than the balance they buy: F+B is 3.9–4.9% slower.

### Measured counters (Nsight Compute, hot-KV TK fine backward)

One profiled launch of the TK fine backward kernel per arm (hot KV, B1/H20/D128/K64, same selection as above):
Nsight Compute 2026.1.1 on an H100 80GB HBM3 (GH100), kernel replay, `--cache-control none`,
`--clock-control none`, `--launch-skip 3 --launch-count 1` after warmup; the replay pass count was not emitted
and is unknown. Kernels: `bwd_attend_ker<128, 0>` (unsplit) and `bwd_attend_ker<128, 1>` (split task mode).
Bytes are decimal GB. The duration is that single profiled kernel, not the unprofiled N10 complete-call medians
of the tables above.

| Tokens | Arm | DRAM read (GB) | DRAM write (GB) | Kernel time (ms) | L2 sector hit | Tensor pipe active |
|---:|---|---:|---:|---:|---:|---:|
| 32,768 | tk | 17.550 | 9.616 | 15.335 | 61.13% | 22.84% |
| 32,768 | tk_split_c64 | 4.834 | 2.710 | 9.088 | 87.68% | 38.71% |
| 220,000 | tk | 252.071 | 127.941 | 159.303 | 27.23% | 14.67% |
| 220,000 | tk_split_c64 | 247.771 | 126.374 | 157.261 | 28.93% | 14.80% |

At 32K the split kernel moves 72% less DRAM data (27.17 → 7.54 GB) with a much higher L2 hit rate, and the profiled
kernel is 41% shorter. At 220K DRAM traffic falls 1.5% and the hit rate barely moves, matching the ~1% timing
change. This is direct evidence that splitting changes local reuse and traffic at 32K and not at 220K. It does not
separate cache reuse from the TMA reduce-add write combining, atomic contention or occupancy, all of which the
schedule also changes. Even at 32K the working set does not fit in L2: the raw Q, K and V alone are 503 MB
(20 heads × 32,768 tokens × 128 × 2 bytes × 3) against a 50 MB L2. The tensor-pipe figure is the fraction of
elapsed cycles the tensor pipe was active, not MFU. These counters do not give pure memory time, pure compute time
or their overlap.

### Real captured Q/K/V (Wan 1.3B), negative result

Captured attention inputs, N = 32,760, H = 12, D = 128, bf16; no model is loaded. Selection: a fresh VSA2 PSP
router and exact global top-k on the capture (not the selection stored in the fixture), K = 64. Gate and dO are
seeded synthetic. Two layouts: "seq" cuts 64 consecutive captured tokens per block; "cube" uses
`build_layout((21, 30, 52), R=16)`, which assumes the captured order is raster (T, H, W). The sampled
complete-operator O check against an independent fp32 computation passed for every arm (48 query blocks; relative
error 0.0042 seq, 0.0039 cube).

| Layout | Mode | TK F+B (ms) | C64 both | C64 bwd-only |
|---|---|---:|---:|---:|
| seq | frozen | 8.202 | 8.478 (+3.4%) | 8.695 (+6.0%) |
| seq | routed | 8.937 | 8.853 (−0.9%) | 8.777 (−1.8%) |
| cube | frozen | 9.995 | 10.767 (+7.7%) | 10.151 (+1.6%) |
| cube | routed | 10.348 | 11.495 (+11.1%) | 10.748 (+3.9%) |

The real selections are not skewed enough for splitting to pay. The small routed seq improvement (< 2%) is not a
stable gain: the same selection frozen is slower.

### TK backward options (experimental / diagnostic)

`set_tk_bwd_opts` / `ECHOSPARSE_VSA_TK_BWD_OPTS`, fixed at forward. 0 is the default and runs the original wait
and walk algorithm; the rebuilt kernel is not shown to compile to identical SASS.

- 1, read-wait: after each dQ reduce-add, wait only until shared memory was read. No broad gain (uniform F+B
  13.430 → 14.143 ms; hot KV 19.721 → 19.512 ms).
- 2, rotate: start each k2q walk at a hashed offset. With read-wait (3) it is slower (TK fine backward stage,
  32K: uniform 9.08 → 10.75 ms, hot KV 15.43 → 18.90 ms).
- 4, skip dQ: a diagnostic of the raw binding only. It does not write dQ, so it is never a correct backward;
  `ops` rejects it. In the stage probe it brought the hot-KV fine backward from 15.43 to 7.90 ms, a counterfactual
  that also changes contention and scheduling, so it is not an additive dQ cost.

## Recommended settings

- Default: no `split_kv`, TK backward options 0.
- Selections with hot kv columns or rare long query rows at short to medium length: `split_kv="auto"`. It
  costs up to ~4% where nothing splits and showed no consistent gain on real captures, so measure on your selections.
- Workloads with a few very long query rows: `SplitKV(chunk=128, fwd=True, bwd=False)`.
- Workloads with hot kv columns: `SplitKV(chunk=64, fwd=False, bwd=True)`.
- Measure on the actual selections first; on uniform it is slower and on the real captures above there is no
  stable gain.

## Code and reproduction

- `csrc/vsa/split.cu`: planner and merges. `csrc/vsa/sm90_tk/fine_tk_h100.cu` and `csrc/vsa/generic/attn_*.cu`:
  `TASK` instantiations of the existing kernels (the unsplit instantiation keeps the original algorithm and grid).
- `echosparse/vsa/ops.py`: `SplitKV`, `set_tk_bwd_opts`.
- `tests/test_vsa2_split.py`: the planner against a Python planner; O and q/k/v/gate gradients against the
  oracle and against the unsplit path (TK and generic, skewed masks, empty rows, tails, scratch cap, routed).
- `benchmarks/vsa2_splitkv_bench.py`: frozen and routed arms, `--split-dirs`, `--phases`, `--real-qkv`; run
  `--help` for the cases and labels.

```bash
python benchmarks/vsa2_splitkv_bench.py --tokens 32768 --cases uniform hotkv --chunks 64 --split-dirs both bwd \
    --tk-bwd-opts 0 --arms tk 'tk_split_*' --routed-tokens --out splitkv.jsonl
```
