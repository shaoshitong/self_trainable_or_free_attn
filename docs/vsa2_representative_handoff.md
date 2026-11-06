# VSA2 64/4/1 representative-block handoff

Start with [section 1](#1-start-here). It holds the current state, the arm model and the measurement rules. The numbered
sections after it hold the operator, the build, the rows and the record of each round.

## 1. Start here

### Status

The work is a ThunderKittens/TMA implementation of VSA2's 64/4/1 representative-block fine branch on H100 (sm_90), measured at
16K/H16/D128/RM4 with the same-build harness in [section 7](#7-tests-and-benchmarks).

| Measure | Start | Now | Change |
|---|---:|---:|---:|
| Full forward | 1.3704 ms | about 1.128 ms | -17.7% |
| Full F+B | 4.1789 ms | about 3.873 ms | -7.3% |

These are as reported by the coordinator; this document does not re-derive them. The campaign's count is six accepted changes and
fourteen measured refutations plus one analytical one. The refutations this document records are in
[H100 status](#h100-status), and it does not name all fifteen.

Source of truth: `vsa2_rep/ops.py`, `csrc/rep/rep_api.cpp` and `csrc/rep_tk/rep_tk.cu` for what each config does, and `FWD_CFGS` in
`bench_compare.py` for what each one means.

### Accepted changes

The default path carries six accepted changes, each with a config that restores what it replaced. Each row was checked against
the source.

| Restored by | What it restores | Read in | Accepted alias | Output against CFG 1 |
|---:|---|---|---:|---|
| 20 | the pre-round-4 tier-B gather (32 x 512 B bulk copies) | `rep_tk.cu` dispatch | 10 | generic cross-check |
| 37 | the pre-round-11 tier-A backward (`zeros(kg)`, `zeros(vg)`, reduce-add kernel) | `rep_tk.cu`, `ops.py` | 36 | forward byte-equal |
| 40 | coarse CSR metadata rebuilt on every call | `ops.py` | 39 | byte-equal |
| 43 | representative tables rebuilt from K and V | `ops.py` | 42 | tolerance only |
| 45 | the two top-k budget chains run one after the other | `rep_api.cpp` | 44 | byte-equal |
| 47 | each top-k chain builds its kv->q CSR with cols / colscan / cols | `rep_api.cpp` | 46 | byte-equal |

CFG 13 is also an alias, of the heuristic tier-C backward split, which has no history arm of its own: CFG 21 restores the old wave
policy.

CFG 43 is conditional. The routed operator uses the pooled tables only with the TK backend, when `64 // mid_reps == R` and
`(nb * mid_reps) % 64 == 0`. Outside those it builds from K and V under every config, so CFG 1 vs CFG 43 is an A/A run there.
`rep_fine_attention` always builds from K and V.

### Arm model

Every config is one kind of arm, and the kind decides what it may be compared with and which rows can mean anything. "History" is a
label on top of a kind: CFG 20 is a kernel arm, 37 a backward arm, and 40, 43, 45 and 47 routed arms.

| Kind | Arms | Compare with | Output rule | Rows that may move | Controls: a ratio away from 1 is drift |
|---|---|---|---|---|---|
| Alias | 1, 10, 13, 36, 39, 42, 44, 46 | each other | byte-equal, or invalid | none; this is an A/A run | all forward rows |
| Backward | 21, 22, 37 | an alias; 21 with 22 | forward byte-equal, or invalid | the tier it reaches, `b_direct`, `f_plus_b` | forward rows, other tiers |
| Backward | 23 | 37 only, because it runs on 37's path | as above | as above | as above |
| Routed, byte-equal | 38, 40, 41, 45, 47 | an alias | byte-equal, or invalid | `full_op` `f` and `f_plus_b` | `kernel` and `fine_op` forward rows |
| Routed, byte-equal | 48, 49 | CFG 1 only | byte-equal, or invalid | `full_op` `f` and `f_plus_b` | `kernel` and `fine_op` forward rows |
| Routed, tolerance-equal | 43 | an alias | digests reported; routed cross-check within tolerance | `full_op` `f` and `f_plus_b` | `kernel` and `fine_op` forward rows |
| Kernel | 11, 20, 24, 32, 33, 34, 35 | CFG 1; each other where one variable differs | digests reported; generic cross-check | any | backward kernels and tables |
| Kernel, bulk gather | 0, 6, 8, 9 | CFG 20 or each other | digests reported; generic cross-check | any | backward kernels and tables |
| Diagnostic | 2, 3, 4, 5, 7 | see [section 7](#7-tests-and-benchmarks) | timing only, never a numerical reference | tier-A rows only | native row |

`arms_error` refuses the other pairings, because they would change two things at once. CFG 35 is also held to byte equality against
CFG 1 or it cannot claim a change, and CFG 34 is not, because a fused multiply-add rounds differently.

Byte equality means the `o` and `lse` digests of every forward phase, the digests of the pooled tensors and of the routed operator's `o`
all match. Each arm's routed selection is also digested in `identity.selection`, including `bits_a` and `bits_ab`, and any difference
invalidates the result, so CFG 45, 47 and 49 are checked on their bitmaps directly. CFG 41's replay is only reached through
`rep_attention`, so its check is the output check that the router reproduces the frozen selection. Tolerance equality means the routed
operator's `o`, dQ, dK, dV and dgate agree with the generic backend at the run's shape.

The guards refuse a routed arm unless host code reads it as `strcmp(x, "N")`, an `REP_TK_FWD_CFG` comparison or `cfg == N`, and refuse any
alias, backward or routed arm whose forward dispatch is not the default.

CFG 38 is superseded while the pooled tables apply. Its earlier negative stands, and it must not be re-measured on the current default. `ops.py` still starts `rep_tk_build` on the side stream under CFG 38, but the
pooled branch comes first and never uses that result, so the arm adds a discarded build and cannot measure the overlap it was
written for. Its overlap path runs only where the pooled guard fails. `bench_stages.py`'s `fine.tables` row calls `tk_tables` directly
on the main stream, so it is not expected to go to zero under CFG 38 and says nothing about an overlap.

### Measurement rules

Each of these cost real time.

- **Same-run A/B only.** Runs drift by about 2% against each other. Decide with `bench_compare.py`, which interleaves arms in one job, and
  read the result against that build's own A/A run. `bench_stages.py --compare` and `bench_bwd_stages.py --compare` compare two separate
  runs: they locate cost and say so in their output, with drift from the rows that cannot move.
- **Suites before timing.** Run the CPU, generic and TK suites on the build, the TK suite under the arm as well, before reading a timing. The
  harness cannot enforce this.
- **No per-call stream or event creation in a hot path.** Creating them per call costs more than some of what is being measured. CFG 44 was
  fixed for this, and now creates its side stream and events once per process. `bench_bwd_stages.py` preallocates its events for the same reason.
- **`bench_stages.py`'s `fine` and `fine.tables` describe the fine-only API.** The routed operator builds its tables from the pooled means, so
  they must not be summed into a decomposition of it. Each row carries `describes`, and `attribution.routed_decomposition` is false.
- **Do not sum stages into a total.** Only `f_plus_b` and `b_direct` are measured calls, and `b_estimate` is a subtraction, labelled as one.
- **H200 figures are historical.** Do not compare them with an H100 run.

### What the harness does not prove

- **That an arm took effect.** A byte-equal arm that silently does nothing looks like a neutral result. The guards read source, not the
  compiled extension. A result cannot claim a change if a source is newer than the extension, but that compares file times only.
- **That outputs match below their resolution.** Digests resolve bf16 `o` and fp32 `lse`, and cross-checks are tolerances.
- **Correctness away from the run's shape.** Harness cross-checks use full blocks at one shape. Zero-count blocks, partial blocks and
  empty selections are only in the test suites, so the suites come first.
- **That a difference is real.** `may_claim_op_level_change` says the outputs passed. It does not compare a ratio with the A/A displacement,
  and `control_drift` is reported, not gated.
- **That the compiled forward is the same, or equally fast**, for arms held to byte equality: read the control rows.

### Where things are

| Entry point | Use |
|---|---|
| `tests/test_bench_compare_cpu.py`, `tests/test_reference_cpu.py` | CPU checks, no extension |
| `tests/test_gpu.py`, `tests/test_gpu_tk.py` | generic and TK suites |
| `bench_compare.py` | same-build A/B and the decision |
| `bench_stages.py`, `bench_bwd_stages.py`, `bench_tables.py` | measurement-only stages of the forward, the backward and the table build |
| `bench_graph.py`, `bench_route_graph.py` | measurement-only CUDA graph probes |
| `fwd_cfg.py`, `bwdc_split.py`, `one_launch.py`, `prof.py`, `bench_tk.py`, `bench.py` | single-configuration diagnostics |
| `unverified/`, `patches/` | the original softmax patch, superseded by CFG 1's vector softmax, kept for the record |

## 2. Operator and softmax semantics

The fine branch is a three-tier approximation of VSA2's block-sparse attention:

| Tier | Pair budget | Columns per selected KV block | Meaning |
|---|---:|---:|---|
| A | top `frac_full`, default 10% | 64 | every valid token |
| B | next `frac_mid`, default 20% | `mid_reps`, default 4 | means of 16-token contiguous groups |
| C | remaining pairs, default 70% | 1 | mean of the whole 64-token block |

`RepTiers(frac_full=0.10, frac_mid=0.20, mid_reps=4)` is the 64/4/1 configuration. The two cumulative
budgets are exact global top-k selections over the same router score matrix. Routing remains VSA2's
PSP router. There is no clustering, no token reordering, no Triton implementation, and no reuse of a
previous call's LSE.

For a representative of `c` tokens, K and V are plain means:

```text
k_bar = (1/c) sum_i k_i
v_bar = (1/c) sum_i v_i
```

The representative is one column in the same online softmax as tiers A and B:

```text
logit(rep) = scale * q dot k_bar + log2(c)     # log2 domain in the kernel
```

The `+ log2(c)` term is before the row maximum. It cannot be factored across a row after taking the
maximum because it varies by column. This biases both numerator and denominator and keeps the
representative's mass in the single normalization. The term is a constant of the frozen selection,
so it has no gradient of its own.

The forward is not a hard selection followed by separate softmaxes. A, B, and C columns all enter one
online softmax. `lse` spans that mixed set.

### Backward

Tier A's token dK/dV use VSA2's native KV-owner backward. Representative dK/dV are reduced into the
small fp32 representative tables first. Because the forward representative is a mean, its adjoint is
exactly:

```text
dK_i += dK_bar / c
dV_i += dV_bar / c
```

only for valid tokens in the represented group; padded token slots receive zero. This is implemented
by `rep_unpool`. Tier C uses the same `1/c` rule with `c = blk_cnt`.

The generic backend exposes `bwd_variant=1` and `2`:

- variant 1 splits tier-A dQ into VSA2's KV-owner atomics and walks only tiers B/C in the query-owner
  kernel. Representative dK/dV remain nondeterministic fp32 reductions.
- variant 2 keeps dQ in registers and is deterministic, but redoes tier-A score work.

The TK backend uses its own fixed split of kernels; its Python signature accepts `bwd_variant` but
`_rep_bwd_tk` does not branch on it.

The coarse branch and tanh gate remain VSA2's existing `pool_fwd`, `attn_fwd`, and `fuse_*` path.

## 3. ThunderKittens/TMA kernels

`csrc/rep_tk/rep_tk.cu` implements sm_90a bf16 WGMMA kernels:

- Tier A forward loads native full-block tensors with tensor TMA.
- Tier B forward uses a pre-swizzled representative table. Each possible 128-byte swizzle phase has a
  matching record. By default all 128 threads copy the selected rows into a wgmma-ready shared tile with
  16-byte `cp.async`; CFG 20 restores per-slot 1-D `cp.async.bulk` on an mbarrier.
- Tier C forward loads the contiguous block-mean table with tensor TMA and masks it with the A/B
  bitmap.
- All tiers feed one online softmax and `mma_AB` output accumulation.
- Tier A backward is VSA2's native TK backward on the tier-A k2q lists. By default each kv block's single owner CTA
  stores its dK/dV, so `kg` and `vg` need no memset; CFG 37 restores the zero fill and reduce-add.
- Tier B backward is query-owner. dQ remains in registers. Representative dK/dV are staged and
  reduce-added into the fp32 mid-gradient table with
  `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32`.
- Tier C backward is KV-owner over block means. dQ uses tensor-TMA store-add into the shared fp32 dQ
  accumulator.
- The C backward query range is split across CTAs by FlashAttention's `num_splits_heuristic`; CFG 21 restores
  the older policy of filling one resident wave.

No Triton, clustering, historical LSE, or new router is used.

## 4. Repository, source, and dependencies

The source repository is:

```text
git@github.com:shaoshitong/self_trainable_or_free_attn.git
```

Base commit:

```text
25e2b37e4b9a415c6135011b67ba01d336c82f92
```

The package is self-contained under `experiments/vsa2_representative`, but it still compiles these
VSA2 sources from the enclosing repository:

```text
csrc/vsa/router/{pool,router,topk,scores}.cu
csrc/vsa/generic/{attn_fwd,attn_bwd}.cu
csrc/vsa/fuse.cu
csrc/vsa/sm90_tk/fine_tk_h100.cu              # TK build only
```

ThunderKittens is not vendored here. Use the repository checked out at the commit the VSA2 source tree pins:

```text
6c27e28c8115d1839d9eeeb530913c184a75fc87
```

Environment of the H100 runs, from the sanitized round 2 report:

| Item | Value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3, sm_90, one GPU |
| PyTorch | `2.11.0+cu130` |
| CUDA runtime | 13.0 |
| nvcc | 13.2.78 |
| Compiler | GCC 12.5.0 |

`setup.py` is the authority for build variables. It does not declare a Python dependency set. Use an
already provisioned PyTorch/CUDA environment matching the measured one. The enclosing repository's
`pyproject.toml` describes the parent `echosparse` package and is not the build recipe for this
standalone experiment.

## 5. Build

Start from the experiment directory:

```bash
cd experiments/vsa2_representative
export VSA2_REP_SOURCE="$(cd ../.. && pwd)"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
```

Generic CUDA backend, sm_90:

```bash
VSA2_REP_BUILD=1 \
VSA2_REP_SOURCE="$VSA2_REP_SOURCE" \
TORCH_CUDA_ARCH_LIST=9.0 \
MAX_JOBS=24 \
python setup.py build_ext --inplace
```

TK/TMA backend, sm_90a:

```bash
VSA2_REP_BUILD=1 \
VSA2_REP_TK=1 \
VSA2_REP_TK_ROOT=/path/to/ThunderKittens \
VSA2_REP_SOURCE="$VSA2_REP_SOURCE" \
TORCH_CUDA_ARCH_LIST=9.0a \
MAX_JOBS=24 \
python setup.py build_ext --inplace
```

The variables enforced by `setup.py` and `ops.py` are:

- `VSA2_REP_BUILD=1`: build `vsa2_rep._C`.
- `VSA2_REP_SOURCE`: source root containing `csrc/vsa/vsa2_kernels.h`.
- `VSA2_REP_TK=1`: compile the TK/TMA source and add `-DVSA2_REP_TK`.
- `VSA2_REP_TK_ROOT`: root containing `include/kittens.cuh`.
- `TORCH_CUDA_ARCH_LIST`: the TK build rejects an arch without `9.0a`.
- `CUDA_HOME`: defaults to `/usr/local/cuda`; the linker needs the toolkit driver stub.
- `VSA2_REP_PTXAS_VERBOSE=1`: optional register/spill diagnostics.

The scripts under `scripts/` are optional parameterized wrappers. They have no host-specific defaults and read `VSA2_REP_HOST`,
`VSA2_REP_REMOTE`, `VSA2_REP_SOURCE` and `VSA2_REP_TK_ROOT` from the environment.

## 6. API

The public API is in `vsa2_rep/ops.py`.

Fine branch on a frozen selection:

```python
from vsa2_rep import PHASE_ALL, RepTiers, rep_fine_attention, route_tiers

sel = route_tiers(q, k, blk_cnt, RepTiers(frac_full=0.10, frac_mid=0.20, mid_reps=4))
o, lse = rep_fine_attention(
    q, k, v, blk_cnt, sel,
    mid_reps=4,
    phases=PHASE_ALL,
    backend="tk",
)
o.sum().backward()
```

Full operator with VSA2's coarse branch and gate:

```python
from vsa2_rep import RepTiers, rep_attention

out = rep_attention(
    q, k, v, gate, blk_cnt,
    R=16,
    tiers=RepTiers(frac_full=0.10, frac_mid=0.20, mid_reps=4),
    backend="tk",
)
```

`backend` is `"auto"`, `"generic"`, or `"tk"`. `"tk"` requires:

- a build made with `VSA2_REP_TK=1`;
- an sm_90 device;
- bf16 q/k/v.

`"auto"` falls back to generic outside those conditions. `phases` is a bitmask over `PHASE_A`,
`PHASE_B`, and `PHASE_C`. `mid_reps` must be one of 1, 2, 4, 8, 16. Inputs are contiguous
`[B,H,L,D]`, `L = NB*64`, block-major layout, with padded rows at the end of each block.

## 7. Tests and benchmarks

CPU tests need no extension:

```bash
cd experiments/vsa2_representative
python tests/test_reference_cpu.py
```

Generic GPU suite:

```bash
<gpu-runner> python tests/test_gpu.py
```

TK GPU suite:

```bash
<gpu-runner> python tests/test_gpu_tk.py
```

The TK suite checks tables and swizzle, per-tier and all-tier forward/LSE against fp64, agreement with
native TK on tier A, TK-vs-generic backward dQ/dK/dV, and the routed autograd operator. Recorded
successful coverage includes D=64/128, `mid_reps=2/4/8`, and an incomplete final block.

`test_masked_and_empty` (generic and TK suites) adds zero-count blocks, all-masked first and intermediate
tiles, empty selections and per-tier gradients. It is part of the suites from round 2 on.

First fair kernel/op benchmark:

```bash
<gpu-runner> python bench_tk.py \
  --len 16384 --heads 16 --dim 128 --mid-reps 4 \
  --warmup 10 --iters 30
```

Generic op benchmark:

```bash
<gpu-runner> python bench.py \
  --len 16384 --heads 16 --dim 128 --batch 1 --iters 20
```

`bench_tk.py` uses warmup=10/iters=30 by default. `bench.py` calls its timing helper with the helper's
warmup=5 and the selected iteration count. Both allocate deterministic BF16 inputs under
`torch.manual_seed(0)`, use a full-block `blk_cnt`, route once, and time a loop with CUDA events and
one final synchronize.

Targeted timing helpers:

```bash
<gpu-runner> python fwd_cfg.py
<gpu-runner> python bwdc_split.py
```

Configurations. The config table in `bench_compare.py` describes the kernel that has the CFG 20 history arm.
Registers and stack bytes are D=128 `ptxas` figures: CFG 1 from CFG 10's round 4 build, the other measured rows
from source `3c291f2`, before the gather changed, with CFG 20's taken from CFG 1 at that source. None has been read
from a build containing CFG 20.

| CFG | Kernel | Softmax | MINB | Tier-B gather | Registers | Stack bytes |
|---:|---|---|---:|---|---:|---:|
| 1 (default) | `fwd_ker`, all tiers; heuristic tier-C backward split; tier-A dK/dV stored by each kv block's single owner CTA, no `kg`/`vg` memset | TK vector | 3 | `cp.async` | 166 | no spills |
| 10, 13, 36, 39, 42, 44, 46 | aliases of CFG 1, kept so old logs stay reproducible | as CFG 1 | 3 | as CFG 1 | as CFG 1 | as CFG 1 |
| 21 | CFG 1 forward; tier-C backward split by the old wave policy | as CFG 1 | 3 | as CFG 1 | as CFG 1 | as CFG 1 |
| 22 | CFG 1 forward; tier-C backward split forced to 4 | as CFG 1 | 3 | as CFG 1 | as CFG 1 | as CFG 1 |
| 23 | CFG 1 forward; tier-A backward `__launch_bounds__(128, 3)` probe, on CFG 37's reduce-add path | as CFG 1 | 3 | as CFG 1 | as CFG 1 | as CFG 1 |
| 37 | CFG 1 forward; the pre-round-11 tier-A backward: `zeros(kg)`, `zeros(vg)` and the reduce-add kernel | as CFG 1 | 3 | as CFG 1 | as CFG 1 | as CFG 1 |
| 38 | CFG 1 asking for the table build on a side stream during `route_tiers`; superseded, see below | as CFG 1 | 3 | as CFG 1 | as CFG 1 | as CFG 1 |
| 40 | history arm: coarse CSR metadata rebuilt on every call | as CFG 1 | 3 | as CFG 1 | as CFG 1 | as CFG 1 |
| 41 | CFG 1 with `route_tiers` replayed from a captured CUDA graph | as CFG 1 | 3 | as CFG 1 | as CFG 1 | as CFG 1 |
| 43 | history arm: representative tables rebuilt from K and V | as CFG 1 | 3 | as CFG 1 | as CFG 1 | as CFG 1 |
| 45 | history arm: the two top-k budget chains run one after the other | as CFG 1 | 3 | as CFG 1 | as CFG 1 | as CFG 1 |
| 47 | history arm: each top-k chain builds its kv->q CSR with cols / colscan / cols | as CFG 1 | 3 | as CFG 1 | as CFG 1 | as CFG 1 |
| 48 | CFG 1 with the `pool_kernel` pass vectorised to two column pairs per thread | as CFG 1 | 3 | as CFG 1 | as CFG 1 | as CFG 1 |
| 49 | CFG 1 with the router's mask, cut and emit kernels merged into one | as CFG 1 | 3 | as CFG 1 | as CFG 1 | as CFG 1 |
| 11 | measured negative arm: `fwd_kdb_ker`, CFG 1 plus a double-buffered K tile | TK vector | 3 | `cp.async` | not recorded here | not recorded here |
| 24 | measured negative arm: `fwd_pipe_ker`, CFG 11's K double buffer plus a two-stage softmax, the next tile's QK issued before the current softmax finishes | TK vector | 2 | `cp.async` | not recorded here | not recorded here |
| 32 | measured negative arm: `fwd_ring_ker`, two CTAs/SM, K and V both double-buffered | TK vector | 2 | `cp.async` | not recorded here | not recorded here |
| 33 | measured negative arm: `fwd_ring_ker`, two CTAs/SM, K and V both triple-buffered | TK vector | 2 | `cp.async` | not recorded here | not recorded here |
| 34 | CFG 1 with the log2(e)*scale multiply and the tier-B/C bias add folded into one FFMA | TK vector | 3 | `cp.async` | not recorded here | not recorded here |
| 35 | CFG 1 skipping the O rescale when no row's running maximum moved in the tile | TK vector | 3 | `cp.async` | not recorded here | not recorded here |
| 20 | the pre-round-4 forward, with CFG 1's backward | TK vector | 3 | bulk | 168 | 24 |
| 6 | `fwd_ker`, all tiers; a softmax pair with CFG 20 | hand-written | 3 | bulk | 168 | 16 |
| 8 | `fwd_ker`, all tiers | TK vector | 2 | bulk | not recorded here | not recorded here |
| 9 | `fwd_ker`, all tiers | hand-written | 2 | bulk | not recorded here | not recorded here |
| 0 | `fwd_ker`, all tiers | TK vector | 4 | bulk | 128 | 184 |
| 2 | `fwd_ker`, `A_ONLY` | TK vector | 4 | n/a | 128 | 8 |
| 3 | `fwd_ker`, `A_ONLY` | TK vector | 3 | n/a | 160 | 0 |
| 4 | `nat_copy_ker` | native | 4 | n/a | 128 | 0 |
| 5 | `nat_copy_ker` | hand-written | 4 | n/a | 126 | 40 |
| 7 | `nat_copy_ker` | TK vector | 4 | n/a | 128 | 0 |

CFG 2/3 use their `A_ONLY` kernel only for `phases=A`; other phases run the default kernel. CFG 0/6/8/9/20 keep the
bulk gather, so CFG 6 is a softmax-only pair with CFG 20, not with CFG 1. `settings.fwd_kernels_in_source` records
the launched template. CFG 24, 32 and 33 must launch something other than the default or an earlier arm's kernel.
CFG 32 and 33 are also refused if the launched kernel's `__launch_bounds__` does not say 2. CFG 34 and 35 launch the default
kernel on purpose, with a host-set `diag` flag; they are refused without that flag assignment or if they launch anything
else. CFG 21/22/23 are measured negative or neutral arms; see H100 status.

Occupancy of the forward arms, D=128. Cells marked `not recorded here` come from a build's `ptxas` output:

| CFG | Registers | Spills | Dynamic shared memory | Register limit | Shared-memory limit | Implied CTAs/SM |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 166 | no | 50,176 B | 3 | 4 | 3 |
| 11 | not recorded here | not recorded here | 66,560 B | 3, while `__launch_bounds__(128, 3)` | 3 | 3 |
| 24 | not recorded here | not recorded here | 66,560 B | not recorded here | 3 | 3, as reported in round 7 |
| 32 | not recorded here | not recorded here | 82,984 B | not recorded here | 2 | 2 |
| 33 | not recorded here | not recorded here | 115,256 B | not recorded here | 2 | 2 |
| 34 | not recorded here | not recorded here | 50,176 B | 3, while `__launch_bounds__(128, 3)` | 4 | 3 |
| 35 | not recorded here | not recorded here | 50,176 B | 3, while `__launch_bounds__(128, 3)` | 4 | 3 |

These limits are derived, not measured, and the implied figure is the smaller of the two. The register limit is 4
CTAs/SM up to 128 registers, 3 up to 168 and 2 up to 255, the caps stated in the kernel's comments. The shared-memory
limit uses the kernel's formula below, which agrees with round 6: 113,000 B allows 2.
`__launch_bounds__(128, 3)` makes the compiler stay within 168 registers, so pressure there shows as spills, as in
CFG 20. A bound of 2 allows up to 255 registers without forcing 2 CTAs/SM, and resident occupancy then follows the
register count: CFG 24 is written with a bound of 2 and its source comment says 2 CTAs/SM, yet it was reported at 3.
It stays at 3 only if its `ptxas` count is at most 168, which is not recorded here.

CFG 32 and 33 are two-CTA kernels, not three-CTA arms. Their shared memory, with a 16 KB tile of 64 x 128 bf16:

| CFG | Q + K + V tiles per CTA | Per CTA | At 2 CTAs/SM | Limit |
|---:|---|---:|---:|---:|
| 32 | 16 + 32 + 32 | 80 KB | 160 KB | 232,448 B |
| 33 | 16 + 48 + 48 | 112 KB | 224 KB (229,376 B) | 232,448 B |

These KB are 1,024 B. The kernel's own limit counts 1 KB reserved per CTA: `CTAs * (dynamic + 1,024) <= 233,472` B. CFG 33's
`ring_cfg` therefore puts its barriers in dynamic memory with 512 B of slack, allocating 115,256 B, because the other
kernels' layout would exceed the limit by 112 B. CFG 32 allocates 82,984 B. `fwd_impl` refuses to launch either arm
unless the occupancy query reports 2 resident CTAs/SM.

CFG 34 and 35 are premised on staying within 168 registers. Both launch the default kernel at a bound of 3, which makes the compiler respect that cap, so too much pressure shows up as spills with the count still at or below
168: CFG 20 is the example. Record both figures in the occupancy table from the build's `ptxas` output. If either arm
reports more than 168 registers, or any spills that CFG 1 does not have, its premise failed: record that here and do
not read its timing as a pure instruction-stream change.

CFG 35's premise, that `alpha` is exactly 1.0f when no maximum moved, is unverified, and no CPU check can settle it
because the value comes from the device's `exp2` of `m - mu`, with `mu` clamped to the lowest finite float.

- With a finite `m` that did not move, `m - mu` is +0. That `exp2` returns exactly 1.0f there is assumed, not checked.
- While a row has seen only masked columns, `m` stays -inf and `alpha` is 0, not 1. `o` is still 0 on such a row, so
  skipping its rescale changes nothing, and the arm depends on that.
- `l` must take the same decision as `o`. Mathematically `alpha` is 1 when `m` did not move, so skipping both is
  correct even if `exp2` were inexact, but rescaling only `l` would let `o / l` drift if `exp2` is inexact.

What the harness does check: compared with CFG 1, CFG 35's forward phase outputs must be byte-identical or it
cannot claim a change. This resolves bf16 `o` and fp32 `lse`, and the skip does not touch `lse` directly, so an `alpha`
within bf16 rounding of 1.0f passes. Read CFG 35 with that in mind. CFG 34 is not held to byte identity, because a fused
multiply-add rounds differently; the generic cross-check and the TK suite are its checks.

The accepted changes and the arm model are in [section 1](#1-start-here).

`REP_TK_BWDC_SPLITS=<positive>` forces the tier-C split and `REP_TK_BWDC_SPLITS=wave` selects the old policy. At
D=128, B=1, H=16 and 132 SMs, read from the kernel's formulas, many pairs of policies coincide, so a comparison
at a shape where they do is an A/A run:

| Length | Default (heuristic) | Wave (CFG 21) | Forced (CFG 22) |
|---:|---:|---:|---:|
| 16K | 4 | 4 | 4 |
| 32K | 2 | 2 | 4 |
| 48K | 4 | 1 | 4 |
| 64K | 1 | 1 | 4 |

`REP_TK_FWD_CFG` is cached on first use, so one process measures one value. A value the kernel does not
dispatch on silently runs the default kernel, so `bench_compare.py`, `bench_tk.py`, `fwd_cfg.py`, `one_launch.py`
and `bench_stages.py` check `rep_tk.cu` on disk. They refuse a source without `cfg == 20`, a value not written as
`cfg == <value>` (a backward arm is written `rep_cfg() == <value>`; the aliases may have no line), an alias or
backward arm whose forward dispatch differs from the default, and a CFG 20 or 24 that launches the default. The check
reads source, not the compiled extension, and does not look at the VSA2 sources compiled alongside it.

CFG 4/5/7 are measurement code. They ignore `phases` and run tier A only, write no `lse` and do not mask
padded output, so they are timing diagnostics and never a numerical reference. `fwd_cfg.py` prints only
their tier-A row, and `bench_tk.py` refuses them. For the same reason, the `A|B|C` values recorded for the
older CFG 4/5 in `results/tk-later-iterations.txt` are tier-A timings.

### Same-build comparison

`bench_compare.py` compares two configurations of one build. It runs one process per arm in ABBA order,
with `q`, `k`, `v`, `dO`, gate and selection from one seed, and writes one JSON result. Each arm records
hardware, source and extension hashes, shape, configuration, per-batch timings and `nvidia-smi` state. It
marks the result invalid if the arms differ in any of the hashes, inputs, selection, shape or settings.
Results record no node name, address or raw device id. Arms are matched on a hash of the GPU id.

`<gpu-runner>` below stands for however the coordinator launches a command on one GPU with the provisioned
Python. The harness requires exactly one visible GPU.

```bash
# representative semantics, baseline CFG1 by default, candidate required: warmup 10, iters 30
<gpu-runner> python bench_compare.py --candidate REP_TK_FWD_CFG=11 --out results/raw/cfg1-cfg11-16k.json

# equal arms give the A/A noise floor
<gpu-runner> python bench_compare.py \
  --baseline REP_TK_FWD_CFG=1 --candidate REP_TK_FWD_CFG=1 --out results/raw/aa-16k.json

# diagnostic triple as two pairs against CFG4: tier A, warmup 10, iters 50, TIMING ONLY
<gpu-runner> python bench_compare.py --profile controlled-softmax \
  --baseline REP_TK_FWD_CFG=4 --candidate REP_TK_FWD_CFG=5 --out results/raw/cfg4-cfg5-16k.json
<gpu-runner> python bench_compare.py --profile controlled-softmax \
  --baseline REP_TK_FWD_CFG=4 --candidate REP_TK_FWD_CFG=7 --out results/raw/cfg4-cfg7-16k.json
```

Export `VSA2_REP_SOURCE` and `VSA2_REP_TK_ROOT` as for the build, so the VSA2 source hashes and the
ThunderKittens commit are recorded.

| Row label | Meaning |
|---|---|
| `scope=kernel` | one kernel, tables prebuilt; `bwd_stage` rows are never summed into a call |
| `scope=fine_op` | complete `rep_fine_attention` call on the frozen selection |
| `scope=full_op` | complete `rep_attention` call: router, coarse branch, gate |
| `stage=f` | forward on the same grad-tracking leaves as `f_plus_b`; the graph is dropped |
| `stage=f_plus_b` | measured forward + backward + grad reset, no synchronize inside the loop |
| `stage=b_direct` | measured backward: fresh graph per sample, forward finished before the interval, grads reset after it |
| `stage=b_estimate` | `median(f_plus_b) - median(f)`, `measured=false`, an estimate |
| `work=tier_a_only` | 10% of the pairs; not equivalent to the 64/4/1 operator |
| `work=tier_b_only`, `tier_c_only`, `tier_subset` | part of the tile stream, tables prebuilt; locates cost, and these rows do not add up to `A\|B\|C` |
| `forward_outputs_byte_identical` | whether the arms' `o` and `lse` have equal digests for every forward phase row; meaningful only if the A/A run is true |
| `control=true` | does not read the compared setting; a ratio away from 1 is run drift |

Backward kernels and table builds read no forward setting, so their kernel rows are controls, with two exceptions:
CFG 21, CFG 22 and a forced split reach `kernel.tk.bwd.C`, and CFG 23 and 37 reach `kernel.tk.bwd.A`. Between CFG 1, its
aliases and the backward arms, one forward kernel by definition, the forward-only rows are controls.

The forward-identical check for those arms has three parts. The forward dispatch in `fwd_impl` must launch the
default. `o` and `lse` of every forward phase row, the pooled tensors and the routed `o` must be byte-equal between arms, or the result is invalid. A
backward arm can be compared only with CFG 1, its aliases, or an arm reaching the same tier, and not on top of
`REP_TK_BWDC_SPLITS`. This does not prove the compiled forward is the same code, or equally fast: a change that
keeps the bytes passes, so also read the forward control ratios. The byte check is meaningful only if the A/A run
reports `forward_outputs_byte_identical` true. A backward-arm result is also invalid if `kernel.tk.bwd.A`,
`kernel.tk.bwd.C`, or the `A|B|C` fine or routed `b_direct` or `f_plus_b` rows are missing.

`b_direct` cannot reuse one graph: backward releases the representative tables, so `retain_graph=True`
would not make a second backward valid. Its synchronizes add idle gaps the `f_plus_b` loop does not have,
so it need not equal the estimate. The older `bench_tk.py` forward rows use leaves without grad tracking;
compare its numbers with its own runs only. `bench_tk.py` has no directly timed backward.

`claims.may_claim_op_level_change` is true only for the representative profile, after the arm's forward,
lse and dQ/dK/dV agree with the generic backend at the run's shape, with the test suite's tolerances. The
CPU, generic and TK suites are still required on that build. The controlled profile can never set it.
Diagnostics that skip bias loads or change the load path (`diag != 0`, used only by `fwd_cfg.py`) locate
cost. They cannot establish numerical or end-to-end acceleration.

No counters are collected, and event timing does not yield MFU or stall figures.

Helper checks, no GPU, build or `torch`:

```bash
python3 tests/test_bench_compare_cpu.py
```

### Profiling one kernel

`one_launch.py <stage> [reps]` runs one kernel after 10 warmup launches, on deterministic 16K/H16/D128/RM4
inputs, with tables and intermediates built first. The profiler selects the kernel by name, and the script
prints the stage, shape, configuration, kernel name and per-name invocation numbers.

| Stage | Kernel | Work |
|---|---|---|
| `tk_fwd_abc` | `fwd_ker` | all tiers |
| `tk_fwd_b`, `tk_fwd_c` | `fwd_ker` | partial: one tier |
| `tk_bwd_a`, `tk_bwd_b`, `tk_bwd_c` | `bwd_attend_ker`, `bwd_b_ker`, `bwd_c_ker` | partial: one tier |

```bash
REP_TK_FWD_CFG=1 <gpu-runner> python one_launch.py tk_fwd_abc
```

It is for kernel-scope diagnosis. It does no timing, is not a correctness check and is not a full-operator
benchmark. It refuses CFG values that do not run all tiers. The generic stages remain available.

### H100 status

Round 2, source `3c291f2`, 16K/H16/D128/RM4: the fresh build and the CPU, generic and TK suites passed, the
TK suite also with CFG 6. CFG 6 vs CFG 1 gave no measured operator improvement above A/A displacement: routed
full-operator F+B ratio 1.002 against an A/A ratio of 0.997. No second length was run.

Round 3: the MINB 2 controls were slower than MINB 3, 22-23% on the forward kernels and 5.0% on fine F+B,
against a +0.9% A/A displacement. The softmax and register-cap line of work is closed. On that build,
kernel-scope forward was tier A 0.574, tier B 0.217, tier C 0.143 and `A|B|C` 0.862 ms. Forward was 1.376 ms
for the routed operator and 0.916 ms for the fine path on a frozen selection, leaving about 0.46 ms to attribute.

Round 4: CFG 10 won against the then-default. `kernel.tk.fwd.B` 0.2172 to 0.1668 ms (0.768), `A|B|C` 0.8613 to
0.7946 (0.923), fine forward 0.910 to 0.809, fine F+B 3.804 to 3.704 and full-op F+B 4.179 to 4.078, against an
A/A displacement of 0.999-1.009. The TK suite passed under CFG 10, with 166 registers and no spills. Its gather
is now the default. CFG 1 figures up to this round describe today's CFG 20.

Round 5, source `3da1d0c`: CPU 7, generic 287 and TK 349 checks passed, and the TK suite passed with CFG 10, 13 and
20 selected. The promoted default reproduced round 4: CFG 20 was 30% slower on the B kernel, 12% on fine forward and
2.3% on F+B, and at 32K full forward was 0.933. CFG 11, the K double buffer, gave no gain: full F+B 1.0003 and full
forward 1.011, against an A/A floor of 1.004 and 1.000. It stays documented as a negative arm.

In that run, A/A on forward rows was wider than on full-op rows: kernel `A|B|C` 1.0143 and fine `f` 1.0131, against
1.004-1.007. In milliseconds that is about 0.011 on those forward rows and 0.016-0.029 on 4.08 ms full F+B, so the
wider ratio reflects a shorter row, with no sign of larger absolute drift. These summary figures cannot separate
ordering, thermal or allocator effects; the per-batch `ms` and `telemetry_*` fields in the result could.

Backward stages: full-op `b_direct` is about 2.98 ms beside 4.08 ms F+B. They are separate measurements, and
`b_direct` includes idle gaps from its synchronizes, so `b_estimate` is the one that subtracts from F+B.

Round 6: the backward line is closed for now. CFG 21 F+B was 1.0002 and CFG 22 1.0013, neutral at 16K, where all
three split policies choose 4 at 132 SMs, so this says nothing about 32K or 48K. CFG 23 was a measured negative: F+B
1.350 and `kernel.tk.bwd.A` 1.874.

The tier-A line is closed: three CTAs/SM are not reachable without a redesign. dK and dV alone hold 128 registers per thread
against the 168 budget of a bound of 3, and CFG 23's forced-168 measurement shows what that cap costs. Shared memory also blocks
it as laid out: the layout needs about 107 KB of tiles per CTA and allocates 113,000 B, against 76,800 B at three CTAs by the
formula above. So CFG 23 itself ran at 2 CTAs/SM. It measures the register cap's cost, not a three-CTA kernel.

Round 7: CFG 24 was reported to keep 3 CTAs/SM and was 29% slower on the forward kernel, a measured negative. Reading
that as free of any occupancy effect depends on its register count; see the occupancy note above.

Round 8: both deep-ring arms are measured negatives, and both TK suites passed. CFG 32 (2 CTAs, 2-deep) was 1.076 on
the forward kernel and 1.022 on F+B; CFG 33 (2 CTAs, 3-deep) was 1.110 and 1.025; the A/A floor was 0.981-1.042.
Depth is not the missing lever, so do not re-run this sweep. `A|B|C` kernel time, as reported across rounds:

| Configuration | `A\|B\|C` kernel |
|---|---:|
| 3 CTAs, 1 stage (current) | 0.78 ms |
| 3 CTAs, 2 K stages | 0.78 ms |
| 2 CTAs, 2 stages | 0.86 ms |
| 2 CTAs, 3 stages | 0.89 ms |

The two-CTA figures imply a CFG 1 of about 0.80 ms in their own run, so read the ratios above for that comparison.

Round 9: CFG 34 and 35 were both neutral. CFG 34 was 0.995 on the forward kernel and 1.000 on F+B, and CFG 35 0.996 and
0.999, against an A/A floor of 0.983-1.011.

Round 11: CFG 36, the single-writer dK/dV store without the `kg` and `vg` memsets, was accepted. Routed `b_direct` went from
2.9389 to 2.8597 ms (0.9731) and F+B from 4.1363 to 4.0495 ms (0.9790), against an A/A floor of 1.0010-1.0017 on those
rows, with all suites passing. `bench_bwd_stages.py` showed about 0.135 ms of memsets directly. The kernel writes zeros itself
for a kv block with an empty list, and `qg` is still zeroed because dQ is still reduce-added.

That store is now the default. CFG 36 is its alias and CFG 37 is the old path, so CFG 37 vs CFG 1 reproduces the change, with
the same forward-identical checks as CFG 21/22/23. In `bench_bwd_stages.py` the `kg` and `vg` rows are now `present=false` under
CFG 1 and timed under CFG 37. CFG 23 runs on CFG 37's path, so it is comparable only with CFG 37.

Round 12: CFG 36 is confirmed as the default: CFG 37 read 1.0369 on `fine_op` `b_direct` and 1.0254 on `full_op` F+B.
`bench_graph.py` measured a replay of one forward and backward at 3.9256 ms against 4.0435 ms eager, 0.118 ms or 2.9% of F+B,
above the run's 0.030 ms floor, with a bit-identical forward and one bf16 ulp on dQ, dK and dV. It is available, not adopted.

CFG 48 and CFG 49 are in the source and are not defaults. No outcome is recorded here yet. On one build, with the suites passing on it:

```bash
REP_TK_FWD_CFG=48 <gpu-runner> python tests/test_gpu_tk.py
REP_TK_FWD_CFG=49 <gpu-runner> python tests/test_gpu_tk.py
<gpu-runner> python bench_compare.py --candidate REP_TK_FWD_CFG=1 --out results/raw/aa-16k.json
<gpu-runner> python bench_compare.py --candidate REP_TK_FWD_CFG=48 --out results/raw/cfg1-cfg48-16k.json
<gpu-runner> python bench_compare.py --candidate REP_TK_FWD_CFG=49 --out results/raw/cfg1-cfg49-16k.json
```

Read `status` first: differing selection, pool or forward digests make the result invalid. Then read `full_op.tk.routed.f` and
`f_plus_b` against the A/A displacement.

### CUDA graph measurement

`bench_graph.py` is measurement only and does not make the graph path a supported API. It captures one full `rep_attention(...)`
plus its `backward()` into a CUDA graph on a non-default stream, on static input tensors, then replays it. Every row is
`measurement_only`: these times are not a performance claim and must not be added to the kernel rows.

| Row | Contents |
|---|---|
| `eager`, `eager.repeat` | routed F+B on the default stream, first and last; their spread is the floor |
| `eager.same_stream` | the same call on the capture stream with autograd multithreading off |
| `graph.replay` | replays in a loop with no synchronize inside, the same boundaries as `eager` |
| `graph.replay_with_copy` | the same with inputs copied into the captured buffers each step, which a real caller must do |
| `graph.event_pair` | an event pair around one replay |
| `graph.host_call` | host wall time of the `graph.replay()` call |

An event pair sits on the GPU timeline, so `graph.event_pair` is the replay's execution time, not its launch cost.
`graph.host_call` is the closest measure of the launch cost itself. `attribution.recoverable_ms` is `eager` minus `graph.replay`,
derived and marked `measured=false`, and a value inside the floor is not a saving.

Capture runs with autograd multithreading off, because the backward would otherwise launch from a worker thread on the default
stream, outside the capture. A graph replays what was captured: the router and the tables re-run on the same tensors, the
outputs live in the graph's memory, and any host decision is frozen at capture. Replaying with different inputs or shape is not
something this measures.

`outputs` reports max absolute differences on `o`, `dq`, `dk` and `dv` for replay against eager and for eager against eager. The
second is the floor, since dQ, dK and dV include fp32 reductions in a nondeterministic order. `equivalent` uses the generic
cross-check's tolerances. If capture or replay raises, the reason is recorded and printed, the graph rows are marked
`available=false`, and the eager rows are still produced.

`equivalent` runs on the captured inputs, so it cannot see a kernel that escaped the capture: that kernel ran once during capture, its output
stays right for those inputs, and the replay also looks faster for skipping it. `fresh_input_equivalent` writes different values into the
captured inputs and compares a replay with an eager call on them. `recoverable_ms` is only meaningful if it is true. Round 12's figure predates
this check, so it should be rerun before being relied on. `graph.replay_with_copy` copies the original values back in, so it times the copy without
testing it. `bench_route_graph.py` has its own check, `replay_vs_eager_changed_input`.

### Backward stages

`bench_bwd_stages.py` splits one real `backward()` on one configuration, for `fine` (`rep_fine_attention`) and `full`
(routed `rep_attention`). Every row is `measurement_only`: these times are not a performance claim and must not be added
to the kernel rows. For that call only, the module's handles on the extension and on torch are proxies that record a CUDA
event before and after each native call and each `zeros` or `zeros_like`, with no synchronize inside the timed region.
No kernel, argument or call order changes.

| Row | In the sum | Contents |
|---|---:|---|
| `call.row_dot` | yes | delta, `fine` only; `fuse_bwd` produces it in `full` |
| `call.tk_native_bwd`, `call.rep_tk_bwd_b`, `call.rep_tk_bwd_c` | yes | the tier kernels in place |
| `call.finalize`, `call.rep_unpool` | yes | dQ finalize, and the unpool with its allocations |
| `call.fuse_bwd`, `call.attn_bwd` | yes | `full` only: fusion, and the coarse backward |
| `call.zeros.token`, `#2`, `#3` | yes | `qg`, `kg`, `vg` in `ops.py`'s order, each with its allocator |
| `call.zeros.mid`, `#2`, `call.zeros.tail`, `#2` | yes | `dk_mid`, `dv_mid`, `kgt`, `vgt`, each with its allocator |
| `call.zeros.other` | yes | `full` only: the coarse `dq` |
| `gap.*` | yes | other Python between two of those calls |
| `total`, `total.repeat` | no | `backward()` with only two events, first and last; `bench_compare.py`'s `b_direct` boundaries |
| `isolated.*` | no | the tier kernels back to back with fixed intermediates |
| `context.*` | no | `f` and `f_plus_b` loops, for the F+B - F estimate |
| `event_pair` | no | two consecutive events on an idle stream |

A table a configuration no longer zeroes keeps its row, at 0.000 with `present=false`, so CFG 36's `kg` and `vg` rows read
zero and `qg`'s does not. The token rows are named by order: if `ops.py` ever drops `qg` while keeping another, their
labels shift. `--compare` prints per-row ratios on one build and keeps rows only one run has. `qg_kg_vg_zeroing_ms` adds
the token rows, and `table_zeroing_and_allocator_ms` adds all of the zeroing rows.

Three things cannot be separated without changing how the backward is called, so these are the closest honest groupings:

- `torch.zeros` cannot be split from its allocator, and native calls that allocate internally, such as `rep_unpool`, keep
  that in their `call.*` row.
- Launch overhead and allocation inside a native call stay in its `call.*` row.
- Host work appears only when the stream has drained while it runs, so host cost behind a long kernel reads as zero.

Each row has per-batch medians and a spread. `attribution` is derived and marked `measured=false`, per scope. It gives the
residual of `total` against the steps, a floor from the spread of the `total` batches, the time outside the tier kernels,
and the F+B - F estimate. A residual outside the floor is reported, since the events themselves can perturb the call:
compare `event_pair`. If `b_estimate_ms` is well below `total`, part of the gap comes from the idle time around a direct
backward, and will not be found in any step.

### Full-operator forward stages

`bench_stages.py` times the stages of the routed forward on one configuration. Every row is `measurement_only`:
these times are not a performance claim and must not be added to the kernel rows.

| Row | In the sum | Contents |
|---|---:|---|
| `pool` | yes | `pool_fwd`; one call makes the router pools and the coarse means |
| `route` | yes | router lse and scores, top-k at two budgets, tier-B lists; one native call |
| `route.one_budget` | no | different work: one top-k, no tier-B lists |
| `coarse.meta`, `coarse.attn` | yes | the coarse CSR, and `attn_fwd` on the coarse means |
| `fine` | yes | `rep_fine_attention` on the built selection |
| `fine.tables`, `fine.kernel` | no | inside `fine`: tables and log counts, and `rep_tk_fwd` |
| `fuse` | yes | `fuse_fwd`; the forward has no finalize |
| `total`, `total.repeat` | no | `rep_attention`, first and last; their spread is the run's floor |

Table build and the flattened tier-B log counts run inside the fine path, so they are not part of `route` or of
the gap between `total` and `fine`. `attribution` is derived and marked `measured=false`. A residual outside
the floor is reported rather than spread over the stages: isolated loops overlap differently on the stream, and
the operator's own autograd wrapper cannot be timed on its own.

Each row carries `describes`: `routed`, `fine_only` or `both`. `fine` and `fine.tables` describe the fine-only entry point and must not
be summed into a decomposition of the routed operator, which builds its tables from the pooled means. `attribution.sum_of_parts_ms` still
adds `fine`, so it is reported and not asserted. `pool` and `coarse.meta` also time the non-default calls: the routed operator now uses
`pool_fwd_vr` and the cached metadata.

Every row keeps its meaning under every all-tier configuration. `reads_fwd_cfg` marks the rows that can change with
`REP_TK_FWD_CFG` (`total`, `fine`, `fine.kernel`, `route`, `route.one_budget`, `pool`); the others should match another configuration within the floor.
Under CFG 13, 21, 22 and 23 each row carries a `config_note`, because the forward repeats CFG 1's. `--compare` prints per-row
ratios and refuses runs that differ in build, inputs, shape, timing or hardware. The round 4 stage figures came from
an earlier build with the old default, so they cannot serve as the CFG 1 side.

## 8. Measured evidence

All principal measurements are NVIDIA H200, B=1, D=128, BF16, full blocks, tiers 10/20/70, mid_reps=4.
No call logs or synchronizes inside the timed loop.

These H200 figures are historical. They must not be compared as if from the same run as any H100 result.

### Same-run first fair round, 16K/H16, warmup 10, iters 30

| Scope | Time |
|---|---:|
| Native TK tier-A forward kernel | 0.487 ms |
| New TK A\|B\|C forward kernel | 1.074 ms |
| Native TK tier-A backward kernel | 1.552 ms |
| Generic ABC op forward | 1.341 ms |
| Generic ABC op backward | 5.505 ms |
| TK ABC op forward | 1.124 ms |
| TK ABC op backward | 3.392 ms |
| TK / generic ABC op fwd+bwd | 0.660x |

This table is one fair run from the same benchmark program. Native TK's 0.487 ms forward is the
10%-pair tier-A baseline. It is not the same computation as the all-tier 64/4/1 kernel.

### Completed generic round, 16K/H16, bench.py

| Scope | Forward | Backward v1 | Backward v2 |
|---|---:|---:|---:|
| A only | 0.995 ms | 4.332 ms | 4.330 ms |
| A+B+C | 1.343 ms | 5.508 ms | 5.815 ms |
| Native VSA2 generic forward on same A lists | 0.994 ms | not measured | not measured |

The directly quoted final generic result in project context is 1.341 ms forward and 5.505 ms
backward. It is a generic-backend result and is not mixed into the TK table.

### Later TK tuning, different iterations

These are ranges or individual measurements from different builds:

- tuned A-only forward kernel: about 0.509-0.521 ms;
- tuned A|B|C forward kernel: about 0.879-0.892 ms;
- native tier-A forward reference in those runs: about 0.485-0.489 ms;
- tier-C backward with one split: 0.792 ms;
- tier-C backward with four splits: 0.247 ms;
- tier-B forward before flat log-counts: 0.286 ms;
- tier-B forward after flat log-counts: 0.216 ms.

These do not form one complete post-optimization op-scope run. Do not add the component deltas and
describe the sum as a measured op result. A new same-build full op run is still needed.

The raw, extracted benchmark text is under `experiments/vsa2_representative/results/`. Raw model
logs and prompts are intentionally not committed.

## 9. Softmax line, closed

The original question was a controlled pair at 0.471 and 0.579 ms: the copied native loop with the native softmax, and with the hand-written
one. That roughly 0.108 ms is confounded: the CFG 5 of that run divided only the `.x` lanes of `o`, by one row's sum, while CFG 4 ran a complete
`div_row`, so the pair also compared two epilogues. It should not be quoted as a softmax cost.

The vector softmax now in CFG 1 was measured against the hand-written CFG 6 in round 2 and gave no operator improvement above A/A. The patch
under `patches/` and `unverified/` was an earlier attempt, is not applied and was never compiled.

## 10. Problems already resolved

- **Warp-collective deadlock.** TK store/wait helpers were initially called only by thread 0 and
  hung. They are warp-collective; the fix calls them from the full first warp. The first hang was
  terminated by exact PID/argv/cwd validation rather than by touching unrelated processes.
- **Mid-table swizzle phase.** Tier-B records are pre-swizzled for each start phase a slot can take.
  Tests un-swizzle the stored table and compare it bit-exactly with plain bf16 group means.
- **Register spill/pressure, not resolved for the all-tier path.** An earlier note here reported 128
  registers and no spill for the target forward. That does not describe the all-tier kernel: at D=128 the bulk-gather
  CFG 20 and CFG 6 use 168 registers and spill, and the `cp.async` gather in CFG 1 reported 166 and no spills (see the
  table in section 7). Recheck resource output after every kernel edit.
- **Tier-C backward under-occupancy.** One CTA per tail tile gave only 64 CTAs at 16K/H16. Splitting
  each tile's query range across CTAs reduced measured tier-C backward from 0.792 to 0.247 ms.
- **Tier-B dependency/load ordering.** Table indices are read into registers before issuing volatile
  bulk copies. Flattened per-list log-counts removed the large bias-read component seen in the
  earlier indirect lookup.
- **Incorrect higher-tier masks and padded writes.** Tests cover partial blocks and require padded
  output rows to be zero while keeping a finite LSE required by TK backward.

## 11. Counter and measurement limits

The harness collects no hardware counters, and no permission or driver workaround is authorized. It reports no MFU, occupancy, stall-reason
or DRAM-throughput numbers. Do not infer them from CUDA-event timings.

Timing labels are equally important:

- kernel scope excludes table construction where stated;
- op scope includes autograd plumbing, table build, zeroing, delta, unpool, and finalization;
- native TK rows compute tier A only;
- all-tier rows compute the 64/4/1 approximation;
- partial sums from later iterations are not a complete measurement.

## 12. Execution and file ownership

GPU execution goes through the coordinator. `<gpu-runner>` in this document stands for however the coordinator launches a command on one GPU with
the provisioned Python. The harness requires exactly one visible GPU and records no node name, address or raw device id.

```text
experiments/vsa2_representative/                  portable package and 64/4/1 sources
experiments/vsa2_representative/results/          selected measured output extracts
experiments/vsa2_representative/patches/          the original softmax patch, not applied
experiments/vsa2_representative/unverified/       its full source snapshot
experiments/vsa2_representative/scripts/          parameterized remote build wrappers
docs/vsa2_representative_handoff.md               this handoff
```

Excluded: `.git`, build trees, `.so`, `__pycache__`, vendored ThunderKittens, weights and media, raw model logs, and credentials.
