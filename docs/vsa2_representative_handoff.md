# VSA2 64/4/1 representative-block handoff

Date: 2026-11-06

Branch:
<https://github.com/shaoshitong/self_trainable_or_free_attn/tree/vsa2-representative-tk-tma-handoff-20261106>

This document hands off the completed VSA2 representative-block work and its H100
ThunderKittens/TMA implementation. It is a release and continuation boundary, not a claim that the
last unverified softmax experiment is complete.

## 1. Status and version boundary

The active CUDA source is:

```text
experiments/vsa2_representative/csrc/rep_tk/rep_tk.cu
SHA-256 6224637ff9a96404e170aa7de568a070c449ae83707564c37cbfae3cffdc784e
```

This is the remote snapshot read from:

```text
aster-debug-1105-worker-0:/efs/astershao/vsa2_rep_work/pkg/csrc/rep_tk/rep_tk.cu
```

The project instruction selects this remote snapshot as the last built/tested baseline. The test log
contains successful complete CPU, generic GPU, and TK GPU suites in the tested sequence. Later
diagnostic native-copy builds and timing runs are also present; the log has no later full-suite pass
line after those diagnostic-only additions. If an exact byte-to-run attestation is required, rerun
the suites before changing the source.

The separate local file was modified after an automatic model downgrade and was never synchronized,
rebuilt, or tested. It is preserved, but is not active:

```text
experiments/vsa2_representative/unverified/rep_tk.softmax-vector-ops-UNVERIFIED.cu
SHA-256 b7f48a21c910ac5e41b1d6c860abde1a8808c00171c5c8b4fb07f835794b4a65
```

Its diff against the active source is:

```text
experiments/vsa2_representative/patches/softmax-vector-ops-UNVERIFIED.patch
```

The patch is not applied. It has not been compiled or tested. Do not replace the active file with it
until the checks in section 9 pass.

No algorithm implementation or GPU test was added during release packaging. This branch is an
organized snapshot plus handoff.

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
  matching record. The selected record's RM rows are copied into a wgmma-ready shared tile with 1-D
  `cp.async.bulk`; the tile's mbarrier still controls completion.
- Tier C forward loads the contiguous block-mean table with tensor TMA and masks it with the A/B
  bitmap.
- All tiers feed one online softmax and `mma_AB` output accumulation.
- Tier A backward reuses VSA2's native TK backward on the tier-A k2q lists.
- Tier B backward is query-owner. dQ remains in registers. Representative dK/dV are staged and
  reduce-added into the fp32 mid-gradient table with
  `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32`.
- Tier C backward is KV-owner over block means. dQ uses tensor-TMA store-add into the shared fp32 dQ
  accumulator.
- The C backward query range is split across CTAs. The automatic split targets one resident wave
  rather than launching only one CTA per KV tile.

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

ThunderKittens is not vendored here. The actual local headers used were:

```text
/Users/astershao/Documents/Codex/2026-11-05/c-kan/work/vsa2/vendor/ThunderKittens
HEAD 6c27e28c8115d1839d9eeeb530913c184a75fc87
```

That SHA is the pin used by the VSA2 source tree and the experiment setup. A clean get is the
ThunderKittens repository checked out at that exact commit.

Validated remote environment:

| Item | Value |
|---|---|
| Host | `aster-debug-1105-worker-0` |
| GPU | NVIDIA H200, sm_90, use GPU 0 only |
| Python | `/opt/venv/bin/python`, Python 3.12 paths |
| PyTorch | `2.13.0+cu130` |
| CUDA runtime | 13.0 |
| nvcc | 13.0.48 |
| CUDA home | `/usr/local/cuda` |
| Work root | `/efs/astershao/vsa2_rep_work` |
| Package | `/efs/astershao/vsa2_rep_work/pkg` |
| VSA2 source | `/efs/astershao/vsa2_rep_work/vsa2_src` |
| ThunderKittens | `/efs/astershao/vsa2_rep_work/ThunderKittens` |

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
/opt/venv/bin/python setup.py build_ext --inplace
```

TK/TMA backend, sm_90a:

```bash
VSA2_REP_BUILD=1 \
VSA2_REP_TK=1 \
VSA2_REP_TK_ROOT=/path/to/ThunderKittens \
VSA2_REP_SOURCE="$VSA2_REP_SOURCE" \
TORCH_CUDA_ARCH_LIST=9.0a \
MAX_JOBS=24 \
/opt/venv/bin/python setup.py build_ext --inplace
```

The variables enforced by `setup.py` and `ops.py` are:

- `VSA2_REP_BUILD=1`: build `vsa2_rep._C`.
- `VSA2_REP_SOURCE`: source root containing `csrc/vsa/vsa2_kernels.h`.
- `VSA2_REP_TK=1`: compile the TK/TMA source and add `-DVSA2_REP_TK`.
- `VSA2_REP_TK_ROOT`: root containing `include/kittens.cuh`.
- `TORCH_CUDA_ARCH_LIST`: the TK build rejects an arch without `9.0a`.
- `CUDA_HOME`: defaults to `/usr/local/cuda`; the linker needs the toolkit driver stub.
- `VSA2_REP_PTXAS_VERBOSE=1`: optional register/spill diagnostics.

The scripts under `scripts/` are optional parameterized wrappers. They have no host-specific defaults:

```bash
export VSA2_REP_HOST=aster-debug-1105-worker-0
export VSA2_REP_REMOTE=/efs/astershao/vsa2_rep_work
export VSA2_REP_SOURCE=/efs/astershao/vsa2_rep_work/vsa2_src
export VSA2_REP_TK_ROOT=/efs/astershao/vsa2_rep_work/ThunderKittens
./scripts/sync_build.sh all
./scripts/wait_run.sh "CUDA_VISIBLE_DEVICES=0 /opt/venv/bin/python tests/test_gpu_tk.py"
```

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
CUDA_VISIBLE_DEVICES= \
  /opt/venv/bin/python tests/test_reference_cpu.py
```

Generic GPU suite:

```bash
CUDA_VISIBLE_DEVICES=0 \
  /opt/venv/bin/python tests/test_gpu.py
```

TK GPU suite:

```bash
CUDA_VISIBLE_DEVICES=0 \
  /opt/venv/bin/python tests/test_gpu_tk.py
```

The TK suite checks tables and swizzle, per-tier and all-tier forward/LSE against fp64, agreement with
native TK on tier A, TK-vs-generic backward dQ/dK/dV, and the routed autograd operator. Recorded
successful coverage includes D=64/128, `mid_reps=2/4/8`, and an incomplete final block.

First fair kernel/op benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 \
  /opt/venv/bin/python bench_tk.py \
  --len 16384 --heads 16 --dim 128 --mid-reps 4 \
  --warmup 10 --iters 30
```

Generic op benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 \
  /opt/venv/bin/python bench.py \
  --len 16384 --heads 16 --dim 128 --batch 1 --iters 20
```

`bench_tk.py` uses warmup=10/iters=30 by default. `bench.py` calls its timing helper with the helper's
warmup=5 and the selected iteration count. Both allocate deterministic BF16 inputs under
`torch.manual_seed(0)`, use a full-block `blk_cnt`, route once, and time a loop with CUDA events and
one final synchronize.

Targeted timing helpers:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/venv/bin/python fwd_cfg.py
CUDA_VISIBLE_DEVICES=0 /opt/venv/bin/python bwdc_split.py
```

Optional diagnostic controls:

```text
REP_TK_FWD_CFG=1                 tuned forward path
REP_TK_FWD_CFG=4                 diagnostic native-loop/native-softmax copy
REP_TK_FWD_CFG=5                 diagnostic native-loop/custom-softmax copy
REP_TK_BWDC_SPLITS=<positive>    force tier-C query split count
```

The diagnostic modes are measurement code, not production configurations.

## 8. Measured evidence

All principal measurements are NVIDIA H200, B=1, D=128, BF16, full blocks, tiers 10/20/70, mid_reps=4.
No call logs or synchronizes inside the timed loop.

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

## 9. Unfinished softmax optimization

The remaining performance question was isolated with a controlled same-build test:

| Diagnostic path | Time |
|---|---:|
| Copied native TK loop + native TK softmax | 0.471 ms |
| Same copied loop + experiment hand-written softmax | 0.579 ms |
| Independent native TK kernel | about 0.486 ms |

The controlled pair attributes about 0.108 ms to the hand-written softmax under that test. It does not
identify TMA setup or the harness as the cause.

The intended follow-up was to replace the hand-written row max/sum/rescale loop with ThunderKittens
vector operations. The local patch attempts that change, plus LSE staging through a TK column vector
and TMA store. It is unverified and may contain compile or numerical errors.

The automatic downgrade edit was the last action involving that file. There was no subsequent rsync,
build, or test command. Do not apply, "fix while publishing", or benchmark it as accepted work.

Suggested controlled continuation:

1. Work on a separate branch and keep the active tested file unchanged.
2. Inspect the patch; do not assume its vector-op types or LSE descriptor are valid.
3. Apply only in a disposable checkout.
4. Build with the exact TK command in section 5 and capture nvcc/ptxas output.
5. Run CPU, generic GPU, and full TK GPU suites.
6. Repeat the `fwd_cfg.py` controlled pair in one build, with the same inputs and warmup/iteration
   settings.
7. Run complete `bench_tk.py` op scope before claiming an op-level improvement.

To inspect the patch without applying it:

```bash
cd experiments/vsa2_representative
git apply --check patches/softmax-vector-ops-UNVERIFIED.patch
```

Successful `--check` would prove textual applicability only, not correctness.

## 10. Problems already resolved

- **Warp-collective deadlock.** TK store/wait helpers were initially called only by thread 0 and
  hung. They are warp-collective; the fix calls them from the full first warp. The first hang was
  terminated by exact PID/argv/cwd validation rather than by touching unrelated processes.
- **Mid-table swizzle phase.** Tier-B records are pre-swizzled for each start phase a slot can take.
  Tests un-swizzle the stored table and compare it bit-exactly with plain bf16 group means.
- **Register spill/pressure.** Forward launch bounds and path specialization were tuned. The target
  forward specialization reports 128 registers with no stack spill; the diagnostic generic
  min-blocks choice spilled. Recheck resource output after every future kernel edit.
- **Tier-C backward under-occupancy.** One CTA per tail tile gave only 64 CTAs at 16K/H16. Splitting
  each tile's query range across CTAs reduced measured tier-C backward from 0.792 to 0.247 ms.
- **Tier-B dependency/load ordering.** Table indices are read into registers before issuing volatile
  bulk copies. Flattened per-list log-counts removed the large bias-read component seen in the
  earlier indirect lookup.
- **Incorrect higher-tier masks and padded writes.** Tests cover partial blocks and require padded
  output rows to be zero while keeping a finite LSE required by TK backward.

## 11. Counter and measurement limits

`ncu` returns:

```text
ERR_NVGPUCTRPERM
```

No permission or driver workaround was attempted. There are therefore no measured MFU, occupancy,
stall-reason, or DRAM-throughput numbers in this work. Do not infer them from CUDA-event timings.

Timing labels are equally important:

- kernel scope excludes table construction where stated;
- op scope includes autograd plumbing, table build, zeroing, delta, unpool, and finalization;
- native TK rows compute tier A only;
- all-tier rows compute the 64/4/1 approximation;
- partial sums from later iterations are not a complete measurement.

## 12. Machine and file ownership

Use only:

```text
SSH host: aster-debug-1105-worker-0
Remote package: /efs/astershao/vsa2_rep_work/pkg
GPU: CUDA_VISIBLE_DEVICES=0
```

Do not use `aster-sparse-attn-fwd-1105-06713-worker-0`; its four GPUs belong to another chat.
Do not alter the nvidia driver or counter permissions on the shared H200 host.

The original WIP checkout under the local source workspace was left untouched. It contains an
unrequested WIP commit with placeholder identity; this branch does not copy that `.git` directory,
history, or identity.

Release inventory:

```text
experiments/vsa2_representative/                  portable package and 64/4/1 sources
experiments/vsa2_representative/results/          selected measured output extracts
experiments/vsa2_representative/patches/          unverified softmax patch, not applied
experiments/vsa2_representative/unverified/       full unverified source snapshot
experiments/vsa2_representative/scripts/          parameterized remote build wrappers
docs/vsa2_representative_handoff.md               this handoff
```

Excluded: `.git`, build trees, `.so`, `__pycache__`, vendored ThunderKittens, weights/media, raw model
logs, system prompts, credentials, and SSH material.
