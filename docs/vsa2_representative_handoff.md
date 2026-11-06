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

The first packaged snapshot on this branch was already published before the handoff was resumed.
Because publishing this release was explicitly authorized and the snapshot is public, the resumed
session preserves that commit rather than rewriting it or force-pushing the branch. Any follow-up
release commit is authored with the verified GitHub account identity; no upstream commit metadata or
placeholder WIP identity is reused. The top-level README now scopes its older Triton "ordinary pointer
loads" statement separately from this package's implemented ThunderKittens/TMA data path.

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

`test_masked_and_empty` (generic and TK suites) adds zero-count blocks, all-masked first and intermediate
tiles, empty selections and per-tier gradients. It has not been run yet, so the recorded passes above do
not include it.

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

Configurations. The config table in `bench_compare.py` describes the kernel that has the CFG 20 history arm.
Registers and stack bytes are D=128 `ptxas` figures: CFG 1 from CFG 10's round 4 build, the other measured rows
from source `3c291f2`, before the gather changed, with CFG 20's taken from CFG 1 at that source. None has been read
from a build containing CFG 20.

| CFG | Kernel | Softmax | MINB | Tier-B gather | Registers | Stack bytes |
|---:|---|---|---:|---|---:|---:|
| 1 (default) | `fwd_ker`, all tiers; heuristic tier-C backward split | TK vector | 3 | `cp.async` | 166 | no spills |
| 10, 13 | aliases of CFG 1, kept so old logs stay reproducible | as CFG 1 | 3 | as CFG 1 | as CFG 1 | as CFG 1 |
| 21 | CFG 1 forward; tier-C backward split by the old wave policy | as CFG 1 | 3 | as CFG 1 | as CFG 1 | as CFG 1 |
| 22 | CFG 1 forward; tier-C backward split forced to 4 | as CFG 1 | 3 | as CFG 1 | as CFG 1 | as CFG 1 |
| 23 | CFG 1 forward; tier-A backward `__launch_bounds__(128, 3)` probe | as CFG 1 | 3 | as CFG 1 | as CFG 1 | as CFG 1 |
| 11 | negative arm: `fwd_kdb_ker`, CFG 1 plus a double-buffered K tile | TK vector | 3 | `cp.async` | not recorded here | not recorded here |
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
the launched template. CFG 21/22/23 are not in the newest kernel source read here, `f634020`, so they are refused
until they land.

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
`cfg == <value>` (a backward arm may use a name such as `bwd_cfg`; the aliases may have no line), an alias or
backward arm whose forward dispatch differs from the default, and a CFG 20 that launches the default. The check
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
CFG 21, CFG 22 and a forced split reach `kernel.tk.bwd.C`, and CFG 23 reaches `kernel.tk.bwd.A`. Between CFG 1, its
aliases and the backward arms, one forward kernel by definition, the forward-only rows are controls.

The forward-identical check for those arms has three parts. The forward dispatch in `fwd_impl` must launch the
default. `o` and `lse` of all six forward phase rows must be byte-equal between arms, or the result is invalid. A
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

Next matrix, D=128/RM4/H16, on one build containing CFG 21, 22 and 23, with the suites passing on it. Each arm needs
its own A/A floor at its shape:

| # | Variable | Run | Shape |
|---:|---|---|---|
| 1 | none | CFG 1 vs CFG 1 | each length below |
| 2 | tier-A backward occupancy | CFG 1 vs CFG 23 | 16K |
| 3 | tier-C split forced to 4 | CFG 1 vs CFG 22 | 32K; 16K is an A/A run at 132 SMs |
| 4 | tier-C wave policy | CFG 1 vs CFG 21 | 48K; 16K and 32K are A/A runs at 132 SMs |

```bash
<gpu-runner> python bench_compare.py --candidate REP_TK_FWD_CFG=1 --out results/raw/aa-16k.json
<gpu-runner> python bench_compare.py --candidate REP_TK_FWD_CFG=23 --out results/raw/cfg1-cfg23-16k.json
<gpu-runner> python bench_compare.py --len 32768 --candidate REP_TK_FWD_CFG=22 --out results/raw/cfg1-cfg22-32k.json
<gpu-runner> python bench_compare.py --len 49152 --candidate REP_TK_FWD_CFG=21 --out results/raw/cfg1-cfg21-48k.json
```

Read `kernel.tk.bwd.C` for CFG 21/22, `kernel.tk.bwd.A` for CFG 23, and `fine_op` and `full_op` `b_direct` and F+B for
all three. Check `multi_processor_count`: the split table assumes 132 SMs. The result that matters is measured
fine-op and full-op F+B against the A/A displacement. Adding isolated stage times does not produce a measured total.

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

Every row keeps its meaning under every all-tier configuration. `reads_fwd_cfg` marks the rows that can change with
`REP_TK_FWD_CFG` (`total`, `fine`, `fine.kernel`); the others should match another configuration within the floor.
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

## 9. Unfinished softmax optimization

The remaining performance question was isolated with a controlled same-build test:

| Diagnostic path | Time |
|---|---:|
| Copied native TK loop + native TK softmax | 0.471 ms |
| Same copied loop + experiment hand-written softmax | 0.579 ms |
| Independent native TK kernel | about 0.486 ms |

The controlled pair attributes about 0.108 ms to the hand-written softmax under that test. It does not
identify TMA setup or the harness as the cause.

This figure is confounded and must be rerun. The CFG 5 of that run divided only the `.x` lanes of `o`, by
one row's sum, while CFG 4 ran a complete `div_row`, so the pair also compared two epilogues. The current
CFG 5 normalizes both rows and all elements, and CFG 7 adds the TK-vector softmax to the same copied loop.

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
6. Repeat the controlled pair in one build with `bench_compare.py --profile controlled-softmax`, which
   keeps the same inputs and warmup/iteration settings. That result is timing only.
7. Run the representative profile of `bench_compare.py`, which includes complete-call scope, before
   claiming an op-level improvement.

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
- **Register spill/pressure, not resolved for the all-tier path.** An earlier note here reported 128
  registers and no spill for the target forward. That does not describe the all-tier kernel: at D=128, CFG 1
  and CFG 6 use 168 registers and spill (see the table in section 7). Recheck resource output after every
  kernel edit.
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
