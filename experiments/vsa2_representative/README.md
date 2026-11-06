# VSA2 with representative blocks — generic CUDA and H100 ThunderKittens/TMA

VSA2's fine branch keeps the top-k KV blocks its router selects and **drops the rest**. This adds a
three-tier KV budget so the dropped blocks are attended to at reduced resolution instead:

| tier | share of (query block, kv block) pairs | kept per kv block |
|---|---|---|
| A — full | top `frac_full` (default **10%**) | all **64** tokens |
| B — mid  | next `frac_mid` (default **20%**) | **`mid_reps`** representatives (default 4, one per 16-token group) |
| C — tail | every remaining pair (default **70%**) | **1** representative (the block mean) |

Every query block therefore sees the **whole sequence**, at three resolutions, instead of 10% of it.
The 64 / 4 / 1 layering is the default; the fractions and `mid_reps` are parameters.

## The one idea that makes it correct

A representative is the plain mean `k̄`, `v̄` of the `c` tokens of a contiguous group. Its column enters
the softmax with `+ log2(c)` added to its log2-domain logit, because

```
  Σ_{i∈S} exp(scale·q·k_i) v_i  ≈  c · exp(scale·q·k̄) · v̄
```

so the factor `c` lands in the numerator **and** the denominator of the *one* online softmax that spans
all three tiers. Consequences, all of them tested:

* the result stays a convex combination of the columns' values — no mass is invented or lost;
* it is **exact** when the tokens of a group are identical (`tests/test_reference_cpu.py` asserts each
  tier then reproduces plain dense attention, including for a partial block where `c < 64`);
* `mid_reps` is one of 1, 2, 4, 8, 16; an empty group gets `log2(c) = -inf` and drops out;
* `log2(c)` is a constant of the tier assignment, so it contributes no gradient of its own.

**Gradients.** A representative is a mean, so `∂rep/∂x_i = 1/c` and its gradient is spread as `drep/c`
over exactly the tokens it averaged (`rep_unpool`; padded slots get zero). Tier-A token `dK`/`dV` come
from VSA2's own KV-owner backward, unmodified — a tier-A column's bias is 0, which is precisely that
kernel's `col < blk_cnt` masking.

**What is deliberately unchanged.** Routing is VSA2's own PSP router and its exact global top-k, run at
two budgets over one score matrix (`budget_A` selects tier A and yields its CSR lists, `budget_AB`
selects A∪B). There is **no clustering**, no token reordering, no added connectivity rule, and no reuse
of any earlier call's LSE. Groups are VSA2's own contiguous R-token router groups (`R = 64/mid_reps`),
so at the default `mid_reps = 4` the four representatives are genuine spatio-temporal sub-cubes under
`layout.py`'s block layout. The coarse branch and the `tanh` gate are kept as they are.

Tiers B and C need **no per-pair index array**: they are read from the two selection bitmaps
(`tier B = bits_ab & ~bits_a`, `tier C = ~bits_ab`), so tier C — 70% of all pairs — costs no list
memory at all.

## Files

```
csrc/rep/rep_kernels.h      parameter structs, launchers, and the layout/semantics contract
csrc/rep/rep_common.cuh     gathered tile load, the warp-parallel tier bit scan, the column stream
csrc/rep/rep_attn_fwd.cu    fused three-tier forward (one online softmax over all tiers)
csrc/rep/rep_attn_bwd.cu    three-tier backward, query-owner direction (dQ + representative dK/dV)
csrc/rep/rep_pool.cu        representative construction, its adjoint (unpool), and delta
csrc/rep/rep_api.cpp        torch bindings, the two-budget tier assignment, re-exported VSA2 stages
csrc/rep_tk/rep_tk.cu       H100 ThunderKittens WGMMA + TMA forward/backward for all three tiers
vsa2_rep/ops.py             autograd wiring, RepTiers, TierSelection, the operators
vsa2_rep/reference.py       pure-PyTorch reference (fp64) + bitmap pack/unpack
tests/test_reference_cpu.py representative softmax semantics, no GPU and no build needed
tests/test_gpu.py           kernels vs the reference: routing, tables, forward, backward, unpool
bench.py / prof.py          end-to-end comparison / per-kernel breakdown with measured scopes
bench_stages.py             measurement-only stages of the full operator's forward
bench_bwd_stages.py         measurement-only stages of one backward() call
bench_graph.py              measurement-only eager F+B vs a CUDA graph replay of it
bench_route_graph.py        measurement-only route-only CUDA graph probe
bench_determinism.py        measurement-only eager-vs-eager determinism probe
bench_tables.py             measurement-only table-build vs log-count timing split
one_launch.py               one generic or TK kernel after warmup, for a profiler to select by name
```

The three-tier kernels are this package's own; the pooling, router, global top-k, generic fine
forward/backward and gate fusion are VSA2's, compiled in unchanged from a source tree you point at.

## Build and test

```bash
VSA2_REP_BUILD=1 VSA2_REP_SOURCE=/path/to/EchoSparse-Attn \
TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace     # or: pip install -e . --no-build-isolation

python tests/test_reference_cpu.py     # CPU only, no extension needed
python tests/test_gpu.py               # needs the extension and a GPU
python bench.py --len 32768 --heads 24
python prof.py --profiler
```

`VSA2_REP_SOURCE` must contain `csrc/vsa` (the `W2GenAI-Lab/EchoSparse-Attn` tree; validated at commit
`25e2b37`). D ∈ {64, 128}, bf16 or fp16.

## Usage

```python
from vsa2_rep import RepTiers, rep_attention, rep_fine_attention, route_tiers

# full operator: O = O_fine(three tiers) + tanh(gate) * repeat(O_coarse)
o = rep_attention(q, k, v, gate, blk_cnt, R=16, tiers=RepTiers(frac_full=0.10, frac_mid=0.20, mid_reps=4))

# fine branch alone, on a frozen selection
sel = route_tiers(q, k, blk_cnt, RepTiers())
o_f, lse = rep_fine_attention(q, k, v, blk_cnt, sel, mid_reps=4)
```

`phases=PHASE_A` runs tier A only and is then **bit-identical** to VSA2's own fine kernel on the same
lists (`tests/test_gpu.py` asserts max|err| = 0 for both `o` and `lse`), which makes every comparison
below an apples-to-apples one.

## Measured results

NVIDIA H200 (sm_90), torch 2.13.0+cu130, bf16, B=1, D=128, `frac_full=0.10 frac_mid=0.20 mid_reps=4`.
CUDA events around a loop; nothing logs or synchronizes per call.

**Preparation and routing** (`prof.py` section 0, L=16384 H=16). Routing is *outside* the attention
timings below, so these are the numbers that say what comes from kernels and what from the router:

| | time |
|---|---|
| `pool_fwd` (router pools + block means, shared with VSA2) | 0.080 ms |
| `route_tiers`, **one** budget (`frac_mid=0`) — this is VSA2's own routing | 0.232 ms (incl. `pool_fwd`) |
| `route_tiers`, **two** budgets (`frac_mid=0.20`) — what the tiers need | 0.328 ms (incl. `pool_fwd`) |
| → the second global top-k pass the tiers add | **+0.096 ms** |

With `frac_mid=0` the two budgets coincide, tier B is empty and A∪B *is* A, so the second pass is
skipped entirely — the degenerate configuration costs exactly VSA2's single routing pass.

**Forward, by scope** (`prof.py` section 3b, per call):

| scope | time |
|---|---|
| fine branch only, frozen selection (no routing) | 1.343 ms |
| fine branch + two-budget routing | 1.672 ms — routing is **20%** |
| full operator (routing + coarse branch + gate + three-tier fine) | 1.760 ms |

So of the full forward, the three-tier attention kernel is 1.29 ms, routing 0.33 ms, and the coarse
branch + gate 0.09 ms; the extra routing pass the tiers require is 0.096 ms, i.e. **5.5% of the full
forward** — not something the wrapper-overhead figure below could have told you.

**Per-kernel, isolated** (`prof.py`, L=16384, H=16, tables prebuilt so the scopes match):

| | time | vs native |
|---|---|---|
| VSA2 `attn_fwd` (native, tier-A lists) | 0.997 ms | 1.000× |
| `rep_attn_fwd` phases=A | **0.941 ms** | **0.944×** |
| `rep_attn_fwd` phases=A\|B | 1.133 ms | 1.136× |
| `rep_attn_fwd` phases=A\|B\|C | **1.291 ms** | **1.294×** (tile count predicts 1.273×) |
| `rep_attn_fwd` phases=B\|C (gathered tiers alone) | 0.396 ms | 0.397× |
| `rep_build` (tables, once per forward) | 0.048 ms | 0.048× |
| autograd wrapper overhead | ±0.005 ms | — |

So the generalization is **free when only tier A runs** (0.94×, slightly faster than the kernel it
extends), and full-sequence coverage costs **1.29× the forward** against a tile-count ideal of 1.27×,
while adding 23.4% more columns.

**What a gathered tile pays over a contiguous one** (`prof.py` section 1b — ablations through a
diagnostic flag, since no hardware counters are available). Adding tiers B|C to a tier-A stream costs
0.350 ms over the 7 gathered tiles it adds = 49.9 µs/tile, **1.36× a tier-A tile** (36.8 µs):

| component | cost | share of the B\|C pass |
|---|---|---|
| softmax bias reads from shared memory | −0.001 ms | ~0% |
| scattered row gather + `expand_tile` (vs contiguous rows) | +0.010 ms | 3% |
| the gather loader's per-chunk index read | +0.012 ms | 3% |
| everything else (MMA, softmax, column stream, bitmap scan) | 0.375 ms | 95% |

The last row is **not** decomposed further, and deliberately not guessed at: skipping the tier bitmap
scan changes how many tiles the stream produces, so ablating it would compare two different workloads
rather than isolate a cost. What is established is that the gathered tier's overhead is **not** the
log2-count bias lookup and **not** the scattered gather — both were plausible and both measured near
zero.

**End-to-end** (`bench.py`, the autograd op, includes `rep_build`). The "A only" rows are **this
implementation with tiers B and C switched off** — the same KV budget VSA2 keeps — **not** VSA2's own
kernel. The native forward is listed separately; a complete native *backward* was not measured, so no
row below should be read as one.

| | forward | bwd v1 | bwd v2 | fwd × | bwd × |
|---|---|---|---|---|---|
| L=16384 H=16 — this impl, A only | 0.995 ms | 4.332 ms | 4.330 ms | 1.00× | 1.00× |
| L=16384 H=16 — A+B+C (full sequence) | 1.343 ms | 5.508 ms | 5.815 ms | 1.35× | 1.27× |
| L=16384 H=16 — *native VSA2 `attn_fwd`, same lists* | *0.994 ms* | — | — | — | — |
| L=32768 H=24 — this impl, A only | 5.441 ms | 24.191 ms | 24.165 ms | 1.00× | 1.00× |
| L=32768 H=24 — A+B+C (full sequence) | 7.560 ms | 30.594 ms | 33.100 ms | 1.39× | 1.26× |
| L=32768 H=24 — *native VSA2 `attn_fwd`, same lists* | *5.689 ms* | — | — | — | — |

For the backward the only native reference measured is VSA2's `attn_bwd` on the tier-A lists:
**2.203 ms** without dQ, **3.522 ms** with its atomic dQ (L=16384, H=16). This implementation's A-only
backward is 4.34 ms end to end, i.e. it walks the tier-A pairs twice under `bwd_variant=2`.

**Where the gathered tiers' backward time goes** (`prof.py` section 2b, a diagnostic flag that skips
parts of the work; L=16384, H=16, phases=B|C = 1.633 ms): dQ over those tiles 0.854 ms (52%), the two
representative MMAs 0.311 ms (19%), the atomic scatter of the representative tables 0.468 ms (29%).

### Optimizations that came out of these measurements

| change | effect |
|---|---|
| tier A reads its mask from the block's token count instead of a 64-entry bias array written by thread 0 | forward A-only **1.527 → 0.973 ms** |
| decide the next tile before its data arrives, so a gathered tile's expansion lands on an existing barrier | removed one `__syncthreads` per gathered tile |
| skip the second global top-k pass when `frac_mid = 0`, where A∪B is A by definition | routing 0.328 → 0.232 ms in that configuration |
| make the tier bit scan warp-parallel instead of single-threaded | forward B\|C **0.464 → 0.423 ms**, per-gathered-tile 1.83× → 1.65×; backward B\|C 0.947 → 0.854 ms |
| read a gathered tile's table indices into registers *before* issuing the copies (`cp.async` is `asm volatile`, so a read left between two copies cannot be hoisted and its latency is exposed per chunk) | forward B\|C **0.433 → 0.396 ms**, that component 11% → 3%, per-gathered-tile → **1.54×** |

Two hypotheses were **refuted by measurement and reverted** — recorded because the negative results are
the useful part:

* overlapping the `log2(c)` load with the MMA instead of leaving it between two barriers: no measurable
  change (0.960 → 0.954 ms, noise). The structure was kept, the claim was not.
* giving each thread one row of a gathered tile so it reads the table index once instead of once per
  16-byte chunk: **worse** (forward A|B|C 1.347 → 1.442 ms), because consecutive threads then stop
  hitting consecutive addresses and the warp's accesses no longer coalesce. Reverted.

## Honest limitations

* **No hardware counters.** `ncu` is installed on the test host but returns `ERR_NVGPUCTRPERM`, and
  enabling GPU performance counters needs an nvidia module reload that would disrupt a shared machine.
  So there is **no measured MFU, occupancy, stall-reason or DRAM-throughput data here**, and none is
  inferred. The attribution above comes from CUDA-event timing of controlled ablations instead.
* **A gathered tile still costs ~1.4× a contiguous one** in the forward (1.36× incremental), after the
  two components that could be ablated were found and fixed. The remaining 95% sits inside the MMA,
  softmax, column stream and bitmap scan together and was **not** separated, because the one ablation
  that would isolate the scan also changes the tile count. Narrowing it further needs hardware counters.
* **14% of gathered column slots are wasted.** Tiers B and C each round up to a whole 64-column tile
  (4 + 3 tiles per query block for 384 real columns). Packing both tiers into one densely filled stream
  — by allocating the mid and tail tables contiguously so one row index spans both — would cut gathered
  tiles 7 → 6. Not implemented.
* **Representative `dK`/`dV` are summed in a nondeterministic fp32 order** (atomics into the small
  representative tables), because one representative row is shared by many query blocks. This is the
  same tradeoff VSA2's own TK backward makes for `dK`/`dV`. `dQ` is exact and deterministic with
  `bwd_variant=2`; `bwd_variant=1` (default, 0.94× the backward) takes tier-A `dQ` from VSA2's fp32
  atomics instead. The two agree to 2.4e-4 in the tests.
* **The measured table above is the generic CUDA backend, not a same-version comparison with the TK/TMA
  backend.** The package also contains `csrc/rep_tk/rep_tk.cu` for sm_90 bf16. Its tier A and contiguous
  tail table use tensor TMA; tier B uses a pre-swizzled table and 1-D `cp.async.bulk`. Native tier-A TK
  is the 10%-pair baseline, while the new kernel also computes tiers B/C; those are different workloads.
  See [`docs/vsa2_representative_handoff.md`](../../docs/vsa2_representative_handoff.md) for the current
  TK version boundary, measurements, and unverified softmax follow-up.
* The backward is ~4× the forward, and `tier_a_bwd_kv` + the query-owner pass walk the tier-A pairs
  twice under `bwd_variant=2`. Measured: folding tier-A `dQ` into the KV-owner kernel's atomics
  (variant 1) recovers only 6%, because those atomics themselves cost 1.32 ms (2.203 → 3.522 ms) — the
  prediction that this would save ~30% was **wrong**, and the measurement is what is reported.
* Only MHA self-attention with `q/k/v` of identical shape, and `blk_cnt` shared across (b, h), exactly
  as in VSA2.
