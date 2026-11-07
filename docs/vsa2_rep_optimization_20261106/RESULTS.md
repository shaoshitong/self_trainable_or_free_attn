# VSA2 representative TK/TMA on H100 — measured results, R1–R22

Every number below comes from a same-build A/B on an H100 80GB HBM3 (CUDA 13.2 / nvcc 13.2.78 / GCC 12.5,
ThunderKittens pinned `6c27e28c`), with the full CPU/generic/TK suites passing on that build before any
timing claim. Shape: B1 / H16 / D128 / RM4 / bf16, ABBA, warmup 10, 30 iters, 3 batches, 6 op samples,
180 direct-backward samples. A/A arms are reported for every round.

## 1. Headline — four accepted changes

Everything below is a same-build A/B with an A/A arm; the absolute times drift 2% between runs, so only the
ratios are claims. In adoption order:

| # | change | measured on | ratio |
|---|---|---|---|
| 1 | tier-B gather: 32 x 512 B `cp.async.bulk` -> CTA-wide 16 B `cp.async` | `fine_op f`, `A\|B\|C` | 0.890, 0.916 |
| 2 | tier-A dK/dV stored by the single owner CTA; `kg`/`vg` memsets removed | `full_op b_direct`, `f+b` | 0.9731, 0.9790 |
| 3 | coarse CSR metadata cached by shape (it is shape-only) | `full_op f`, `f+b` | 0.9739, 0.9904 |
| 4 | representative tables built from the router pools instead of re-reading K/V | `full_op f`, `f+b` | 0.9633, 0.9905 |

Current state (job 806532, two A/A pairs in one run): **full-operator forward ~1.20 ms**, **full F+B ~3.95 ms**,
against 1.3704 and 4.1789 on the pre-round-4 build — about **-12% forward** and **-5.5% F+B**.

Two of the four are not kernel arithmetic at all: one deletes a memset, one deletes a 128 MiB duplicate read.
That is where the remaining headroom in this operator lives; the kernel-side instruction and occupancy
experiments are all in §2 for the opposite reason.

The newest two (#3, #4) also carry their own promotion check — the old path was measured again after the
default flipped, so the win is not an artefact of comparing against a stale build.

## 2. Negative results (all same-build, all with an A/A floor)

| round | change | result | verdict |
|---|---|---|---|
| R2 | vector-softmax rewrite (CFG1 vs CFG6) | F+B 1.0018, AA 0.9967 | no gain |
| R3 | register-cap control MINB=2 (CFG8/9) | forward +22.8%, F+B +5.0%, AA +0.9% | clearly worse |
| R5B | K-only double buffer as a separate kernel (CFG11) | F+B 1.0003, full f 1.011, AA 1.004 | no gain |
| R6 | tier-A backward `__launch_bounds__(128,3)` (CFG23) | F+B **1.350**, `bwd.A` kernel **1.874**, AA 1.005 | much worse |
| R6 | tier-C split: wave policy (CFG21) / forced 4 (CFG22) | F+B 1.0002 / 1.0013 | neutral at 16K |
| R7 | two-stage softmax on the K double buffer (CFG24) | kernel 1.292, F+B 1.063 | kept 3 CTAs and still lost |
| R8 | 2 CTAs with 2- or 3-deep K/V rings (CFG32/33) | 1.076 / 1.110 on the kernel | depth bought with occupancy loses |
| R9 | fused scale+bias FFMA (CFG34) / skipped O rescale (CFG35) | 0.995 / 0.996 | inside the A/A floor |
| R13 | whole-operator CUDA graph | 3.940 replay vs 4.045 eager, but 4.118 with fresh inputs | input copies eat it |
| R14 | table build on a second stream, overlapped with routing (CFG38) | full f 1.018 | the two streams contend for the same K/V reads |
| R17 | route-only graph inside the operator (CFG41) | 1.009 then 0.998 after dropping the clones | launch saving does not survive integration |

Why the two "should have worked" ones failed:

- **MINB=2** raised the register cap to 255 and removed spills, but dropped CTAs per SM. The kernel is not
  spill-limited, so the occupancy loss was pure cost.
- **K double buffer** bought nothing because three CTAs per SM already hide the K TMA latency; the extra
  buffer only added state.
- **CFG23** could not buy occupancy at all: `bwd_attend_ker` requests 113,000 B of dynamic shared memory
  (about 107 KB), so only two CTAs fit per SM regardless of registers. The probe therefore measured pure
  spilling and lost 1.87x on the kernel. Any future occupancy work on that kernel must cut shared memory,
  not registers.
- **CFG21/22** are neutral at 16K because the heuristic, the old wave policy and a forced 4 all choose 4
  splits there; they only diverge at 8K and 32K, where the affected kernel (`bwd.C`, 0.31 ms) is too small
  to move the operator total.
- **CFG24** kept three CTAs per SM (168 registers, 66,560 B of shared memory, so neither limit binds) and was
  still 29% slower on the forward kernel. So the loss is not occupancy: issuing the next tile's QK before the
  current softmax adds a second live `att` tile and 16/10 B of spills in the inner loop, and the wait for that
  QK ends up in front of the current PV. Per-CTA software pipelining does not pay when three CTAs already
  interleave on the SM.

Counting: eleven measured directions in the table above, plus MINB=4 from the early rounds and the router
single-histogram idea (rejected on analysis — its cost is eighteen kernel launches, not redundant arithmetic):
**twelve measured refutations and one analytical one**. Each has a same-build A/A arm and a suite gate on the
build it was measured on.


## 3. Where the time actually goes now (16K)

Kernel-scope forward: tier A ~0.545, tier B 0.165, tier C 0.142, `A|B|C` 0.779 ms. Per useful column:
A 0.327 us, B 0.809, C 0.793 — B and C are 23% of A's columns but 36% of its time, and that residual is
structural (masked tiles, 4-wide and 1-wide contributions), not another gather problem.

Full-operator forward 1.274 ms: `fine` 0.84 (tables 0.07 + kernel 0.78), `route` 0.26, `pool` 0.10,
`coarse` 0.11, `fuse` 0.05, with the residual explained by overlap. The second top-k budget inside
`route` is ~0.10 ms and is the largest un-attacked block outside the attention kernels.

Backward is now **73% of F+B** (2.98 of 4.08 ms at 16K, 9.65 of 13.0 at 32K): tier A 1.55 (native TK,
245 regs, 107 KB smem), tier B 0.50, tier C 0.31, and ~0.6 ms of table/unpool/zero-init/delta work.

## 3b. Why producer/consumer warp specialization is not the obvious next step here

The standard H100 recipe (FA3, TK's `mha_h100`) is one CTA per SM, one producer warpgroup issuing TMA and
two consumer warpgroups in ping-pong. The reason two consumer warpgroups pay is that **both consume the same
K/V stream**: one load feeds two query tiles, so the TMA traffic per unit of MMA work halves and the two
warpgroups cover each other's softmax.

That precondition does not hold in this operator. Every query block has its **own** tier-A list, its own
tier-B list and its own adaptive budgets (`a_ptr/a_idx`, `b_ptr/b_idx`, `bits_ab` are all per query row).
Two query tiles therefore cannot share a K/V tile, and a ping-pong pair would simply load two independent
streams — which is exactly what two of our three CTAs already do, only without paying for a shared CTA and
a producer warpgroup.

So the three-CTA model is not an accident or a missed optimization: it is the natural consequence of
selection-per-query-block. Warp specialization can still buy deeper staging and fewer stalls within one
tile stream, but the FA3-style 2x amortization is unavailable, and the measured record above says the
smaller wins in that direction (K double buffer, two-stage softmax) already lost to the extra state they
cost. That is the concrete reason the remaining upside is small without changing the algorithm rather than
the kernel.

## 4. What is still on the table

| candidate | expected | risk | why it is not done |
|---|---|---|---|
| producer/consumer warp specialization, 2-3 stages, 1 CTA/SM (CFG17) | 20-35% forward | high (rewrite) | the only remaining structural lever; the current design hides latency with 3 CTAs instead |
| fuse `rep_unpool` into the bwd epilogue (CFG18) | removes one kernel + a round trip | medium | queued, not measured |
| single top-k with rank split (router) | ~0.10 ms of full fwd (8%) | medium (VSA2 shared code) | needs bitmap-equality proof against the two-pass version |
| tier-A backward smem restructure to <77 KB | 3 CTAs, unknown | high | touches VSA2's kernel; registers alone are proven not to work |

## 5. Reproduce

Source: `experiments/vsa2_representative` in the `vsa2-representative-opt-20261106` worktree at commit
`d23d31fb84a3d3b8af8e9e4b977c497ff4c8dcdc`; ThunderKittens pinned `6c27e28c`; build with
`VSA2_REP_TK=1`, `TORCH_CUDA_ARCH_LIST=9.0a`; H100 only (sm90a).

`REP_TK_FWD_CFG`:

| value | kernel | status |
|---|---|---|
| unset / 1 | `fwd_ker<D,3,false,false,true>` — tier-B gather by CTA-wide `cp.async` | **default, accepted** |
| 10 | alias of 1 | kept for old logs |
| 20 | pre-R4 kernel, per-slot `cp.async.bulk` gather | history arm |
| 11 | K double buffer, separate kernel | measured negative |
| 24 | K double buffer + two-stage softmax | measured negative |
| 6 / 8 / 9 / 0 / 2 / 3 / 4 / 5 / 7 | softmax and register-cap diagnostics | unchanged from earlier rounds |
| 13 / 21 / 22 / 23 | backward-only arms (split heuristic / wave / forced 4 / tier-A launch bounds) | 21/22 neutral, 23 negative |

Jobs: `h100-r4`, `h100-r5`, `h100-r6`, `h100-r7` under the control directory, each with a source manifest
(per-file SHA-256, 42 files), a fresh-build gate, and the CPU/generic/TK suites passing before any timing.
Observations are `latest.json` in each `private/` directory; the A-only R5 job also carries the 32K arm.

The one-line summary for a reader who will not open the logs: the only change that paid was replacing 32
x 512 B `cp.async.bulk` gather issues with warpgroup-wide 16 B `cp.async`; everything tried after that
(occupancy, buffering, software pipelining, split policy) lost or tied, for reasons documented above.

## 6. Rounds 8-9 (after the accepted change): depth and softmax-instruction negatives

**R8 — depth × occupancy sweep (job 806124, source `34e66c9`).** Two new kernels with 2-CTA launch bounds and
2- or 3-deep K/V rings, both passing the full TK suite:

| arm | `A\|B\|C` kernel | fine fwd | full fwd | full F+B |
|---|---:|---:|---:|---:|
| CFG32: 2 CTAs, 2-deep rings | 1.076 | 1.109 | 1.064 | 1.022 |
| CFG33: 2 CTAs, 3-deep rings | 1.110 | 1.087 | 1.089 | 1.025 |

A/A floor 0.981-1.042. The complete grid is now: 3 CTAs/1 stage **0.78**, 3 CTAs/2 K stages 0.78, 2 CTAs/2
stages 0.86, 2 CTAs/3 stages 0.89. The current point is the optimum; "the pipeline never had enough stages"
is refuted, and so is the idea that buying depth with occupancy can pay.

**R9 — softmax instruction reductions (job 806177, source `4d957ae`).** CFG34 fused the `log2e*scale`
multiply with the tier-B/C bias add into one FFMA; CFG35 skipped the per-tile O rescale when no row's running
maximum moved (warp-uniform `__any_sync`). Both on the default kernel, both passing the TK suite:

| arm | `A\|B\|C` kernel | fine fwd | full fwd | full F+B |
|---|---:|---:|---:|---:|
| CFG34 fused scale+bias | 0.995 | 0.991 | 1.002 | 1.000 |
| CFG35 skipped rescale | 0.996 | 0.975 | 0.998 | 0.999 |

A/A floor on the same build was 0.983-1.011 on those rows, so both are inside the noise band. At three CTAs
per SM the epilogue's instruction stream is absorbed by inter-CTA overlap — the third consecutive result
pointing the same way.

Consequence for the plan: the forward side is at a local optimum that survives every zero-register
instruction change we have thrown at it. The remaining measured pools are (a) ~0.62 ms of backward
bookkeeping outside the three backward kernels, (b) ~0.26 ms of router with ~0.10 ms in the second top-k,
and (c) whatever selection-aware scheduling can win on L2, which we cannot measure directly without counters.

## 7. R11 — accepted: tier-A dK/dV store instead of memset + reduce-add (CFG36)

The routed backward memset `qg`, `kg` and `vg` (384 MiB) because the tier-A kernel reduce-adds and leaves
empty kv blocks unwritten. On the row_ptr path each kv block has exactly one owner CTA, so the kernel can
**store** its rows instead, and the two big memsets disappear. Default keeps the old path; CFG36 is the arm.

Job 806242, source `adebb1f`, all suites pass (TK under CFG36 included, 349 checks):

| row (ms) | CFG1 | CFG36 | ratio |
|---|---:|---:|---:|
| `fine_op ... b_direct` | 3.0290 | 2.9294 | **0.9671** |
| `full_op ... b_direct` | 2.9389 | 2.8597 | **0.9731** |
| `fine_op ... f_plus_b` | 3.7521 | 3.6857 | **0.9823** |
| `full_op ... f_plus_b` | 4.1363 | 4.0495 | **0.9790** |

A/A floor on the same build: `b_direct` 1.0017 / 1.0010, `f+b` 1.0004 / 0.9984, so this is outside the noise.
It is the first accepted change since the tier-B gather.

## 8. The backward, now itemised (CFG1, full operator, total 2.957 ms)

| step | ms |
|---|---:|
| host/autograd gap before the first kernel | **0.118** |
| `fuse_bwd` | 0.099 |
| coarse `attn_bwd` + tail zeros | 0.039 |
| **three token-table memsets (qg, kg, vg)** | **0.135** |
| tier-A `tk_native_bwd` | 1.531 |
| tier-B `rep_tk_bwd_b` | 0.494 |
| tier-C `rep_tk_bwd_c` | 0.311 |
| `finalize` (dQ fp32→bf16) | 0.073 |
| `rep_unpool` (dK/dV fp32→bf16) | 0.172 |
| small launch gaps (13 x ~0.003) | ~0.05 |

So the 0.62 ms previously called "bookkeeping" is mostly necessary bandwidth: `fuse_bwd`, `finalize` and
`rep_unpool` are the fp32→bf16 conversions and the gate gradient, which have to happen somewhere. The two
removable items are the kg/vg memsets (now taken by CFG36, ~0.09 ms) and the **0.118 ms host-side gap before
the first kernel**, which is launch/autograd latency and is a CUDA-graph candidate.

## 9. R12 — CFG36 promoted to default, and the launch-cost question answered

**Promotion verified (job 806290, source `14387d3`).** Default is now the single-writer store path; CFG37 is the
old memset + reduce-add path. Same build:

| row (ms) | default (store) | CFG37 (old) | ratio |
|---|---:|---:|---:|
| `fine_op ... b_direct` | 2.8994 | 3.0063 | 1.0369 |
| `full_op ... b_direct` | 2.8344 | 2.9394 | 1.0370 |
| `full_op ... f_plus_b` | 4.0379 | 4.1403 | 1.0254 |

A/A floor on that build was 0.9999-1.0012 on those rows, so the promotion reproduces the round-11 win and the
old path is now a documented 3.7% regression.

**CUDA-graph probe (`bench_graph.py`, measurement only).** Capture one full `rep_attention(...)` + `.backward()`
with static tensors and replay it:

| row | ms |
|---|---:|
| eager (production path) | 4.0435 |
| graph replay | 3.9256 |
| host cost of one `replay()` call | 0.0028 |

Recoverable: **0.118 ms, 2.9% of F+B**, above the run's 0.030 ms floor. Equivalence: the forward output is
bit-identical (`max_abs_diff` 0.0) and dq/dk/dv differ by at most one bf16 ulp (1.2e-4 against a 0.01-0.02
relative tolerance), so the replay is numerically equivalent within bf16 rounding — but not bitwise, which is
worth noting before a graph path is offered as a supported mode.

That closes the launch-overhead question: the 0.118 ms host gap plus ~13 small gaps are real, and a graph
recovers essentially all of it. Whether to ship a graph mode is a product decision (the caller has to own
static buffers), so it is recorded as a measured opportunity rather than a default change.

## 10. R13 — the router is launch-bound, and the graph only pays on reused inputs

**Router.** `bench_stages.py` already says `route_tiers` is one native call whose parts cannot be split from
Python. The static reading of `rep_api.cpp` and `topk.cu` explains the 0.10 ms difference between `route`
(0.262) and `route.one_budget` (0.162) without needing a kernel rewrite: one `launch_global_topk` is nine
kernels (3 x hist + find, then mask / cut / emit-rows / cols / colscan / cols) over NB^2 = 65,536 scores per
head, which is only 4 MiB per histogram pass. The bandwidth is negligible; the **launch latency of eighteen
tiny kernels** across the two calls is the cost. A single-histogram fusion would remove three 4 MiB passes and
keep the rest, so it is not worth a kernel rewrite. The router lever is launch overhead, not redundant work.

**Graph, with realistic inputs.** `bench_graph.py` gained a `graph.replay_with_copy` row: copy fresh q/k/v/gate
into the captured buffers, then replay — what a denoising step that produces new data each call actually pays.

| row | ms |
|---|---:|
| eager | 4.045 |
| graph.replay (captured tensors reused) | 3.940 |
| **graph.replay_with_copy (fresh inputs)** | **4.118** |
| graph.host_call | 0.003 |

So the 0.116 ms recoverable in the static case is **more than eaten by the input copies** (~0.18 ms for
3 x 64 MiB + gate): a graph is a loss for a caller with new inputs every step. Equivalence stays fine
(forward bit-identical, grads within one bf16 ulp), so this is a usage-shape conclusion, not a correctness one.
The CUDA-graph idea is dropped rather than shipped.

With that, both remaining cheap candidates (router, graph) are measured and rejected, which leaves one
substantive lever: the tier-A backward co-design — 113,000 B of dynamic shared memory pins it to two CTAs per
SM, and cutting that below ~76 KB is the only way to a third CTA. That is kernel work for account 1.

### 10b. Why the tier-A 3-CTA idea is a redesign, not a tuning knob

Shared memory of `bwd_attend_ker<128>` (fine_tk_h100.cu), 113,000 B requested:

| tile | KB |
|---|---:|
| `k_tile` `st_bf<64,128>` | 16 |
| `v_tile` | 16 |
| `q_tile` | 16 |
| `og_tile` (dO) | 16 |
| `qg_tile` `st_fl<64,128>` — the fp32 dQ staging accumulator | **32** |
| `attn_tile` (64x64 bf16) | 8 |
| allocator metadata + barriers | ~6 |

Three CTAs per SM need <= 75.6 KB each, so **at least 35 KB has to go**, and the only item that size is the
fp32 dQ accumulator. It exists because the kernel is KV-owner: each CTA owns one KV block and loops over query
blocks, so dQ is accumulated across CTAs and therefore needs a global fp32 destination and an smem staging
tile. Removing it means changing the ownership direction (query-owner, where dQ never crosses CTAs and dK/dV
do instead) — that is a different kernel, not a smaller one.

The one cheaper route worth asking account 1 about: Q and dO as register A-operands (RS wgmma) would drop
32 KB of smem to ~78 KB, which is within 1 KB of the three-CTA threshold; whether the RS layout is legal for
those two operands and whether the remaining metadata can be trimmed enough is an open question, not a
conclusion. That is the first thing to put to account 1 when its quota resets.

## 11. R14 — overlapping the table build with routing does not pay either

`fine.tables` measured 0.070 ms (5.4% of the full forward) and runs serially right after `route_tiers`
(0.262 ms). The build needs only K/V and `blk_cnt`; only the flattened tier-B log counts need the selection,
so CFG38 starts `rep_tk_build` on a second stream right after `pool_fwd` and lets `route_tiers` run on the main
stream, joining before the fine kernel (record_stream + wait_stream, log counts still on the main stream).

Job 806374, source `d9668da`, all suites passing:

| row (ms) | CFG1 | CFG38 | ratio |
|---|---:|---:|---:|
| `full_op ... f` | 1.2782 | 1.3007 | 1.0176 |
| `full_op ... f+b` | 4.0259 | 4.0310 | 1.0013 |
| `fine_op ... f` | 0.8303 | 0.8387 | 1.0101 |

A/A floor was 0.9861-1.0008 on those rows. Stage view: `fine` fell from 0.857 to 0.821 (the build did overlap)
but `route` rose from 0.263 to 0.271 and `total` rose from 1.279 to 1.297 — **the two streams contend more than
the overlap saves**. So route is not leaving the GPU idle enough to hide 70 us behind it.

Measurement note for whoever reads `bench_stages.py`: its `fine.tables` row calls `tk_tables` directly, so it
keeps reading ~0.070 in CFG38; the overlap only shows up in `fine` and `total`, which is why those are the rows
that decide.

That is the eleventh refuted direction. Remaining untried items from the plan (L2 access-policy window, TMA
prefetch, persistent grid) are all kernel or C++ changes and need account 1, which resets at 11:30.

## 12. R15 — a route-only graph works and is bit-exact; and the pooling math is reusable

**Route-only CUDA graph (job 806394, source `43c88bd`, measurement only).** Capture `C.route_tiers(qr, kr, …)`
alone — its inputs are the *pooled* q/k (8 MiB total, not the 192 MiB of q/k/v) — and replay:

| row | ms |
|---|---:|
| route.eager | 0.2686 |
| route.replay | 0.2381 |
| route.replay_with_copy (fresh pooled inputs, realistic) | **0.2455** |
| host cost of one replay | 0.0030 |

That is **−0.023 ms in the realistic case (−8.6% of route, 1.8% of the full forward)**, and all eight outputs
(`q2k_ptr/idx`, `k2q_ptr/idx`, `bits_a`, `bits_ab`, `b_ptr/idx`) are **bit-identical** to eager, including when
the pooled inputs change. Unlike the full-operator graph, the copy is small enough that the win survives.

**Mathematical reuse between routing and the representative tables.** `pool_kernel` (csrc/vsa/router/pool.cu)
already computes, for R = 16:

- `kr` = mean of each 16-token K group = **exactly the tier-B K representative** when `mid_reps = 64/R = 4`;
- `kc`, `vc` = mean of each 64-token block = **exactly the tier-C K and V means** that tier C uses.

`rep_tk_build` recomputes all of those from raw K/V (a 128 MiB read) even though the router pass has just
produced them. Only the tier-B **V** representatives (16-token V groups) are not in the pool output. So the
table build can consume pools instead of reducing K and V again — a genuine reuse, with one caveat: the
summation order would differ from today's, so the result is tolerance-equivalent (bf16 ulp), not bit-identical,
and it needs the usual output/PSNR check rather than the bitmap-equality check.

This also explains why the two-stream overlap in §11 did not pay: the two passes are reading the same tensors,
so overlapping them competes for the same bandwidth. The right fix is to **stop doing the work twice**, not to
schedule it twice — i.e. one pass over K/V that emits router pools, coarse means and the representative tables,
which is a single fused kernel rather than a stream trick.

## 13. R16 — accepted: cache the shape-only coarse CSR metadata (CFG39, now the default)

`_coarse_meta` builds three small int32 tensors purely from `(B, H, nb, nbc_pad)` on **every** forward call.
The stage timing put that row at 0.084-0.095 ms; the tensors never depend on data and the backward only reads
them, so caching them by shape is exact.

| row (ms) | CFG1 (rebuild) | CFG39 (cache) | ratio |
|---|---:|---:|---:|
| `full_op ... f` | 1.2878 | 1.2542 | **0.9739** |
| `full_op ... f+b` | 4.0265 | 3.9876 | **0.9904** |
| `fine_op ... f` | 0.8598 | 0.8697 | 1.0114 |

A/A floor on that build: `full_op f` 0.9983, `f+b` 0.9978, so the forward gain is 13x the floor. The stage
script's own `coarse.meta` row does not move (it measures `_coarse_meta` directly rather than through the
operator), but `total` fell 1.292 -> 1.258, which is the operator path the change actually targets.

Promoted to the default; **CFG40** rebuilds per call and stays as the history arm.

That is the third accepted change, and the second one found by looking at the *host* side rather than the
kernels — the two together are now worth more than the kernel work that preceded them.

## 14. R17 — the CFG39 promotion is confirmed; the route graph does not pay in place

Job 806430, source `2974047`, all suites passing (the generic suite caught a `row_norm` double-mapping in my
first CFG41 attempt before any timing, which is exactly what it is for).

**CFG39 promotion verified** — default (cached) vs CFG40 (rebuild per call):

| row (ms) | default | CFG40 | ratio |
|---|---:|---:|---:|
| `full_op ... f` | 1.2528 | 1.2924 | **1.0316** |
| `full_op ... f+b` | 3.9826 | 4.0286 | 1.0115 |

A/A floor 0.9921 on `full_op f`, so the cache is worth 3.2% of the full forward — a little more than the 2.6%
measured at adoption.

**CFG41 (route-only CUDA graph, in the operator) — no gain.** The isolated probe had replay at 0.2455 ms
against 0.2686 eager with bit-identical outputs, but inside the operator:

| row (ms) | CFG1 | CFG41 | ratio |
|---|---:|---:|---:|
| `full_op ... f` | 1.2617 | 1.2732 | 1.0091 |
| `full_op ... f+b` | 3.9964 | 4.0096 | 1.0033 |

So the launch saving is eaten by the pooled-input copies plus the output clones plus the fact that the replay
is serialized behind pool/coarse on the same stream. Recorded as the twelfth refuted direction; the graph code
stays behind CFG41 (off by default) as the measured negative.

Running total against the pre-round-4 build: full forward 1.3704 -> 1.2528 (**-8.6%**), full F+B
4.1789 -> 3.9826 (**-4.7%**), with three accepted changes and twelve refuted directions.

## 15. R18 — `fine.tables` is 88% table build, so the pool-reuse idea is worth a kernel change

`fine.tables` is 0.0692 ms. Split on the same build (job 806440, source `9a4c1a0`):

| row | ms | note |
|---|---:|---|
| `rep_tk_build` | **0.0610** | reads K and V (128 MiB), writes the swizzled tables |
| `rep_tk_lcb` | 0.0058 | gathers the flattened tier-B log2 counts |
| `tables.total` | 0.0692 | build + lcb = the `fine.tables` row |

0.061 ms for 128 MiB is ~2.1 TB/s effective, i.e. the build is read-bound, and §12 showed the router pass has
already computed exactly those means (`kr` = 16-token K groups, `kc`/`vc` = 64-token block means). Only the
tier-B **V** representatives are missing from the pool output.

So the reuse has a real target: feeding pools into the build turns a 128 MiB read into a ~10 MiB read, worth
roughly 0.04-0.05 ms (3-4% of the full forward) — provided the extra V 16-token pools are produced inside the
existing pool pass rather than by a second kernel that re-reads V (which would halve the gain).

Plan for account 1 when its quota returns:

1. extend `pool_kernel` to write the 16-token **V** group means alongside `qr`/`kr` (one extra destination in
   the `tensor == 2` branch; do not add a second pass over V);
2. add `build_kernel_pooled<D>` that mirrors `build_kernel`'s swizzle-record writes
   (`sw_off(pi, b, g)`, the `phi = (pi*RM)&7` chunk XOR, the `NB` zero record and the `Bt` tail rows) but reads
   group means instead of accumulating tokens;
3. gate it as CFG42, keep the default path, and verify with the TK suite plus an output comparison against
   CFG1 — this one is **tolerance-equivalent, not bit-identical**, because the group means are rounded to bf16
   before the block mean is formed rather than after.

## 16. R19 — accepted: representative tables built from the router pools (CFG42, now the default)

`rep_tk_build` re-read K and V (128 MiB, 0.061 ms) to recompute means the router pass had already produced:
`kr` = the 16-token K group means (== tier-B K representatives when `64/mid_reps == R`), `kc`/`vc` = the
64-token block means (== tier-C representatives). The only missing piece was the 16-token **V** groups, which
`pool_fwd_vr` now writes inside the same pass (`vr`, 4 MiB, no extra read).

`build_kernel_pooled` assembles the same swizzled records (`sw_off`, the `phi = (pi*RM)&7` chunk XOR, the `NB`
zero record, the `Bt` tail rows) and the same log-count tables, but reads the pooled means instead of
accumulating tokens. Job 806481, source `41cd546`, TK suite passing under CFG42 (349 checks):

| row (ms) | CFG1 (rebuild) | CFG42 (pooled) | ratio |
|---|---:|---:|---:|
| `full_op ... f` | 1.2494 | 1.2035 | **0.9633** |
| `full_op ... f+b` | 3.9930 | 3.9550 | **0.9905** |
| `fine_op ... f` | 0.8549 | 0.8776 | 1.0265 |

A/A floor on that build: `full_op f` 1.0051, `f+b` 0.9985, so the forward gain is ~7x the floor. `fine_op f`
moving the other way is the one row not touched by this change (the fine-only operator never pools) and sits
inside that row's own drift (A/A 1.0151 there).

Promoted to the default; **CFG43** rebuilds from K and V and stays as the history arm. This one is
**tolerance-equivalent, not bit-identical** — group means are rounded to bf16 before the block mean is formed —
so it was accepted on the suite plus the output cross-check against the generic backend, not on bitmap equality.

Four accepted changes now, and the two biggest are not kernel arithmetic at all: one removes a memset, one
removes a duplicate read.

## 17. R20/R21 — pool reuse promoted and verified; the route graph is closed for good

**CFG42 promotion verified** (job 806499, source `0f4b351`): default (pooled tables) vs CFG43 (rebuild from
K/V) gives `full_op f` 1.2167 vs 1.2465 (1.0245) and `f+b` 3.9700 vs 3.9932 (1.0058), against an A/A floor of
1.0077/1.0035. The later adoption run measured 0.9633 on the same row; the difference between the two runs is
cross-run drift, and both are well outside their own floor.

**CFG41 second attempt, without the output clones** (job 806510, source `b3b6eab`). The first attempt cloned
the eight selection tensors, which cost roughly the same eight launches the graph was meant to save. Returning
the captured tensors directly:

| row (ms) | CFG1 | CFG41 | ratio |
|---|---:|---:|---:|
| `full_op ... f` | 1.2128 | 1.2107 | 0.9983 |
| `full_op ... f+b` | 3.9452 | 3.9458 | 1.0001 |

A/A floor 0.9994/1.0000. So removing the clones was worth about 0.9% (the first attempt measured 1.0091) but
still leaves nothing: the 0.023 ms the isolated route probe measured does not survive integration, because the
replay is queued behind pool and coarse on the same stream and the GPU is not idle there. The route-graph line
is closed with two independent measurements rather than one.

Running total after four accepted changes: full forward ~1.213 ms, full F+B ~3.945 ms, from 1.3704 and 4.1789.

## 18. R22 — the stage script no longer describes the routed operator (measurement caveat)

Two A/A runs in one job (806532) put the current default at `full_op f` 1.1929/1.2000 and 1.2169/1.2017, and
`f+b` 3.9549/3.9627 and 3.9546/3.9427 — i.e. about 1.20 ms and 3.95 ms, with **2% drift between the two pairs in
the same job**. That drift is the reason every claim in this document is a same-run A/B ratio and not a
cross-run subtraction.

The stage rows now disagree with the operator by more than the drift, and the reason is structural rather than a
bug: `bench_stages.py` measures `rep_fine_attention` for its `fine` row and `tk_tables` for `fine.tables`, but
under the current defaults the **routed** operator builds its tables from the pooled means and never calls either
of those. So `fine.tables` still reads 0.0696 while the operator pays roughly 0.01, and the stage sum
(1.361) exceeds `total` (1.212) because it is describing a path the operator no longer takes.

Fix for account 2 when it is back: make `bench_stages.py` either (a) call the pooled builder for the `fine.tables`
row under the new defaults, or (b) rename the rows so a reader cannot add them into a decomposition of the routed
operator. Until then, `total`, `full_op f` and `f+b` are the only rows that describe the shipped path; the
per-stage rows are a description of the fine-only API.

## 19. R23/R24 — accepted: the two top-k budget chains run on two streams (CFG44, now the default)

`route` is 0.262 ms and one chain alone is 0.162, so the second chain is almost pure launch latency: each
`launch_global_topk` is fourteen tiny kernels. Once `scores` exists the two chains are independent — the A|B
chain writes `ab_ptr`/`bits_ab`, the tier-A chain writes `q2k_ptr`/`q2k_buf`/`bits_a` — so CFG44 gives the A|B
chain its own top-k workspace, its own kv->q scratch, and a non-blocking stream, with events ordering it after
the score pass and joining before the next iteration rewrites `scores`.

The first attempt looked unstable (forward 0.9772, F+B 1.0046) because it created and destroyed the stream and
both events **on every routing call**. With them process-static, two independent A/B pairs in one job
(806591, source `0f640d4`):

| pair | `full_op f` | `full_op f+b` |
|---|---|---|
| 1 | 1.2192 -> 1.1367 (**0.9323**) | 3.9765 -> 3.8876 (**0.9777**) |
| 2 | 1.2005 -> 1.1360 (**0.9463**) | 3.9786 -> 3.8980 (**0.9797**) |
| A/A | 1.0086 / 0.9829 | 0.9994 / 1.0015 |

So the forward is 5-7% faster and F+B **2% faster**, both outside their floors. The kernels, their order inside
each chain and every number are unchanged; only the stream placement differs, so this is exact rather than
tolerance-equivalent.

Promoted to the default; **CFG45** restores the sequential order. Five accepted changes now, and this is the
largest single one since the gather.

## 20. R25 — parallel top-k promotion verified

Job 806606, source `518a0f8`, all suites passing. Default (two streams) vs CFG45 (sequential):

| row (ms) | default | CFG45 | ratio |
|---|---:|---:|---:|
| `full_op ... f` | 1.1351 | 1.2190 | **1.0739** |
| `full_op ... f+b` | 3.9016 | 3.9698 | **1.0175** |

A/A floor 1.0124 / 1.0013, so the sequential order is 7.4% slower on the forward and 1.75% slower on F+B.
Absolute state on this build: **full forward ~1.135 ms, full F+B ~3.90 ms**, from 1.3704 and 4.1789 on the
pre-round-4 build — **-17.2% forward, -6.6% F+B** across the five accepted changes.

## 21. Handoff — what is verified, what is open, what to try next

**Verified and in the default path** (each with a same-build A/A arm and a suite gate, and the last four also
re-measured after the default flipped):

| # | change | CFG that restores the old behaviour | commit |
|---|---|---|---|
| 1 | tier-B gather via CTA-wide `cp.async` | 20 | `5422727` (R4), promoted in `3da1d0c` |
| 2 | tier-A dK/dV single-writer store, no `kg`/`vg` memset | 37 | `0251613` (R11), promoted in `734441b` |
| 3 | coarse CSR metadata cached by shape | 40 | `b980e3d` (R16), promoted in `083da7c` |
| 4 | representative tables from the router pools | 43 | `9148ed0` (R19), promoted in `0f4b351` |
| 5 | two top-k budget chains on two streams | 45 | `4edc2a6` (R23), promoted in `518a0f8` |

**Open, in the order I would attack them:**

1. **tier-A backward at three CTAs** — needs the analysis in §10b confirmed against the code first. My reading:
   113,000 B of smem with the 32 KB fp32 dQ staging tile as the only removable block, and RS operands cost
   +64 registers against a 245-register kernel. If that is right, the only route is a query-owner rewrite and
   it is a new project, not a tuning round.
2. **Router kernel count** — `route` is still 0.26 ms and it is eighteen kernels; the two chains now overlap,
   but the individual chains are launch-bound. Merging the mask/cut/emit/cols stages per chain is the next lever.
3. **`pool` bandwidth** — 0.092 ms for 207 MiB is ~1.5x off the roofline; a vectorized two-column-per-thread
   variant is the obvious first try.
4. **`rep_unpool` + `finalize` fusion** — 0.245 ms combined for two elementwise fp32->bf16 passes; fusing them
   saves one launch and not much else, so it is the smallest of the four.

**Measurement lessons that cost real time and should not be re-learned:**

- Per-call `cudaStreamCreate`/`cudaEventCreate` inside a hot path cost more than the launch gaps they fill —
  that single bug turned an accepted 7% win into an "unstable" result once (R23 vs R24).
- `bench_stages.py`'s `fine` and `fine.tables` rows describe the fine-only API, not the routed operator, since
  the latter builds its tables from the pools; only `total` and the `bench_compare` rows describe what ships.
- Absolute times drift ~2% between runs on this machine, so only same-run A/B ratios are claims.
- The cluster occasionally rejects `sbatch` with "Access/permission denied" and leaves a `failed` receipt that
  then trips the submit assertion; clear `metadata/<sha>-submit.json` and retry.

## 22. R26 — pre-handoff review of the default path found two real bugs

Neither is a performance issue; both are "works on the benchmark shape, breaks elsewhere".

1. **Short sequences would raise instead of falling back.** `build_kernel_pooled` asserts
   `kr rows == NB * mid_reps`, but `pool_fwd_vr` pads its row count to 64, so the two agree only when
   `NB * mid_reps` is 64-aligned. At `mid_reps = 4` that means `NB % 16 == 0`: L=16384 (NB=256) is fine and was
   the only shape ever measured, but L=640 (NB=10) would have thrown. The operator now checks
   `(nb * mid_reps) % 64 == 0` and takes the build-from-K/V path otherwise (commit `5720453`).
2. **The parallel top-k path kept a process-level static stream.** A stream belongs to the device that created
   it, so on a second GPU the A|B chain would have been launched on the wrong device's stream. It is now
   recreated when the current device changes (commit `7c23b15`); the old stream is deliberately not destroyed
   because it may still be in use.

Verified on job 806627 (source `7c23b15`): cpu 7 / generic 287 / tk 349 all passing, A/A 0.9820 / 1.0009 — no
performance change, as expected for correctness fixes.

Also reviewed and found correct rather than broken: `k2q_buf` is already sized for `BH * keff_ab` (the larger
budget), so the second scratch is `empty_like` of a correctly sized buffer; `launch_tier_b_lists` runs after the
per-iteration join, so it always sees a finished `ab_ptr`/`ab_idx`/`bits_ab`. No debug leftovers in the diff
between the R12 push and `HEAD`.

## 23. R27 — all five old paths measured against the current default, in one run

This is the cleanest single table in the document: one build, one thermal state, one A/A arm, and each accepted
change's "restores the old behaviour" config compared against the default as it now stands. Job 806639, source
`7c23b15`, all suites passing. Baseline is the **current default**; the candidate column is the **old path**.

| old path | `full_op f` | `full_op f+b` |
|---|---|---|
| A/A (no change) | 1.1416 -> 1.1344 (0.994) | 3.9018 -> 3.8974 (0.999) |
| CFG20 old tier-B gather | 1.1410 -> 1.2195 (**1.069**) | 3.8959 -> 3.9747 (**1.020**) |
| CFG37 old tier-A store | 1.1420 -> 1.1366 (0.995) | 3.8951 -> 3.9917 (**1.025**) |
| CFG40 rebuild coarse metadata | 1.1279 -> 1.1753 (**1.042**) | 3.8843 -> 3.9308 (**1.012**) |
| CFG43 rebuild tables from K/V | 1.1144 -> 1.1733 (**1.053**) | 3.8859 -> 3.9333 (**1.012**) |
| CFG45 sequential top-k chains | 1.1277 -> 1.2064 (**1.070**) | 3.8966 -> 3.9570 (**1.016**) |

Every one of the five old paths is slower than the current default: **4.2-7.0% on the full forward and
1.2-2.5% on F+B**, against an A/A displacement of 0.6% and 0.1%. CFG37 shows nothing on the forward row, which
is correct — it changes only the backward.

Two things this table settles that the individual rounds could not: the five changes are all live at once (no
change silently disabled another), and their contributions are additive rather than overlapping, which is what
the -17% forward and -6.6% F+B totals imply.

## 24. R28 — corrected tier-A shared-memory inventory (my earlier list was wrong)

While account 1 independently checks the three-CTA argument, I re-read the allocation in `bwd_attend_ker` and
found my own inventory was wrong in a way that changes the arithmetic.

The fp32 dK/dV accumulators are **aliased over the bf16 tiles**, not allocated separately:

```cpp
kg_tile(*kg_smem) = reinterpret_cast<kg_tile*>(&k_smem[0].data[0]);   // fp32 dK over the bf16 K tile
vg_tile(*vg_smem) = reinterpret_cast<vg_tile*>(&q_smem[0].data[0]);   // fp32 dV over the bf16 Q tile
```

So the declared set at D=128 is: k, v, q, og at 16 KB each (64 KB), `qg` at 32 KB (the fp32 dQ staging tile),
the two row vectors, and the 64x64 attention tile at 8 KB — about 104 KB plus allocator metadata, which is
where the hardcoded 113,000 B comes from. My earlier phrasing implied kg and vg were additional 32 KB blocks;
they are not.

**The consequence matters:** 110 KB minus the 32 KB dQ staging tile is still about 78 KB, which is **2.4 KB
above the 75.6 KB three-CTA limit**. So the dQ tile is necessary to remove but not sufficient — reaching three
CTAs needs that *plus* a few more KB shaved from the 8 KB attention tile or the row vectors, and the dQ staging
tile exists because the kernel is KV-owner. That makes the three-CTA route look worse, not better, than the
earlier note said: it is a coordinated removal, not one block.

This is exactly the kind of thing the independent check is for; I will reconcile its answer against this
arithmetic rather than against my earlier summary.

## 25. R30 — the independent tier-A check corrected two of my claims (and confirmed the conclusion)

Account 1's verdict, against the code rather than my summary:

**The conclusion holds — there is no route to three CTAs per SM — but the binding constraint is registers, not
shared memory.** Two of my claims were wrong:

1. **Shared memory can be met without register operands.** Each thread can add its `qg_reg` straight into global
   dQ with `red.global.add.f32` instead of staging it through the 32 KB tile (the fp32 reduction across CTAs is
   already unordered, so determinism is unchanged), and `l`/`d` can be allocated after the dS tile to drop the
   allocator padding. That lands near **74,240 B**, under the ~76.8 KB that three CTAs allow. My "2.4 KB short"
   arithmetic in §24 was right about the tile, but wrong that removing it alone was the only option.
2. **The register-operand idea was not available in the first place.** Q and dO are the **B operands** of
   `dK += dS^T Q` and `dV += P^T dO`, and wgmma reads B only from shared memory — my +64-register argument was
   analysing an A-operand path that these products do not have. (Streaming them in 64-column halves works
   arithmetically, but each half is used twice, before and after the softmax, so both halves stay resident.)

**What actually blocks it:** three CTAs of 128 threads allow **168 registers per thread**, and the dK and dV
accumulators alone are 2 x 64 = **128 registers** before S, dP, dS and the 64-register dQ fragment. The only
routes down are splitting dK and dV into separate passes (about 1.5x the score recompute) or a second
warpgroup, which does not change registers per thread. CFG23 already measured forced 168 registers at **1.87x
slower**, so no shared-memory saving reaches three CTAs.

The tile inventory in §24 is confirmed: K/V/Q/dO 16 KB each, fp32 dQ staging 32 KB, l/d 1 KB each after
padding, dS 8 KB = 109,568 B plus base alignment, with dK/dV aliased over the K/V and Q/dO regions. The tier-A
line is closed.

## 26. R30 — accepted: the router's column stage is one kernel (CFG46, now the default)

Account 1 wrote it: `topk_cols_fused_kernel` counts column j with the same reads as `topk_cols`, scans the
counts exactly as `topk_colscan`, then emits column j's rows — three kernels per chain become one, and the
column stage produces only integers, so the kv->q CSR is bit-identical to CFG1.

| measurement | `full_op f` | `full_op f+b` |
|---|---|---|
| adoption (job 806682, CFG1 vs 46) | 1.1386 -> 1.1235 (0.9867) | 3.8860 -> 3.8730 (0.9967) |
| promotion check (job 806691, default vs 47) | 1.1277 -> 1.1394 (1.0104) | 3.8728 -> 3.8852 (1.0032) |
| A/A floor | 0.9940 / 1.0037 | 1.0003 / 1.0015 |

Promoted to the default; **CFG47** restores the three-kernel form. Six accepted changes now: full forward
~1.128 ms and full F+B ~3.873 ms, from 1.3704 and 4.1789 — **-17.7% forward and -7.3% F+B**.

**Account 2 also corrected the harness**, and one correction affects how earlier numbers should be read:

- arms 38-45 had been registered as aliases, which labelled their `full_op` forward rows as "controls" even
  though moving that row is the whole point of those arms; they are now routed arms whose only controls are the
  kernel and fine-only rows. The numbers themselves are unchanged.
- CFG42 is byte-equal again (it is the default path now); only CFG43 is tolerance-equivalent.
- **CFG38 is superseded and no longer measures what it was adopted to test.** Since the pooled tables became the
  default, the `if pooled:` branch runs first and never consumes the side-stream build, so a CFG38 run after
  `0f4b351` pays for a discarded table build rather than an overlap. Its negative result in §11 stands as
  originally measured; it must not be re-measured on the current default.

## 27. R31 — the pool vectorisation is inside the noise; merging cut+emit is slower

Job 806704, source `6d5d742`, all suites passing including both new arms.

| arm | `full_op f` | `full_op f+b` | verdict |
|---|---|---|---|
| A/A | 1.1313 -> 1.1177 (0.9880) | 3.8944 -> 3.8747 (0.9949) | floor |
| CFG48 pool, two column pairs per thread | 1.1302 -> 1.1247 (0.9951) | 3.8918 -> 3.8711 (0.9947) | **inside the floor** |
| CFG49 router cut+emit merged | 1.1290 -> 1.1522 (**1.0205**) | 3.8952 -> 3.9078 (1.0032) | **2% slower** |

- **CFG48 is not adopted.** It is byte-identical and 0.5% faster on the forward, but that A/A pair displaced
  1.2% on the same row and 0.5% on F+B, so the measurement cannot separate the change from the run. The pool
  pass is latency-bound rather than instruction-bound, which is consistent with halving the threads per CTA not
  helping: there are simply fewer warps to hide the row-stride loads.
- **CFG49 is refuted.** Merging `topk_cut` and `topk_emit_rows` into one 1024-thread CTA per head serialises the
  emit stage across the 32 warps that used to run as separate CTAs, and costs 2% of the forward. Account 1 was
  right to leave `topk_mask` out of the merge; this shows the emit stage should stay out of it too.

That is fourteen measured refutations. The router is now back to what it was: the two chains overlap, the column
stage is fused, and the remaining launches are the histogram/select chain, which is sequential by construction.

## 28. Account 2 disagrees with my tier-A wording, and it is right to

Account 2 recorded the tier-A line as closed with **both** limits documented rather than "registers, not shared
memory", and its reasoning is worth keeping:

- registers: dK and dV alone are 128 per thread against a 168 budget for three CTAs;
- shared memory: the current tiles need 107,008 B against 76,800 B allowed by the kernel's formula, on a
  113,000 B allocation.

Account 1's ~74,240 B figure assumed *removing* the 32 KB staging tile, which is a code change and not the
current layout, so the two statements are about different things and both are true. Account 2 also made a point
I had missed: **CFG23 kept the same 113,000 B allocation**, so its 1.87x measurement is the cost of the register
cap at two CTAs and does not isolate shared memory at all. The honest summary is therefore "both limits block
three CTAs today; the register budget is the one that survives every layout change we found", and the handoff
document now says exactly that.

## 29. R32 — the operator's own backward is not run-to-run deterministic

Two eager runs, identical inputs, in one process (job 806746):

| tensor | bit-identical | differing elements |
|---|---|---|
| `o` | yes | 0 / 33,554,432 |
| `q2k_idx`, `bits_a`, `bits_ab` | yes | 0 |
| `dq` | **no** | 318 / 33,554,432 |
| `dk` | **no** | 1,194 / 33,554,432 |
| `dv` | **no** | 347 / 33,554,432 |

The forward is bit-exact across runs; the gradients are not, because they are built with fp32 reduce-adds across
CTAs whose order is not fixed. A second run reproduced this with slightly different counts (335/1,149/351), which
is the signature of a race in the *order* of adds rather than a bug.

Two consequences:

- **The whole-operator CUDA graph was rejected on a false premise.** In round 13 its gradients differed from
  eager by one bf16 ulp and that was treated as the graph changing something. It was not: eager differs from
  itself by the same amount. The graph's own numbers stand (3.940 ms replay against 4.045 ms eager, 4.118 ms with
  fresh inputs copied in), so it is still not worth shipping for a real training loop — but the equivalence
  objection is withdrawn.
- Any future arm whose evidence is a gradient comparison must use a tolerance, not byte equality. The harness's
  cross-check against the generic backend was already tolerance-based, which is why it still means something.

## 30. R32 — the adopted arms re-verified under the hardened harness

Account 2 found six places where the harness could silently accept a wrong result; the one that matters here is
that **the output cross-check never reached the routed operator** — it only ran the fine path. So pooled tables,
the coarse branch and the fusion had no correctness check, and CFG43's "tolerance-equal" label was checking no
tolerance at all.

After taking the fix (commit `10aa818`), the three adopted arms were re-measured with the routed cross-check
comparing `o`, `dq`, `dk`, `dv` and `dgate` against the generic backend (job 806746):

| arm | routed cross-check |
|---|---|
| CFG42 pooled representative tables | passes |
| CFG44 parallel top-k chains | passes |
| CFG46 fused column stage | passes |

The timings in that run are not meaningful for these arms — the cross-check itself runs inside each child and
changes the thermal profile — so the adoption numbers in §16, §19 and §26 remain the ones to quote. What this run
establishes is that the three changes are **correct under a check that actually covers the routed operator**,
which is the first time that has been true for CFG42 and CFG46.

The checker script in the job template had to be relaxed in the same round: it hardcoded six output checks and
rejected the new, larger set. It now requires every check to pass, requires the router digest and requires at
least one routed cross-check.

## 31. R33 — the last three accepted changes verified; the remaining ideas assessed and declined

**Verification closed for the three arms the routed cross-check cannot see** (job 806768, source `d74de3f`).
Account 2 pointed out that both backends share the pool pass, router, coarse branch and fusion, so on that shared
code the generic side moves with the TK side and the check is blind. Coarse cache, parallel top-k and the fused
column stage therefore rest on a same-run comparison instead. Those three were re-run under the hardened harness:

| arm (old behaviour vs default) | result |
|---|---|
| CFG40 coarse metadata rebuilt per call | passes |
| CFG45 sequential top-k chains | passes |
| CFG47 three-kernel column stage | passes |

With §30, all six accepted changes now have a verification path that the hardened harness actually covers:
pooled tables, gather and tier-A store through the routed cross-check; the other three through same-run byte
equality against the config that restores what they replaced.

**The three remaining ideas, assessed and declined by account 1:**

- **TMA prefetch: no.** CFG11 already loaded K a tile early straight into shared memory — a strictly stronger
  form of "get it closer sooner" — and measured 1.000, so K/V latency is not exposed at three CTAs. The tier-B
  tables are already L2-resident while their head is active (§29's analysis), so prefetching toward L2 is a
  no-op; the remaining tier-B cost is L2-to-smem copying, which a prefetch does not shorten. It also corrected my
  attribution: the 0.167 ms quoted in round 4 is the whole tier-B-only kernel row, not the gather alone.
- **Persistent router: legal and exact, but not now.** The dependencies are per head, not across the grid, so
  the chain can live in one launch with only per-head synchronisation: one CTA per head (`__syncthreads`), or a
  thread-block cluster per head with the histogram reduced through distributed shared memory and `cluster.sync()`
  between stages. The cluster shape is the one it would choose; a grid barrier is the shape it would avoid,
  because cooperative launch fails when the grid cannot be resident and a hand-written spin deadlocks — both made
  worse by two chains running at once. Exactness is preserved either way: the histograms are integer atomics, the
  select is deterministic and the later stages are integer-only. The reason to wait is that most of the saving
  overlaps what a CUDA graph already achieves (0.118 ms for the entire F+B chain), leaving an estimated
  0.03-0.05 ms — about 1% of F+B — for a rewrite of most of `topk.cu`.
- **Warp specialisation: its own earlier suggestion, retracted.** `setmaxnreg` moves registers between warps; it
  does not add any. Two CTAs of 128 threads at 245 registers already use ~63K of the 65,536-register file, and an
  FA3-style layout with two consumer warpgroups at ~240 plus a producer warpgroup at ~24 uses about the same file
  and the same eight consumer warps per SM. So there is no headroom to unlock, only scheduling to change — a
  producer issuing loads that double-buffering already showed are hidden, and a ping-pong that two independent
  CTAs already approximate by interleaving. Plausible outcome 0-5% of the kernel, at most 2% of F+B, for a
  rewrite of VSA2's TK backward.

That closes the plan. Every item in NEXT_STEPS has either been adopted, measured and refuted, or assessed and
declined with the reasoning recorded; the accepted changes are verified under a harness whose blind spots have
been found and closed.
