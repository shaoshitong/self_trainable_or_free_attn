"""VSA2 balanced split-KV scheduling experiment on H100: baselines vs SplitKV on identical selected block pairs.

Arms (every result row carries backend / fwd_kernel / bwd_algorithm / bwd_variant / schedule labels):
  tk               production H100 path: ThunderKittens fine forward and backward (backend="tk")
  generic_v{0,2}   generic mma.sync kernels, unsplit (NB, BH) grids; v0 = KV-owner + atomic dQ (the generic
                   auto choice on H100), v2 = KV-owner dK/dV + separate Q-owner dQ pass
  split_v{0,2}_c*  the same generic kernels and backward algorithm as generic_v{0,2}, balanced split-KV tasks
  tk_split_c*      the same TK kernels as tk, balanced split-KV tasks: forward chunks -> fp32 partials + merge;
                   backward k2q chunks reduce-add dK/dV (and dQ, as before) into TK's fp32 accumulators
  tk*_rw / _rot / _rw_rot  TK backward options (--tk-bwd-opts): dQ reduce-add waits only for the shared-memory
                   read instead of global completion (_rw); rotated k2q walk start (_rot). Same pairs and operator.
                   Unsplit TK arms also time bwd_tk_skipdq_DIAG (dQ not written; diagnostic, invalid gradients).
  tk_auto_c*       --auto-chunks: SplitKV.auto(chunk), the GPU SplitPolicy decides per direction and per list
                   (--policy overrides, --auto-scratch-mib forward cap). plan_stats records the actual plans
                   (eligible / split rows, within budget, per-head split rows and chunks) for every split arm.
  *_fwdonly / *_bwdonly  --split-dirs fwd / bwd: SplitKV(fwd=True, bwd=False) / SplitKV(fwd=False, bwd=True);
                   rows carry split_fwd / split_bwd, and the fine-stage helpers time the unsplit kernels for the
                   direction that is not split. Default --split-dirs both (no extra arms).
Scheduling alone: tk_split* vs the unsplit TK arm with the same options (ref_unsplit), and split_vN vs generic_vN
(field vs_unsplit_same_kernels). split_vN vs tk
also changes the kernels (TK -> generic) and the backward algorithm (field vs_tk).

Scheduling experiment, not a claim about natural router distributions. Synthetic cases fix NNZ = NB*K per
(b,h) and only move edges between rows/columns; router cases run the real PSP router (exact global top-k) on
seeded random inputs ("router") or on seeded inputs with synthetic sink keys and per-block query temperatures
("router_skew"). Both arms of a case share the same Selection object (identical pairs), inputs and dO.

Seeds: at one shape every synthetic case uses the same Q/K/V/gate/dO (data_seed, independent of the case) and its
own mask_seed, so cross-topology differences come from the masks only. Router cases replace Q/K by their router
inputs (qk_source). Rows carry shape_role: "reference" for B1/H20/D128/K64, else "diagnostic" (e.g. small K or
few heads to expose under-parallelism; not representative model settings). --arms filters arms by glob.

  uniform       every query block has K kv blocks, uniformly random columns
  hotq          NB/4 query blocks hold all NB*K edges (~4K each), the rest are empty
  hotq_mixed    NB/8 query blocks with 4K edges, the rest share the remainder (~4K/7 each, nonempty)
  hotq_extreme  K query blocks attend to all NB kv blocks, the rest are empty
  straggler_q   scheduling stress: per head ONE query block (the last, NB-1) attends to all NB kv blocks; the
                other NB-1 blocks share the remaining NB*K - NB edges (~K-1 each). A rare long straggler among
                many short tasks, placed at the tail of the unsplit Q grid
  hotkv         K per query block, K/2 of them from NB/16 hot kv columns (~8K per hot column)
  hotqkv        hotq rows, each taking min(NB/16, 2K) blocks from the hot kv columns

Selection modes (every row carries selection_mode / includes_router):
  frozen  all cases above, router cases included: the Selection is computed once at setup and passed to
          every call, so "full" timings are the operator without the router (scheduling diagnostic).
  routed  --routed-tokens (default 122880, case "router" inputs): vsa2_attention(topk=K) runs the PSP router
          and exact global top-k inside every forward, for every arm; complete VSA2 timing. Arms are
          checked to produce the identical selection (selection_identical_to_first_arm).

Real captures (--real-qkv, case "real:<label>", shape_role real_capture): captured attention Q/K/V (dict "qkv" =
3 x [1, N, H, D], loaded with weights_only=True, mmap=True, transposed to [1, H, N, D], cast to bf16), no model.
Selection: a fresh VSA2 PSP router + exact global top-k on the capture (never a selection stored in the fixture);
per-head NNZ is checked against the exact budget and per-head Q/KV degree statistics are recorded. Gate and dO
are seeded synthetic (gate_source / dout_source). Layout: 1-D sequence_layout by default, or --real-grid TxHxW
(build_layout cube tiles, assuming the captured order is raster (T, H, W)). Both a frozen (selection computed
once) and a routed (router inside every forward) measurement run per arm; frozen arms also report
vs_fp32_oracle_sampled_o, the complete-operator O on sampled query blocks against an independent fp32 computation.
These are operator/performance tests, not quality claims.

--phases (frozen arms): phases_ms times pool, coarse_meta, coarse_fwd, fine_fwd, fuse_fwd, fuse_bwd, coarse_bwd,
fine backward and finalize as independent callables on intermediates built as in ops._Function (same selection,
split flags, TK options), plus router_routed_calls_only. Each includes its own allocations/launches; their sum is
not the critical path of a full call. Complete F / B / F+B (full_ms) stays the acceptance metric.
Sampled O oracle: pass = finite and max|err| <= 0.02 * max|ref| + 1e-5 (forward O only, not a gradient oracle).

Measured with CUDA events after warmup, every sample kept: fine-stage timings (plan, chunk/direct tasks,
merge; KV-owner and Q-owner backward stages), complete operator forward / backward / forward+backward (the
planner and merges are inside), TK fine stages for the tk arm, scratch peak over the pre-call allocation (outputs included, same for both
arms), degree statistics, plan size (*_plan_info = [tasks, split rows, split slots, rows]; rows past the slot
cap stay direct) and output/gradient differences: vs_unsplit_same_kernels (scheduling only) and vs_tk. Raw rows go to --out (JSONL); a median summary is printed.

  # reference shape (B1/H20/D128/K64): frozen topologies + one routed length
  python benchmarks/vsa2_splitkv_bench.py --tokens 32768 76800 122880 220000 \\
      --cases uniform hotq hotq_mixed straggler_q hotkv hotqkv --chunks 32 64 128 --bwd-variants 2 0 \\
      --routed-tokens 32768 --out vsa2_splitkv_h100.jsonl
  # targeted follow-up: TK vs TK split only, no routed run
  python benchmarks/vsa2_splitkv_bench.py --tokens 32768 --cases straggler_q hotq_extreme --chunks 64 128 \\
      --arms tk 'tk_split_*' --routed-tokens --out vsa2_splitkv_tk_followup.jsonl
  # real capture (Wan 1.3B, 480p latent grid), TK arms only; inspect first with --real-inspect
  python benchmarks/vsa2_splitkv_bench.py --cases none --routed-tokens --arms tk 'tk_split_c64*' \\
      --real-qkv /path/to/real_qkv_fixture.pt \\
      --real-label wan1.3b --real-grid 21x30x52 --out vsa2_splitkv_real.jsonl
  # diagnostic under-parallelism (labelled shape_role=diagnostic): small K, few heads
  python benchmarks/vsa2_splitkv_bench.py --tokens 32768 --heads 2 --topk 4 --cases hotq_extreme straggler_q \\
      --chunks 32 64 128 --routed-tokens --out vsa2_splitkv_diag.jsonl
"""

import argparse
import fnmatch
import json
import math
import os
import platform
import subprocess
import sys
import time

import torch

from echosparse.vsa import ops

CASES = ["uniform", "hotq", "hotq_mixed", "hotq_extreme", "straggler_q", "hotkv", "hotqkv", "router", "router_skew"]
# where the long query rows sit in the Q grid (blockIdx order of the unsplit kernels)
PLACEMENT = {"hotq": "random_perm(mask_seed)", "hotq_mixed": "random_perm(mask_seed)",
             "hotq_extreme": "random_perm(mask_seed)", "straggler_q": "tail (q block NB-1 of every head)",
             "hotqkv": "random_perm(mask_seed)"}
REFERENCE_SHAPE = dict(B=1, H=20, D=128, K=64)


# ----------------------------------------------------------------------------- selections


def _spread(total, n):
    base, rem = divmod(total, n)
    d = torch.full((n,), base, dtype=torch.int64)
    d[:rem] += 1
    return d


def synthetic_degrees(case, nb, K, gen):
    """(hot-column degree, cold-column degree, hot-column mask) per query block of one (b,h)."""
    total = nb * K
    perm = torch.randperm(nb, generator=gen)
    hot_cols = torch.zeros(nb, dtype=torch.bool)
    dh = torch.zeros(nb, dtype=torch.int64)
    dc = torch.zeros(nb, dtype=torch.int64)
    if case == "uniform":
        dc[:] = K
    elif case == "hotq":
        dc[perm[: nb // 4]] = _spread(total, nb // 4)
    elif case == "hotq_mixed":
        n_long = nb // 8
        dc[perm[:n_long]] = 4 * K
        dc[perm[n_long:]] = _spread(total - n_long * 4 * K, nb - n_long)
    elif case == "hotq_extreme":
        dc[perm[:K]] = nb
    elif case == "straggler_q":
        dc[nb - 1] = nb
        dc[: nb - 1] = _spread(total - nb, nb - 1)
    elif case == "hotkv":
        hot_cols[torch.randperm(nb, generator=gen)[: nb // 16]] = True
        dh[:] = K // 2
        dc[:] = K - K // 2
    elif case == "hotqkv":
        hot_cols[torch.randperm(nb, generator=gen)[: nb // 16]] = True
        rows = perm[: nb // 4]
        deg = _spread(total, nb // 4)
        dh[rows] = torch.clamp(deg, max=min(nb // 16, 2 * K))
        dc[rows] = deg - dh[rows]
    else:
        raise ValueError(case)
    n_hot = int(hot_cols.sum())
    if (dh > n_hot).any() or (dc > nb - n_hot).any() or (dh < 0).any() or (dc < 0).any():
        raise ValueError(f"{case}: degrees exceed available columns at NB={nb}, K={K}")
    return dh, dc, hot_cols


def _pick(scores, deg):
    """Per row, the deg[r] columns with the largest random scores."""
    if scores.shape[1] == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    rank = scores.argsort(1, descending=True).argsort(1)
    return rank < deg[:, None]


def synthetic_selection(case, B, H, nb, K, seed, device):
    gen = torch.Generator().manual_seed(seed)
    gdev = torch.Generator(device=device).manual_seed(seed)
    mask = torch.zeros(B * H, nb, nb, dtype=torch.bool, device=device)
    for bh in range(B * H):
        dh, dc, hot = synthetic_degrees(case, nb, K, gen)
        dh, dc, hot = dh.to(device), dc.to(device), hot.to(device)
        s = torch.rand(nb, nb, generator=gdev, device=device)
        hi, ci = hot.nonzero().squeeze(1), (~hot).nonzero().squeeze(1)
        row = torch.zeros(nb, nb, dtype=torch.bool, device=device)
        row[:, hi] = _pick(s[:, hi], dh)
        row[:, ci] = _pick(s[:, ci], dc)
        mask[bh] = row
    nnz = mask.sum((1, 2))
    assert bool((nnz == nb * K).all()), f"{case}: NNZ per (b,h) must be NB*K"
    sel = ops.mask_to_selection(mask.reshape(B, H, nb, nb))
    del mask
    return sel


def router_inputs(case, B, H, L, D, seed, device):
    g = torch.Generator(device=device).manual_seed(seed + 1)
    q = torch.randn(B, H, L, D, generator=g, device=device)
    k = torch.randn(B, H, L, D, generator=g, device=device)
    if case == "router_skew":  # synthetic sink kv blocks and per-block query temperatures
        nb = L // 64
        sinks = torch.rand(B, H, nb, generator=g, device=device) < 1.0 / 32
        k = k + (sinks.repeat_interleave(64, -1)[..., None] * 3.0) * k.mean(2, keepdim=True).sign()
        temp = torch.exp(0.5 * torch.randn(B, H, nb, generator=g, device=device)).repeat_interleave(64, -1)
        q = q * temp[..., None]
    return q, k


# ----------------------------------------------------------------------------- measurement


def events_time(fn, warmup, repeats, pre=None):
    for _ in range(warmup):
        if pre:
            pre()
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        if pre:
            pre()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return out


def scratch_peak(fn):
    """Peak bytes allocated by fn above what was allocated before it (results freed after)."""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    r = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del r
    return int(peak)


def degree_stats(ptr):
    d = (ptr[..., 1:] - ptr[..., :-1]).reshape(-1).float()
    qs = torch.quantile(d, torch.tensor([0.5, 0.95], device=d.device)).tolist()
    return dict(p50=qs[0], p95=qs[1], max=int(d.max()), mean=float(d.mean()), zero=int((d == 0).sum()))


def diff(a, b):
    a, b = a.float(), b.float()
    err = (a - b).abs().max().item()
    return dict(max_abs=err, rel=err / max(b.abs().max().item(), 1e-6))


def fmt(line):
    return "  ".join(f"{kk}={vv:.3f}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in line.items())


def med(x):
    return sorted(x)[len(x) // 2] if x else float("nan")


# ----------------------------------------------------------------------------- stages


def tk_fine_stages(C, q, k, v, dout, cnt, sel, split, W, N, opts=0):
    """TK fine forward / backward kernels (tk and tk_split arms), same inputs as fine_stages. opts: TK backward
    options. Unsplit arms also time bwd_tk_skipdq_DIAG (opts | 4: dQ not written, invalid gradients) to measure
    the cost of the dQ reduce-adds."""
    B, H, L, D = q.shape
    res = {}
    if split is None or not split.fwd:
        res["fwd_fine"] = events_time(lambda: C.attn_fwd(q, k, v, sel.q2k_ptr, sel.q2k_idx, cnt, True), W, N)
        of, lse = C.attn_fwd(q, k, v, sel.q2k_ptr, sel.q2k_idx, cnt, True)
    else:
        plan = lambda: ops.split_plan_for(C, q, sel.q2k_ptr, sel.q2k_idx, split, True, "fwd", 64 * (D + 1) * 4)
        res["fwd_plan"] = events_time(plan, W, N)
        tasks, merges, info, slots = plan()
        part_o = torch.empty(max(slots, 1), 64, D, device=q.device, dtype=torch.float32)
        part_l = torch.empty(max(slots, 1), 64, device=q.device, dtype=torch.float32)
        res["fwd_tasks"] = events_time(lambda: C.tk_fine_fwd_tasks(q, k, v, sel.q2k_idx, cnt, tasks, info, part_o,
                                                                   part_l), W, N)
        of, lse = C.tk_fine_fwd_tasks(q, k, v, sel.q2k_idx, cnt, tasks, info, part_o, part_l)
        if slots:
            res["fwd_merge"] = events_time(lambda: C.split_merge_fwd(part_o, part_l, merges, info, cnt, of, lse, False),
                                           W, N)
            C.split_merge_fwd(part_o, part_l, merges, info, cnt, of, lse, False)
        res["fwd_fine"] = events_time(lambda: ops.tk_split_fine_fwd(C, q, k, v, sel.q2k_ptr, sel.q2k_idx, cnt, split),
                                      W, N)
        res["fwd_plan_info"] = info.tolist()
        del part_o, part_l
    nbc_pad = (L // 64 + 63) // 64 * 64
    oc0 = torch.zeros(B, H, nbc_pad, D, device=q.device, dtype=q.dtype)
    dout_f, delta, _, _, _ = C.fuse_bwd(dout, of, oc0, None, cnt, 0, True)
    acc = [torch.zeros(q.shape, device=q.device, dtype=torch.float32) for _ in range(3)]
    zero = lambda: [t.zero_() for t in acc]
    if split is None or not split.bwd:
        res["bwd_tk"] = events_time(lambda: C.tk_fine_bwd(q, k, v, dout_f, lse, delta, sel.k2q_ptr, sel.k2q_idx, cnt,
                                                          *acc, opts), W, N, pre=zero)
        res["bwd_tk_skipdq_DIAG"] = events_time(lambda: C.tk_fine_bwd(q, k, v, dout_f, lse, delta, sel.k2q_ptr,
                                                                      sel.k2q_idx, cnt, *acc, opts | 4), W, N, pre=zero)
    else:
        plan = lambda: ops.split_plan_for(C, q, sel.k2q_ptr, sel.k2q_idx, split, True, "bwd_kv", None)
        res["bwd_kv_plan"] = events_time(plan, W, N)
        tasks, _, info, _ = plan()
        res["bwd_tk_tasks"] = events_time(lambda: C.tk_fine_bwd_tasks(q, k, v, dout_f, lse, delta, sel.k2q_idx, cnt,
                                                                      tasks, info, *acc, opts), W, N, pre=zero)
        res["bwd_kv_plan_info"] = info.tolist()
    return res


def fine_stages(C, q, k, v, dout, cnt, sel, split, variant, W, N):
    """Generic fine-branch stage timings (no coarse branch, no fusion)."""
    B, H, L, D = q.shape
    res = {}
    if split is None or not split.fwd:
        res["fwd_fine"] = events_time(lambda: C.attn_fwd(q, k, v, sel.q2k_ptr, sel.q2k_idx, cnt, False), W, N)
        of, lse = C.attn_fwd(q, k, v, sel.q2k_ptr, sel.q2k_idx, cnt, False)
    else:
        nnz = sel.q2k_idx.numel()
        plan = lambda: ops.split_plan_for(C, q, sel.q2k_ptr, sel.q2k_idx, split, False, "fwd", 64 * (D + 1) * 4)
        res["fwd_plan"] = events_time(plan, W, N)
        tasks, merges, info, slots = plan()
        part_o = torch.empty(slots, 64, D, device=q.device, dtype=torch.float32)
        part_l = torch.empty(slots, 64, device=q.device, dtype=torch.float32)
        res["fwd_tasks"] = events_time(lambda: C.attn_fwd_tasks(q, k, v, sel.q2k_idx, cnt, tasks, info, part_o, part_l),
                                       W, N)
        of, lse = C.attn_fwd_tasks(q, k, v, sel.q2k_idx, cnt, tasks, info, part_o, part_l)
        res["fwd_merge"] = events_time(lambda: C.split_merge_fwd(part_o, part_l, merges, info, cnt, of, lse), W, N)
        res["fwd_fine"] = events_time(lambda: ops.split_fine_fwd(C, q, k, v, sel.q2k_ptr, sel.q2k_idx, cnt, split), W, N)
        res["fwd_plan_info"] = info.tolist()
        del part_o, part_l
    nbc_pad = (L // 64 + 63) // 64 * 64
    oc0 = torch.zeros(B, H, nbc_pad, D, device=q.device, dtype=q.dtype)
    _, delta, _, _, _ = C.fuse_bwd(dout, of, oc0, None, cnt, 0, False)
    f32 = dict(device=q.device, dtype=torch.float32)
    dq_acc = torch.zeros(q.shape, **f32) if variant != 2 else None
    zero = (lambda: dq_acc.zero_()) if dq_acc is not None else None
    if split is None or not split.bwd:
        res["bwd_kv"] = events_time(lambda: C.attn_bwd(q, k, v, dout, lse, delta, sel.k2q_ptr, sel.k2q_idx, cnt, dq_acc,
                                                       None, None, False, variant == 1), W, N, pre=zero)
        if variant == 2:
            res["bwd_q"] = events_time(lambda: C.attn_bwd_q(q, k, v, dout, lse, delta, sel.q2k_ptr, sel.q2k_idx, cnt,
                                                            None), W, N)
    else:
        plan = lambda: ops.split_plan_for(C, q, sel.k2q_ptr, sel.k2q_idx, split, False, "bwd_kv", 2 * 64 * D * 4)
        res["bwd_kv_plan"] = events_time(plan, W, N)
        tasks, merges, info, slots = plan()
        pk, pv = torch.empty(slots, 64, D, **f32), torch.empty(slots, 64, D, **f32)
        res["bwd_kv_tasks"] = events_time(
            lambda: C.attn_bwd_tasks(q, k, v, dout, lse, delta, sel.k2q_idx, cnt, tasks, info, dq_acc, None, None, pk, pv,
                                     variant == 1), W, N, pre=zero)
        dk, dv = C.attn_bwd_tasks(q, k, v, dout, lse, delta, sel.k2q_idx, cnt, tasks, info, dq_acc, None, None, pk, pv,
                                  variant == 1)
        res["bwd_kv_merge"] = events_time(lambda: (C.split_merge_sum(pk, merges, info, None, cnt, dk),
                                                   C.split_merge_sum(pv, merges, info, None, cnt, dv)), W, N)
        res["bwd_kv_plan_info"] = info.tolist()
        del pk, pv
        if variant == 2:
            plan = lambda: ops.split_plan_for(C, q, sel.q2k_ptr, sel.q2k_idx, split, False, "bwd_q", 64 * D * 4)
            res["bwd_q_plan"] = events_time(plan, W, N)
            tasks, merges, info, slots = plan()
            pq = torch.empty(slots, 64, D, **f32)
            res["bwd_q_tasks"] = events_time(lambda: C.attn_bwd_q_tasks(q, k, v, dout, lse, delta, sel.q2k_idx, cnt, tasks,
                                                                        info, None, pq), W, N)
            dq = C.attn_bwd_q_tasks(q, k, v, dout, lse, delta, sel.q2k_idx, cnt, tasks, info, None, pq)
            res["bwd_q_merge"] = events_time(lambda: C.split_merge_sum(pq, merges, info, None, cnt, dq), W, N)
            res["bwd_q_plan_info"] = info.tolist()
            del pq
    return res


def full_op(q, k, v, gate, cnt, sel, backend, split, variant, dout, W, N, route_kw=None, tk_bwd_opts=None, R=16):
    ops.set_tk_bwd_opts(tk_bwd_opts)
    try:
        return _full_op(q, k, v, gate, cnt, sel, backend, split, variant, dout, W, N, route_kw, R)
    finally:
        ops.set_tk_bwd_opts(None)


def _full_op(q, k, v, gate, cnt, sel, backend, split, variant, dout, W, N, route_kw, R):
    """Complete operator timings. sel is a frozen Selection (router not run; R still sets the pooled router
    tensors built by pool_fwd), or None with route_kw (R, topk): the PSP router and exact global top-k then run
    inside every forward."""
    leaves = [x.detach().clone().requires_grad_(True) for x in (q, k, v, gate)]
    kw = dict(backend=backend, bwd_variant=variant, split_kv=split)
    if sel is not None:
        kw.update(selection=sel, R=R)
    else:
        kw.update(route_kw)
    res = {}
    with torch.no_grad():
        res["fwd"] = events_time(lambda: ops.vsa2_attention(q, k, v, gate, cnt, **kw), W, N)
        res["fwd_scratch_bytes"] = scratch_peak(lambda: ops.vsa2_attention(q, k, v, gate, cnt, **kw))

    def fb():
        o = ops.vsa2_attention(*leaves, cnt, **kw)
        return torch.autograd.grad(o, leaves, dout)

    res["fwd_bwd"] = events_time(fb, W, N)
    res["fwd_bwd_scratch_bytes"] = scratch_peak(fb)
    bwd_t = []
    for i in range(W + N):
        o = ops.vsa2_attention(*leaves, cnt, **kw)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g = torch.autograd.grad(o, leaves, dout)
        b.record()
        torch.cuda.synchronize()
        if i >= W:
            bwd_t.append(a.elapsed_time(b))
        del o, g
    res["bwd"] = bwd_t
    o = ops.vsa2_attention(*leaves, cnt, **kw)
    grads = torch.autograd.grad(o, leaves, dout)
    return res, (o.detach(), *grads)



def operator_phases(a, C, q, k, v, gate, dout, cnt, sel, backend, split, variant, tk_opts, W, N):
    """Independent callable timings of the complete operator's phases, on intermediates computed exactly as in
    ops._Function (same selection, split flags and TK options). Each callable includes its own allocations and
    launches; the phases were warmed and timed separately, so their sum is not the critical-path time of a full
    call. The acceptance metric stays the complete F / B / F+B timing (full_ms)."""
    use_tk = backend == "tk"
    B, H, L, D = q.shape
    nb = L // 64
    gact = ops.GATE_ACTS["tanh"]
    f32 = dict(device=q.device, dtype=torch.float32)
    res = {}
    res["pool"] = events_time(lambda: C.pool_fwd(q, k, v, cnt, a.R), W, N)
    qr, kr, qc, kc, vc = C.pool_fwd(q, k, v, cnt, a.R)
    res["router_routed_calls_only"] = events_time(
        lambda: C.route(qr, kr, cnt, a.R, nb * a.topk, ops.ROW_NORMS["paper"], 2 << 30), W, N)
    del qr, kr
    res["coarse_meta"] = events_time(lambda: ops._coarse_meta(B, H, nb, qc.shape[2], q.device), W, N)
    c_cnt, c_ptr, c_idx = ops._coarse_meta(B, H, nb, qc.shape[2], q.device)
    res["coarse_fwd"] = events_time(lambda: C.attn_fwd(qc, kc, vc, c_ptr, c_idx, c_cnt, False), W, N)
    oc, lse_c = C.attn_fwd(qc, kc, vc, c_ptr, c_idx, c_cnt, False)
    qp, qi, kp, ki = sel.q2k_ptr, sel.q2k_idx, sel.k2q_ptr, sel.k2q_idx
    res["fine_fwd"] = events_time(lambda: ops._fine_fwd(C, q, k, v, qp, qi, cnt, use_tk, split), W, N)
    of, lse_f = ops._fine_fwd(C, q, k, v, qp, qi, cnt, use_tk, split)
    res["fuse_fwd"] = events_time(lambda: C.fuse_fwd(of, oc, gate, cnt, gact), W, N)
    res["fuse_bwd"] = events_time(lambda: C.fuse_bwd(dout, of, oc, gate, cnt, gact, use_tk), W, N)
    dout_f, delta_f, dout_c, delta_c, _ = C.fuse_bwd(dout, of, oc, gate, cnt, gact, use_tk)
    if not use_tk:
        dout_f = dout
    del of
    cvar = None if use_tk else variant  # ops: ctx.bwd_variant is None on the TK path

    def coarse_bwd():
        dqc = torch.zeros(qc.shape, **f32)
        dkc, dvc = C.attn_bwd(qc, kc, vc, dout_c, lse_c, delta_c, c_ptr, c_idx, c_cnt, dqc, None, None, True, cvar == 1)
        return dqc, dkc, dvc

    res["coarse_bwd"] = events_time(coarse_bwd, W, N)
    dqc, dkc, dvc = coarse_bwd()
    if use_tk:
        def fine_bwd():
            acc = [torch.zeros(q.shape, **f32) for _ in range(3)]
            ops.tk_fine_bwd(C, q, k, v, dout_f, lse_f, delta_f, kp, ki, cnt, *acc, split, tk_opts)
            return acc

        res["fine_bwd_incl_acc_zero"] = events_time(fine_bwd, W, N)
        acc = fine_bwd()
        res["finalize_dq_dk_dv"] = events_time(lambda: (C.finalize(acc[1], dkc, cnt, k), C.finalize(acc[2], dvc, cnt, v),
                                                        C.finalize(acc[0], dqc, cnt, q)), W, N)
        del acc
    else:
        res["fine_bwd_incl_finalize"] = events_time(
            lambda: ops._generic_fine_bwd(C, variant, q, k, v, dout_f, lse_f, delta_f, qp, qi, kp, ki, cnt, dqc, dkc,
                                          dvc, split), W, N)
    res["R"] = a.R
    res["note"] = ("independent warmed callables incl. their allocations; sum != full-call critical path; "
                   "router_routed_calls_only is not part of frozen-selection calls")
    return res


NAMES = ["o", "dq", "dk", "dv", "dgate"]


def compare_refs(row, arm, outs, refs):
    """Differences against the unsplit arm with the same kernels/options and against plain tk."""
    same = arm["ref_unsplit"]
    if arm["schedule"] == "split" and same in refs:
        row["vs_unsplit_same_kernels"] = {n: diff(x, y) for n, x, y in zip(NAMES, outs, refs[same])}
    if arm["arm"] != "tk" and "tk" in refs:
        row["vs_tk"] = {n: diff(x, y) for n, x, y in zip(NAMES, outs, refs["tk"])}
    if arm["schedule"] == "unsplit":
        refs[arm["arm"]] = outs


def make_split(a, arm):
    if arm["chunk"] is None:
        return None
    if arm.get("split_mode") == "auto":
        return ops.SplitKV.auto(chunk=arm["chunk"], fwd=arm["split_fwd"], bwd=arm["split_bwd"],
                                max_scratch_bytes=int(a.auto_scratch_mib * 2**20), policy=a.policy)
    return ops.SplitKV(chunk=arm["chunk"], fwd=arm["split_fwd"], bwd=arm["split_bwd"],
                       max_scratch_bytes=int(a.max_scratch_gib * 2**30))


SPLIT_DIRS = {"both": ("", True, True), "fwd": ("_fwdonly", True, False), "bwd": ("_bwdonly", False, True)}


def measure_frozen(a, C, arms, meta, q, k, v, gate, dout, cnt, sel, W, N, emit, summary, oracle=None):
    """Every arm on one frozen Selection (router excluded): fine stages, complete operator, differences.
    oracle: optional (rows, o_ref) from sampled_oracle; each arm's O is checked on those rows."""
    meta["generic_auto_bwd_variant"] = list(ops.resolve_bwd_variant(q))
    meta["R"] = a.R  # pool_fwd group size of the frozen full calls and of phases_ms (selection itself is frozen)
    refs = {}  # arm name -> (o, dq, dk, dv, dgate) of every unsplit arm
    for arm in arms:
        split = make_split(a, arm)
        variant = arm["bwd_variant"]
        try:
            row = dict(kind="result", **meta, **arm)
            if not a.no_stages:
                row["stages_ms"] = (tk_fine_stages(C, q, k, v, dout, cnt, sel, split, W, N, arm["tk_bwd_opts"])
                                    if arm["backend"] == "tk"
                                    else fine_stages(C, q, k, v, dout, cnt, sel, split, variant, W, N))
            if split is not None:  # actual plans, read after timing (diagnostic sync, not in the timed calls)
                row["plan_stats"] = ops.describe_split(q, sel, split, backend=arm["backend"],
                                                       bwd_variant=variant or 0)
                if split.policy is not None:
                    row["policy"] = dict(vars(split.policy))
            if a.phases:
                row["phases_ms"] = operator_phases(a, C, q, k, v, gate, dout, cnt, sel, arm["backend"], split, variant,
                                                   arm["tk_bwd_opts"] or 0, W, N)
            if not a.no_full:
                full, outs = full_op(q, k, v, gate, cnt, sel, arm["backend"], split, variant, dout, W, N,
                                     tk_bwd_opts=arm["tk_bwd_opts"], R=a.R)
                row["full_ms"] = full
                if oracle is not None:
                    row["vs_fp32_oracle_sampled_o"] = oracle_error(outs[0], *oracle)
                compare_refs(row, arm, outs, refs)
                del outs
            emit(row)
            st, f = row.get("stages_ms", {}), row.get("full_ms", {})
            fine_bwd = sum(med(st[x]) for x in ("bwd_tk", "bwd_tk_tasks", "bwd_kv", "bwd_q", "bwd_kv_plan",
                                                "bwd_kv_tasks", "bwd_kv_merge", "bwd_q_plan", "bwd_q_tasks",
                                                "bwd_q_merge") if x in st)
            line = dict(tokens=meta["tokens"], case=meta["case"], sel="frozen", arm=arm["arm"],
                        fwd_fine=med(st.get("fwd_fine", [])), bwd_fine_sum=fine_bwd if st else float("nan"),
                        fwd=med(f.get("fwd", [])), bwd=med(f.get("bwd", [])), fb=med(f.get("fwd_bwd", [])),
                        scratch_gib=f.get("fwd_bwd_scratch_bytes", 0) / 2**30)
            if "bwd_tk_skipdq_DIAG" in st:
                line["bwd_skipdq_DIAG"] = med(st["bwd_tk_skipdq_DIAG"])
            if "vs_fp32_oracle_sampled_o" in row:
                line["oracle_rel"] = row["vs_fp32_oracle_sampled_o"]["rel"]
                line["oracle_pass"] = row["vs_fp32_oracle_sampled_o"]["pass"]
            ps = row.get("plan_stats")
            if ps:
                line.update({f"{d}_split_rows": ps[d]["info"]["split_rows"] for d in ps})
            ph = row.get("phases_ms")
            if ph:
                line.update({f"ph_{n}": med(ph[n]) for n in ("coarse_fwd", "coarse_bwd", "fuse_fwd", "fuse_bwd")})
            summary.append(line)
            print("  " + fmt(line), flush=True)
        except Exception as e:
            emit(dict(kind="error", stage="measure", error=repr(e), **meta, **arm))
            print(f"  {arm['arm']} failed: {e!r}", flush=True)
            torch.cuda.empty_cache()
    del refs


def measure_routed(a, C, arms, meta, q, k, v, gate, dout, cnt, sel0, W, N, emit, summary):
    """Every arm as a complete VSA2 call (PSP router + global top-k inside every forward)."""
    refs = {}
    route_kw = dict(R=a.R, topk=a.topk)
    for arm in arms:
        split = make_split(a, arm)
        variant = arm["bwd_variant"]
        try:
            row = dict(kind="result", **meta, **arm)
            with torch.no_grad():
                sel = ops.vsa2_attention(q, k, v, gate, cnt, backend=arm["backend"], split_kv=split,
                                         return_selection=True, **route_kw)[1]
            row["selection_identical_to_first_arm"] = bool(torch.equal(sel.q2k_ptr, sel0.q2k_ptr)
                                                           and torch.equal(sel.q2k_idx, sel0.q2k_idx))
            del sel
            full, outs = full_op(q, k, v, gate, cnt, None, arm["backend"], split, variant, dout, W, N, route_kw,
                                 tk_bwd_opts=arm["tk_bwd_opts"])
            row["full_ms"] = full
            if split is not None:  # plans on the (identical) routed selection, read after timing
                row["plan_stats"] = ops.describe_split(q, sel0, split, backend=arm["backend"], bwd_variant=variant or 0)
            compare_refs(row, arm, outs, refs)
            del outs
            emit(row)
            line = dict(tokens=meta["tokens"], case=meta["case"], sel="routed", arm=arm["arm"], fwd=med(full["fwd"]),
                        bwd=med(full["bwd"]), fb=med(full["fwd_bwd"]),
                        scratch_gib=full["fwd_bwd_scratch_bytes"] / 2**30)
            if "plan_stats" in row:
                line.update({f"{d}_split_rows": row["plan_stats"][d]["info"]["split_rows"] for d in row["plan_stats"]})
            summary.append(line)
            print("  " + fmt(line), flush=True)
        except Exception as e:
            emit(dict(kind="error", stage="routed_measure", error=repr(e), **meta, **arm))
            print(f"  routed {arm['arm']} failed: {e!r}", flush=True)
            torch.cuda.empty_cache()
    del refs


# ----------------------------------------------------------------------------- routed full call


def routed(a, C, arms, tokens, dev, W, N, emit, summary, shape_role):
    """Complete VSA2 (router inside every forward) per arm on seeded random "router" inputs."""
    nb = -(-tokens // 64)
    L = nb * 64
    cnt = torch.full((nb,), 64, dtype=torch.int32, device=dev)
    cnt[-1] = tokens - 64 * (nb - 1)
    seed = a.seed * 1000003 + tokens * 31 + 999
    meta = dict(tokens=tokens, nb=nb, L=L, B=a.batch, H=a.heads, D=a.dim, K=a.topk, R=a.R, case="router",
                seed=seed, shape_role=shape_role, selection_mode="routed", includes_router=True, selection_source="psp_router_each_forward")
    try:
        qr, kr = router_inputs("router", a.batch, a.heads, L, a.dim, seed, dev)
        pad = (torch.arange(L, device=dev) % 64 >= cnt.repeat_interleave(64))
        q, k = qr.to(torch.bfloat16), kr.to(torch.bfloat16)
        del qr, kr
        g = torch.Generator(device=dev).manual_seed(seed)
        v = torch.randn(a.batch, a.heads, L, a.dim, generator=g, device=dev).to(torch.bfloat16)
        gate = torch.randn(a.batch, a.heads, L, generator=g, device=dev).to(torch.bfloat16)
        dout = torch.randn(a.batch, a.heads, L, a.dim, generator=g, device=dev).to(torch.bfloat16)
        for t in (q, k, v, dout):
            t[..., pad, :] = 0
        with torch.no_grad():
            sel0 = ops.vsa2_attention(q, k, v, gate, cnt, R=a.R, topk=a.topk, backend="generic",
                                      return_selection=True)[1]
        meta.update(nnz=sel0.q2k_idx.numel(), row_deg=degree_stats(sel0.q2k_ptr), col_deg=degree_stats(sel0.k2q_ptr),
                    generic_auto_bwd_variant=list(ops.resolve_bwd_variant(q)))
        emit(dict(kind="case", **meta))
        print(f"\n== routed tokens={tokens} NB={nb} nnz={meta['nnz']} row={meta['row_deg']} col={meta['col_deg']}",
              flush=True)
    except Exception as e:
        emit(dict(kind="error", stage="routed_setup", error=repr(e), **meta))
        print(f"routed setup failed: {e!r}", flush=True)
        torch.cuda.empty_cache()
        return
    measure_routed(a, C, arms, meta, q, k, v, gate, dout, cnt, sel0, W, N, emit, summary)
    del q, k, v, gate, dout
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- real Q/K/V captures


def load_real_qkv(path, heads, device):
    """Captured attention inputs -> (q, k, v) bf16 [1, H, N, D] in captured token order, plus metadata.

    Fixture convention (EchoSparse real-qkv captures): torch.save of a dict whose "qkv" is a sequence of three
    [1, N, H, D] tensors (the CUDA consumer used x[0] = [N, H, D]). Loaded with weights_only=True, mmap=True; only
    the selected heads are copied to the GPU."""
    fx = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=True, mmap=True)
    qkv = fx["qkv"] if isinstance(fx, dict) else fx
    if len(qkv) != 3:
        raise ValueError(f"{path}: expected 3 tensors in 'qkv', got {len(qkv)}")
    shapes = [tuple(x.shape) for x in qkv]
    if any(len(sh) != 4 or sh[0] != 1 or sh != shapes[0] for sh in shapes):
        raise ValueError(f"{path}: expected three equal [1, N, H, D] tensors, got {shapes}")
    _, n, h_all, d = shapes[0]
    hs = list(range(h_all)) if heads is None else list(range(min(heads, h_all)))
    out = []
    for x in qkv:
        x = x[0][:, hs[0]:hs[-1] + 1]                                 # [N, H, D], still on CPU (mmap)
        out.append(x.to(device).to(torch.bfloat16).permute(1, 0, 2).unsqueeze(0).contiguous())  # [1, H, N, D]
    meta = dict(path=path, file_bytes=os.path.getsize(os.path.expanduser(path)), captured_shape=list(shapes[0]),
                captured_layout="[1, N, H, D] -> transposed to [1, H, N, D]", captured_dtype=str(qkv[0].dtype),
                heads_used=len(hs), heads_total=h_all, N=n, D=d)
    if isinstance(fx, dict):  # small descriptive entries only (never tensors)
        meta["fixture_keys"] = sorted(map(str, fx.keys()))
        for key, val in fx.items():
            if key != "qkv" and isinstance(val, (int, float, str, bool)) and len(str(val)) < 200:
                meta[f"fixture_{key}"] = val
    del fx, qkv
    return out, meta


def per_head_degrees(sel, B, H):
    rows = []
    for b in range(B):
        for h in range(H):
            qp, kp = sel.q2k_ptr[b:b + 1, h:h + 1], sel.k2q_ptr[b:b + 1, h:h + 1]
            rows.append(dict(b=b, h=h, nnz=int(qp[..., -1] - qp[..., 0]), row=degree_stats(qp), col=degree_stats(kp)))
    return rows


def sampled_oracle(q, k, v, gate, cnt, sel, per_head, seed):
    """Independent fp32 complete-operator O on sampled query blocks (no LxL matrix): per (b, h) the longest row,
    the shortest nonempty row, an empty row if any, the final query block and random rows. Coarse pools are rounded to bf16 as the
    kernels' pooling is. Returns (rows [(b, h, i)], O_ref [n, 64, D] with padded rows 0)."""
    B, H, L, D = q.shape
    nb = L // 64
    scale = 1.0 / math.sqrt(D)
    cntf = cnt.float()
    valid = (torch.arange(L, device=q.device) % 64 < cnt.repeat_interleave(64))
    mean = lambda x: ((x.float() * valid[:, None]).reshape(B, H, nb, 64, D).sum(3) / cntf[:, None]).to(
        torch.bfloat16).float()
    qc, kc, vc = mean(q), mean(k), mean(v)
    oc = torch.softmax(qc @ kc.transpose(-1, -2) * scale, -1) @ vc  # [B, H, nb, D]
    ptr = sel.q2k_ptr.cpu()
    g = torch.Generator().manual_seed(seed)
    rows, refs = [], []
    for b in range(B):
        for h in range(H):
            deg = (ptr[b, h, 1:] - ptr[b, h, :-1])
            pick = {int(deg.argmax()), nb - 1}  # longest row and the final (often partial) query block
            nz = (deg > 0).nonzero().squeeze(1)
            if nz.numel():
                pick.add(int(nz[deg[nz].argmin()]))
            z = (deg == 0).nonzero().squeeze(1)
            if z.numel():
                pick.add(int(z[0]))
            pick.update(torch.randint(0, nb, (max(per_head - len(pick), 0),), generator=g).tolist())
            for i in sorted(pick):
                beg, end = int(ptr[b, h, i]), int(ptr[b, h, i + 1])
                qi = q[b, h, i * 64:(i + 1) * 64].float()
                if end > beg:
                    blks = sel.q2k_idx[beg:end].long()
                    tok = (blks[:, None] * 64 + torch.arange(64, device=q.device)).reshape(-1)
                    s_ = qi @ k[b, h, tok].float().t() * scale
                    s_ = s_.masked_fill(~valid[tok][None, :], float("-inf"))
                    of = torch.softmax(s_, -1) @ v[b, h, tok].float()
                else:
                    of = torch.zeros(64, D, device=q.device)
                o = of + torch.tanh(gate[b, h, i * 64:(i + 1) * 64].float())[:, None] * oc[b, h, i]
                o[int(cnt[i]):] = 0
                rows.append((b, h, i))
                refs.append(o)
    return rows, torch.stack(refs)


ORACLE_RTOL, ORACLE_ATOL = 2e-2, 1e-5  # the test suite's full-operator O tolerance: err <= rtol * max|ref| + atol


def oracle_error(o, rows, o_ref):
    """Sampled complete-operator O check (forward only; not a gradient oracle). pass = finite and within tolerance."""
    got = torch.stack([o[b, h, i * 64:(i + 1) * 64].float() for b, h, i in rows])
    got_finite, ref_finite = bool(torch.isfinite(got).all()), bool(torch.isfinite(o_ref).all())
    err = (got - o_ref).abs().max().item() if got.numel() else 0.0
    scale = o_ref.abs().max().item() if o_ref.numel() else 0.0
    rel = err / max(scale, 1e-6)
    finite = got_finite and ref_finite and all(math.isfinite(x) for x in (err, scale, rel))
    ok = finite and err <= ORACLE_RTOL * scale + ORACLE_ATOL
    return dict(max_abs=err, rel=rel, rows=len(rows), finite=finite, got_finite=got_finite, ref_finite=ref_finite,
                rtol=ORACLE_RTOL, atol=ORACLE_ATOL, **{"pass": ok})


def real_run(a, C, arms, path, label, grid, W, N, emit, summary):
    """Real captured Q/K/V (no model): fresh VSA2 PSP router + exact global top-k selection, synthetic seeded
    gate and dO, frozen-selection and routed measurements of every arm on identical inputs."""
    from echosparse.vsa import layout as vl
    dev = torch.device("cuda")
    meta = dict(case=f"real:{label}", shape_role="real_capture", K=a.topk, R=a.R, gate_source="synthetic",
                dout_source="synthetic")
    try:
        (qr, kr, vr), cap = load_real_qkv(path, a.real_heads, dev)
        n, H, D = cap["N"], cap["heads_used"], cap["D"]
        if D not in (64, 128):
            raise ValueError(f"head dim {D} unsupported (64 or 128)")
        if grid:
            T_, H_, W_ = grid
            if T_ * H_ * W_ != n:
                raise ValueError(f"--real-grid {grid} has {T_ * H_ * W_} tokens, capture has {n}")
            lay = vl.build_layout(tuple(grid), R=a.R, device=dev)
            lay_desc = f"build_layout({T_}x{H_}x{W_}, tile 4x4x4, R={a.R}); captured order assumed raster (T, H, W)"
        else:
            lay = vl.sequence_layout(n, device=dev)
            lay_desc = "sequence_layout: 64 consecutive captured tokens per block (captured order)"
        q, k, v = (vl.to_blocked(x, lay).contiguous() for x in (qr, kr, vr))
        del qr, kr, vr
        cnt = lay.blk_cnt
        seed = a.seed * 1000003 + n * 31 + 4242
        g = torch.Generator(device=dev).manual_seed(seed)
        gate = vl.to_blocked(torch.randn(1, H, n, generator=g, device=dev).to(torch.bfloat16), lay).contiguous()
        dout = vl.to_blocked(torch.randn(1, H, n, D, generator=g, device=dev).to(torch.bfloat16), lay).contiguous()
        L = q.shape[2]
        nb = L // 64
        meta.update(tokens=n, nb=nb, L=L, B=1, H=H, D=D, capture=cap, layout=lay_desc,
                    synthetic_seed=seed, gate_desc="randn(seed) per real token, bf16, pre-tanh",
                    dout_desc="randn(seed) per real token, bf16")
        torch.cuda.synchronize()
        t0 = time.time()
        sel = ops.route(q, k, cnt, R=a.R, topk=a.topk)
        torch.cuda.synchronize()
        budget = nb * a.topk
        meta.update(nnz=sel.q2k_idx.numel(), nnz_per_head_budget=min(budget, nb * nb),
                    route_setup_s=time.time() - t0, row_deg=degree_stats(sel.q2k_ptr), col_deg=degree_stats(sel.k2q_ptr),
                    per_head=per_head_degrees(sel, 1, H))
        if any(r["nnz"] != min(budget, nb * nb) for r in meta["per_head"]):
            raise RuntimeError("router NNZ per head differs from the exact global budget")
        emit(dict(kind="case", selection_mode="fresh_psp_router_on_capture", **meta))
        print(f"\n== real {label}: N={n} H={H} D={D} NB={nb} nnz={meta['nnz']} row={meta['row_deg']} "
              f"col={meta['col_deg']}\n   layout: {lay_desc}", flush=True)
        for r in meta["per_head"]:
            print(f"   h{r['h']:>2} nnz={r['nnz']} row p50/p95/max/zero={r['row']['p50']:.0f}/{r['row']['p95']:.0f}/"
                  f"{r['row']['max']}/{r['row']['zero']} col={r['col']['p50']:.0f}/{r['col']['p95']:.0f}/"
                  f"{r['col']['max']}/{r['col']['zero']}", flush=True)
        oracle = sampled_oracle(q, k, v, gate, cnt, sel, a.real_check_rows, seed) if a.real_check_rows > 0 else None
    except Exception as e:
        emit(dict(kind="error", stage="real_setup", error=repr(e), path=path, **meta))
        print(f"real setup failed ({path}): {e!r}", flush=True)
        torch.cuda.empty_cache()
        return
    if "frozen" in a.real_modes:
        fm = dict(meta, selection_mode="frozen", includes_router=False,
                  selection_source="fresh VSA2 PSP router + exact global top-k on the capture, computed once")
        measure_frozen(a, C, arms, fm, q, k, v, gate, dout, cnt, sel, W, N, emit, summary, oracle=oracle)
    if "routed" in a.real_modes:
        rm = dict(meta, selection_mode="routed", includes_router=True,
                  selection_source="VSA2 PSP router + exact global top-k inside every forward")
        measure_routed(a, C, arms, rm, q, k, v, gate, dout, cnt, sel, W, N, emit, summary)
    del q, k, v, gate, dout, sel, oracle
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- driver


def env_info():
    C = ops.ext()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except OSError:
        commit = ""
    return dict(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                torch=torch.__version__, cuda=torch.version.cuda, commit=commit, host=platform.node(),
                limits=C.device_limits(), fwd_mt=C.get_fwd_variant(), argv=sys.argv)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, nargs="+", default=[32768, 76800, 122880, 220000])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=20)
    ap.add_argument("--dim", type=int, default=128, choices=[64, 128])
    ap.add_argument("--topk", type=int, default=64, help="K: NNZ per (b,h) is NB*K")
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--cases", nargs="+", default=["all"], help='"none" runs only the routed measurement')
    ap.add_argument("--chunks", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--max-scratch-gib", type=float, default=4.0,
                    help="SplitKV.max_scratch_bytes per stage (library default 1 GiB); the partial buffer is "
                         "min(cap, 2*ceil(nnz/chunk)) slots, allocated whether or not rows split (no host readback)")
    ap.add_argument("--tk-bwd-opts", type=int, nargs="+", default=[0, 1, 3], choices=[0, 1, 2, 3],
                    help="TK backward options per tk arm: 0 original, 1 dQ read-wait (_rw), 2 rotated k2q walk (_rot), "
                         "3 both (_rw_rot)")
    ap.add_argument("--split-dirs", nargs="+", default=["both"], choices=list(SPLIT_DIRS),
                    help="split directions per chunk: both (SplitKV fwd+bwd), fwd (_fwdonly: fwd=True, bwd=False), "
                         "bwd (_bwdonly: fwd=False, bwd=True); compared with the unsplit arm of the same options")
    ap.add_argument("--auto-chunks", type=int, nargs="*", default=[],
                    help="add tk_auto_c<chunk> arms: SplitKV.auto (GPU SplitPolicy picks the lists per direction); "
                         "rows record the actual per-direction / per-head plan (plan_stats)")
    ap.add_argument("--auto-scratch-mib", type=float, default=64.0, help="forward partial cap of the auto arms")
    ap.add_argument("--policy", default="", help="SplitPolicy overrides, e.g. rel_mean=4,tail_frac=0.05,bwd_waves=8")
    ap.add_argument("--phases", action="store_true",
                    help="frozen arms: also time the operator phases (pool, coarse fwd/bwd, fine fwd/bwd, fusion "
                         "fwd/bwd, finalize) as independent callables on ops._Function-matching intermediates")
    ap.add_argument("--bwd-variants", type=int, nargs="+", default=[2, 0],
                    help="generic backward variants, each run unsplit and split")
    ap.add_argument("--routed-tokens", type=int, nargs="*", default=[122880],
                    help="lengths for the routed full-call measurement (router inside each forward); empty to skip")
    ap.add_argument("--arms", nargs="+", default=["*"],
                    help="glob filters over arm names, e.g. tk tk_split_c128 generic_v0 split_v0_c128. Differences "
                         "are only reported when the reference arm (tk / generic_vN) is also selected")
    ap.add_argument("--shape-role", choices=["auto", "reference", "diagnostic"], default="auto",
                    help="auto: reference for B1/H20/D128/K64, otherwise diagnostic")
    ap.add_argument("--no-tk", action="store_true", help="skip the tk arm (it is skipped anyway without TK/SM90)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-full", action="store_true", help="fine-stage timings only")
    ap.add_argument("--no-stages", action="store_true", help="complete-operator timings only")
    ap.add_argument("--real-qkv", nargs="*", default=[],
                    help="captured Q/K/V fixtures (dict 'qkv' = 3 x [1, N, H, D]); runs every arm on them with a "
                         "fresh VSA2 selection. Use with --cases none --routed-tokens to skip synthetic runs")
    ap.add_argument("--real-label", nargs="*", default=[], help="one label per --real-qkv path")
    ap.add_argument("--real-grid", nargs="*", default=[],
                    help="one TxHxW per path (or '-' for the 1-D sequence layout, the default) if the captured "
                         "order is raster (T, H, W); e.g. 21x30x52 for Wan 480p, 21x45x80 for 720p")
    ap.add_argument("--real-heads", type=int, default=None, help="use only the first N captured heads")
    ap.add_argument("--real-inspect", action="store_true",
                    help="CPU only: print each --real-qkv fixture's keys, shapes and dtypes (mmap) and exit")
    ap.add_argument("--real-modes", nargs="+", default=["frozen", "routed"], choices=["frozen", "routed"])
    ap.add_argument("--real-check-rows", type=int, default=4,
                    help="query blocks per head checked against an independent fp32 O (0 disables)")
    ap.add_argument("--out", default="vsa2_splitkv.jsonl")
    a = ap.parse_args()
    a.policy = ops.SplitPolicy(**{kk: (int(vv) if kk in ("min_chunks", "max_task_waves", "resident_ctas") else float(vv))
                                  for kk, vv in (x.split("=") for x in a.policy.split(",") if x)})
    if a.real_inspect:
        for path in a.real_qkv:
            fx = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=True, mmap=True)
            items = fx.items() if isinstance(fx, dict) else [("<root>", fx)]
            print(f"{path} ({os.path.getsize(os.path.expanduser(path))} bytes)")
            for key, val in items:
                if isinstance(val, torch.Tensor):
                    print(f"  {key}: tensor {tuple(val.shape)} {val.dtype}")
                elif isinstance(val, (list, tuple)):
                    print(f"  {key}: {type(val).__name__}[{len(val)}] " + ", ".join(
                        f"{tuple(x.shape)} {x.dtype}" if isinstance(x, torch.Tensor) else type(x).__name__ for x in val))
                else:
                    print(f"  {key}: {type(val).__name__} {str(val)[:120]}")
            del fx
        return
    cases = CASES if a.cases == ["all"] else [c for c in a.cases if c != "none"]
    dev = torch.device("cuda")
    C = ops.ext()
    W, N = a.warmup, a.repeats
    out = open(a.out, "a")

    def emit(row):
        out.write(json.dumps(row) + "\n")
        out.flush()

    tk_ok = (not a.no_tk) and C.has_tk() and torch.cuda.get_device_capability() == (9, 0)
    arms = []
    if tk_ok:
        sfx = {0: "", 1: "_rw", 2: "_rot", 3: "_rw_rot"}
        for opts in a.tk_bwd_opts:
            for chunk, (dsfx, sf, sb) in [(None, ("", False, False))] + [(c, SPLIT_DIRS[d]) for c in a.chunks
                                                                          for d in a.split_dirs]:
                arms.append(dict(arm=("tk" if chunk is None else f"tk_split_c{chunk}{dsfx}") + sfx[opts], backend="tk",
                                 fwd_kernel="tk", bwd_algorithm="tk_kv_owner+reduce_add_dq_dk_dv", bwd_variant=None,
                                 tk_bwd_opts=opts, tk_dq_wait="read" if opts & 1 else "complete",
                                 tk_k2q_walk="rotated" if opts & 2 else "ascending",
                                 schedule="unsplit" if chunk is None else "split", chunk=chunk,
                                 split_fwd=sf, split_bwd=sb, split_mode="manual" if chunk else None,
                                 ref_unsplit="tk" + sfx[opts]))
            for chunk in a.auto_chunks:
                arms.append(dict(arm=f"tk_auto_c{chunk}" + sfx[opts], backend="tk", fwd_kernel="tk",
                                 bwd_algorithm="tk_kv_owner+reduce_add_dq_dk_dv", bwd_variant=None, tk_bwd_opts=opts,
                                 tk_dq_wait="read" if opts & 1 else "complete",
                                 tk_k2q_walk="rotated" if opts & 2 else "ascending", schedule="split", chunk=chunk,
                                 split_fwd=True, split_bwd=True, split_mode="auto", ref_unsplit="tk" + sfx[opts]))
    bwd_alg = {0: "kv_owner+atomic_dq_f2", 1: "kv_owner+atomic_dq_word", 2: "kv_owner+q_owner"}
    for variant in a.bwd_variants:
        for chunk, (dsfx, sf, sb) in [(None, ("", False, False))] + [(c, SPLIT_DIRS[d]) for c in a.chunks
                                                                      for d in a.split_dirs]:
            arms.append(dict(arm=f"generic_v{variant}" if chunk is None else f"split_v{variant}_c{chunk}{dsfx}",
                             backend="generic", fwd_kernel="generic", bwd_algorithm=bwd_alg[variant],
                             bwd_variant=variant, tk_bwd_opts=None, schedule="unsplit" if chunk is None else "split",
                             chunk=chunk, split_fwd=sf, split_bwd=sb, ref_unsplit=f"generic_v{variant}"))
    arms = [x for x in arms if any(fnmatch.fnmatchcase(x["arm"], pat) for pat in a.arms)]
    shape = dict(B=a.batch, H=a.heads, D=a.dim, K=a.topk)
    shape_role = a.shape_role if a.shape_role != "auto" else ("reference" if shape == REFERENCE_SHAPE else "diagnostic")
    emit(dict(kind="env", time=time.time(), tk_arm=tk_ok, arms=[x["arm"] for x in arms], shape_role=shape_role,
              **env_info()))
    summary = []
    for tokens in a.tokens if cases else []:
        nb = -(-tokens // 64)
        L = nb * 64
        cnt = torch.full((nb,), 64, dtype=torch.int32, device=dev)
        cnt[-1] = tokens - 64 * (nb - 1)
        data_seed = a.seed * 1000003 + tokens * 31
        pad = (torch.arange(L, device=dev) % 64 >= cnt.repeat_interleave(64))
        g = torch.Generator(device=dev).manual_seed(data_seed)
        mk = lambda *shp: torch.randn(*shp, generator=g, device=dev).to(torch.bfloat16)
        base = dict(q=mk(a.batch, a.heads, L, a.dim), k=mk(a.batch, a.heads, L, a.dim), v=mk(a.batch, a.heads, L, a.dim),
                    gate=mk(a.batch, a.heads, L), dout=mk(a.batch, a.heads, L, a.dim))
        for n in ("q", "k", "v", "dout"):
            base[n][..., pad, :] = 0
        for case in cases:
            mask_seed = data_seed + 7919 * (CASES.index(case) + 1)
            meta = dict(tokens=tokens, nb=nb, L=L, B=a.batch, H=a.heads, D=a.dim, K=a.topk, case=case,
                        shape_role=shape_role, data_seed=data_seed, mask_seed=mask_seed,
                        heavy_row_placement=PLACEMENT.get(case, "none"),
                        selection_mode="frozen", includes_router=False,
                        selection_source="psp_router_at_setup" if case.startswith("router") else "synthetic",
                        qk_source=f"router_inputs({case}, mask_seed)" if case.startswith("router") else "shared(data_seed)")
            try:
                v, gate, dout = base["v"], base["gate"], base["dout"]
                if case.startswith("router"):
                    qr, kr = router_inputs(case, a.batch, a.heads, L, a.dim, mask_seed, dev)
                    q, k = qr.to(torch.bfloat16), kr.to(torch.bfloat16)
                    del qr, kr
                    q[..., pad, :] = 0
                    k[..., pad, :] = 0
                    sel = ops.route(q, k, cnt, R=a.R, topk=a.topk)
                else:
                    q, k = base["q"], base["k"]
                    sel = synthetic_selection(case, a.batch, a.heads, nb, a.topk, mask_seed, dev)
                meta.update(nnz=sel.q2k_idx.numel(), row_deg=degree_stats(sel.q2k_ptr),
                            col_deg=degree_stats(sel.k2q_ptr))
                emit(dict(kind="case", **meta))
                print(f"\n== frozen tokens={tokens} NB={nb} case={case} role={shape_role} nnz={meta['nnz']} "
                      f"row={meta['row_deg']} col={meta['col_deg']}", flush=True)
            except Exception as e:  # keep going; failures are data
                emit(dict(kind="error", stage="setup", error=repr(e), **meta))
                print(f"setup failed: {e!r}", flush=True)
                torch.cuda.empty_cache()
                continue
            measure_frozen(a, C, arms, meta, q, k, v, gate, dout, cnt, sel, W, N, emit, summary)
            del q, k, v, gate, dout, sel
            torch.cuda.empty_cache()
        del base
        torch.cuda.empty_cache()
    for tokens in a.routed_tokens:
        routed(a, C, arms, tokens, dev, W, N, emit, summary, shape_role)
    for idx, path in enumerate(a.real_qkv):
        label = a.real_label[idx] if idx < len(a.real_label) else os.path.basename(os.path.dirname(os.path.dirname(path)))
        spec = a.real_grid[idx] if idx < len(a.real_grid) else "-"
        grid = None if spec in ("-", "") else tuple(int(x) for x in spec.lower().split("x"))
        real_run(a, C, arms, path, label, grid, W, N, emit, summary)
    out.close()
    print("\nmedian ms. fwd_fine: fine forward (split: plan+tasks+merge); bwd_fine_sum: sum of fine backward stage "
          "medians (kernels, plans, merges; excludes dq_acc zeroing); fwd/bwd/fb: complete operator")
    for mode, title in (("frozen", "FROZEN selection (router excluded; scheduling diagnostic)"),
                        ("routed", "ROUTED full calls (PSP router + global top-k inside every forward)")):
        rows = [line for line in summary if line["sel"] == mode]
        if rows:
            print(f"\n-- {title}; synthetic shape_role={shape_role}, case real:* = real_capture")
            for line in rows:
                print(fmt(line))


if __name__ == "__main__":
    main()
