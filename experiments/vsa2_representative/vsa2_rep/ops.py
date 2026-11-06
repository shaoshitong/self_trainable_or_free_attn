"""VSA2 representative-block attention: autograd wiring around the native CUDA kernels.

VSA2 routes every query block to the top-k KV blocks of its PSP router and drops the rest. Here the
dropped blocks are kept at reduced resolution: the most important ``frac_full`` of the (query block,
kv block) pairs keep all 64 tokens, the next ``frac_mid`` keep ``mid_reps`` representatives per block,
and every remaining pair keeps one -- the block mean. A representative standing for c tokens enters
the single online softmax as one column with ``+ log2(c)`` on its logit, so its mass is counted in the
numerator and the denominator alike (see csrc/rep/rep_kernels.h).

Routing is VSA2's own: the same PSP router scores and the same exact global top-k, run at two budgets
over one score matrix. No clustering, no token reordering, no reuse of any earlier call's lse.

Preblocked layout contract is VSA2's (see its layout.py): q, k, v are [B, H, L, D] contiguous bf16 or
fp16 with L = NB * 64, tokens in block-major order, block i holding blk_cnt[i] valid tokens first.
``mid_reps`` representatives of a block are its contiguous 64/mid_reps-token groups, which at
mid_reps = 4 are exactly VSA2's R = 16 router groups.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

_C = None

PHASE_A, PHASE_B, PHASE_C = 1, 2, 4
PHASE_ALL = PHASE_A | PHASE_B | PHASE_C
ROW_NORMS = {"valid_groups": 0, "paper": 1}
GATE_ACTS = {"tanh": 0, "identity": 1}
MID_REPS_OK = (1, 2, 4, 8, 16)


def ext():
    global _C
    if _C is None:
        try:
            from . import _C as C
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "vsa2_rep needs its native extension; build it with "
                "VSA2_REP_BUILD=1 pip install -e . --no-build-isolation (see README)"
            ) from e
        _C = C
    return _C


@dataclass(frozen=True)
class RepTiers:
    """Three-tier KV budget, as fractions of all NB*NB (query block, kv block) pairs.

    The default 0.10 / 0.20 keeps 64 tokens for the top 10% of the pairs, 4 representatives for the
    next 20% and 1 for the remaining 70%. ``frac_full`` alone reproduces VSA2's native budget: a
    global top-k of ``frac_full * NB * NB`` connections is ``frac_full * NB`` kv blocks per query block
    on average, i.e. VSA2's ``topk = frac_full * NB``.
    """

    frac_full: float = 0.10
    frac_mid: float = 0.20
    mid_reps: int = 4

    def __post_init__(self):
        for name in ("frac_full", "frac_mid"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"RepTiers.{name} must be a real number, got {type(v).__name__}")
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"RepTiers.{name} must be finite and >= 0, got {v!r}")
        if self.frac_full + self.frac_mid > 1.0 + 1e-9:
            raise ValueError(f"frac_full + frac_mid must be <= 1, got {self.frac_full} + {self.frac_mid}")
        if self.mid_reps not in MID_REPS_OK:
            raise ValueError(f"RepTiers.mid_reps must be one of {MID_REPS_OK}, got {self.mid_reps!r}")

    def budgets(self, nb: int) -> Tuple[int, int]:
        """(tier-A budget, tier-A|B budget) in pairs per (b, h), both clamped to NB*NB."""
        nn = nb * nb
        ba = min(nn, int(round(self.frac_full * nn)))
        bab = min(nn, max(ba, int(round((self.frac_full + self.frac_mid) * nn))))
        return ba, bab

    @property
    def group_tokens(self) -> int:
        return 64 // self.mid_reps


@dataclass
class TierSelection:
    """Frozen three-tier assignment (stop-gradient, like VSA2's own selection).

    Tier A is a CSR block selection; tiers B and C are read from the two bitmaps, so the reduced tiers
    carry no per-pair index array: tier B is ``bits_ab & ~bits_a`` and tier C is ``~bits_ab``.
    """

    q2k_ptr: torch.Tensor  # [B, H, NB+1] int32, absolute offsets into q2k_idx
    q2k_idx: torch.Tensor  # [nnz_a] int32, kv block ids ascending per row
    k2q_ptr: torch.Tensor
    k2q_idx: torch.Tensor
    bits_a: torch.Tensor   # [B, H, NB, ceil(NB/32)] int32 viewed as a bitmap
    bits_ab: torch.Tensor
    # Tier-B CSR lists (ascending), produced by routing; the TK backend walks these instead of the bitmaps.
    b_ptr: Optional[torch.Tensor] = None  # [B, H, NB+1] int32
    b_idx: Optional[torch.Tensor] = None

    def with_tier_b(self) -> "TierSelection":
        """Fill b_ptr/b_idx from the bitmaps (for hand-made selections in tests; allocates NB^2)."""
        if self.b_ptr is not None:
            return self
        nb = self.bits_a.shape[2]
        _, mb, _ = self.tier_masks(nb)
        B, H, N, _ = mb.shape
        cnt = mb.reshape(-1, N).sum(-1).to(torch.int64)
        incl = torch.cumsum(cnt, 0)
        ptr = torch.cat([(incl - cnt).reshape(B * H, N), incl.reshape(B * H, N)[:, -1:]], dim=1)
        idx = mb.reshape(-1, N).nonzero()[:, 1].to(torch.int32)
        if idx.numel() == 0:
            idx = torch.zeros(1, dtype=torch.int32, device=mb.device)
        return TierSelection(self.q2k_ptr, self.q2k_idx, self.k2q_ptr, self.k2q_idx, self.bits_a, self.bits_ab,
                             ptr.reshape(B, H, N + 1).to(torch.int32).contiguous(), idx.contiguous())

    def tier_masks(self, nb: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dense bool masks [B, H, NB, NB] of the three tiers (test/diagnostic; allocates NB^2)."""
        a = _unpack_bits(self.bits_a, nb)
        ab = _unpack_bits(self.bits_ab, nb)
        return a, ab & ~a, ~ab


def _unpack_bits(bits: torch.Tensor, nb: int) -> torch.Tensor:
    w = bits.shape[-1]
    bit = torch.arange(32, device=bits.device, dtype=torch.int32)
    x = (bits.unsqueeze(-1) >> bit) & 1  # [..., NB, W, 32]
    return x.reshape(*bits.shape[:-1], w * 32)[..., :nb].to(torch.bool)


def route_tiers(q, k, blk_cnt, tiers: RepTiers = RepTiers(), *, R: int = 16, row_norm: str = "paper",
                max_score_bytes: int = 2 << 30) -> TierSelection:
    """VSA2's PSP router + exact global top-k at two budgets (stop-gradient)."""
    C = ext()
    nb = q.shape[2] // 64
    qr, kr, _, _, _ = C.pool_fwd(q, k, k, blk_cnt, R)
    ba, bab = tiers.budgets(nb)
    out = C.route_tiers(qr, kr, blk_cnt, R, ba, bab, ROW_NORMS[row_norm], int(max_score_bytes))
    return TierSelection(*out)


BACKENDS = ("auto", "generic", "tk")


def resolve_backend(backend: str, q: torch.Tensor) -> bool:
    """True when the ThunderKittens + TMA kernels should run (sm_90, bf16, a VSA2_REP_TK=1 build)."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "generic":
        return False
    ok = (ext().has_tk() and q.is_cuda and q.dtype == torch.bfloat16
          and torch.cuda.get_device_capability(q.device) == (9, 0))
    if backend == "tk" and not ok:
        raise RuntimeError("backend='tk' needs an sm_90 GPU, bf16 inputs and a VSA2_REP_TK=1 build")
    return ok


def _row_mask(x: torch.Tensor, blk_cnt: torch.Tensor) -> torch.Tensor:
    """x with padded token rows (>= blk_cnt inside each block) zeroed; the TK backward needs a masked dO."""
    L = x.shape[2]
    valid = (torch.arange(64, device=x.device)[None, :] < blk_cnt.to(torch.int64)[:, None]).reshape(1, 1, L, 1)
    return x * valid


def _coarse_meta(B: int, H: int, nb: int, nbc_pad: int, device):
    """Dense CSR over the 64-row tiles of the pooled sequence (shape-only metadata), as in VSA2."""
    T = nbc_pad // 64
    cnt = torch.clamp(nb - 64 * torch.arange(T, device=device), max=64).to(torch.int32)
    bh = torch.arange(B * H, device=device, dtype=torch.int64)
    ptr = bh[:, None] * (T * T) + torch.arange(T + 1, device=device, dtype=torch.int64)[None, :] * T
    ptr = ptr.to(torch.int32).reshape(B, H, T + 1).contiguous()
    idx = torch.arange(T, device=device, dtype=torch.int32).repeat(B * H * T).contiguous()
    return cnt.contiguous(), ptr, idx


_POOLED_TABLES_CFG = "42"


def _pooled_tables_enabled() -> bool:
    """REP_TK_FWD_CFG=42: build the representative tables from the router pass's pooled means.

    pool_fwd_vr already produces the 16-token K/V group means (`kr`, `vr`) and the 64-token block means
    (`kc`, `vc`), which are exactly what the tier-B and tier-C representatives are; the plain build re-reads
    K and V (128 MiB) to recompute them. This is tolerance-equivalent, not bit-identical: the group means are
    rounded to bf16 before the block mean is formed.
    """
    import os
    return os.environ.get("REP_TK_FWD_CFG") != "43"  # 42 is the accepted alias, 43 rebuilds from K/V


_ROUTE_GRAPH_CACHE = {}


def _route_graph_enabled() -> bool:
    """REP_TK_FWD_CFG=41: replay a captured route_tiers instead of launching its eighteen kernels.

    Routing's inputs are the pooled q/k (8 MiB), so copying fresh pools into the captured buffers is cheap
    enough that the launch saving survives — unlike capturing the whole operator, whose 192 MiB of inputs
    cost more to copy than the graph saves.
    """
    import os
    return os.environ.get("REP_TK_FWD_CFG") == "41"


def _route_tiers_graphed(C, qr, kr, blk_cnt, R, ba, bab, norm, max_bytes):
    """None when capture is unavailable for this shape, so the caller falls back to the eager call."""
    key = (tuple(qr.shape), tuple(kr.shape), tuple(blk_cnt.shape), int(R), int(ba), int(bab), int(norm),
           qr.device.index)
    hit = _ROUTE_GRAPH_CACHE.get(key)
    if hit is False:
        return None
    if hit is None:
        try:
            qr_s, kr_s, cnt_s = qr.clone(), kr.clone(), blk_cnt.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    C.route_tiers(qr_s, kr_s, cnt_s, R, ba, bab, norm, max_bytes)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outs = C.route_tiers(qr_s, kr_s, cnt_s, R, ba, bab, norm, max_bytes)
            hit = (qr_s, kr_s, cnt_s, graph, [t.detach() for t in outs])
        except Exception:  # a shape that cannot be captured simply keeps the eager path
            _ROUTE_GRAPH_CACHE[key] = False
            return None
        _ROUTE_GRAPH_CACHE[key] = hit
    qr_s, kr_s, cnt_s, graph, outs = hit
    qr_s.copy_(qr)
    kr_s.copy_(kr)
    cnt_s.copy_(blk_cnt)
    graph.replay()
    return tuple(t.clone() for t in outs)  # a caller may keep the selection past the next call


_COARSE_META_CACHE = {}


def _cached_coarse_meta_enabled() -> bool:
    """Reuse the coarse CSR metadata across calls; the default since it was measured at -2.6% forward.

    It is shape-only (B, H, nb, nbc_pad), so the same three tensors serve every call of a given shape;
    rebuilding them per call costs a handful of tiny kernel launches inside the forward.
    REP_TK_FWD_CFG=39 is the accepted alias, 40 rebuilds them every call (the pre-cache behaviour).
    """
    import os
    return os.environ.get("REP_TK_FWD_CFG") != "40"


def _coarse_meta_cached(B: int, H: int, nb: int, nbc_pad: int, device):
    key = (B, H, nb, nbc_pad, str(device))
    hit = _COARSE_META_CACHE.get(key)
    if hit is None:
        hit = _coarse_meta(B, H, nb, nbc_pad, device)
        _COARSE_META_CACHE[key] = hit
    return hit


def _rep_fwd(C, q, k, v, blk_cnt, sel: TierSelection, mid_reps: int, phases: int):
    tab = C.rep_build(k, v, blk_cnt, mid_reps)
    o, lse = C.rep_attn_fwd(q, k, v, *tab, sel.q2k_ptr, sel.q2k_idx, sel.bits_a, sel.bits_ab, blk_cnt,
                            mid_reps, phases, 0)
    return tab, o, lse


def _rep_bwd(C, q, k, v, tab, dout, lse, delta, blk_cnt, sel: TierSelection, mid_reps: int, phases: int,
             add_dq=None, add_dk=None, add_dv=None, variant: int = 1):
    """(dq, dk, dv) of the three-tier fine branch.

    Tier-A token dK/dV always come from VSA2's own KV-owner backward on the tier-A lists (its
    unbiased `col < blk_cnt` masking is exactly a tier-A column's bias). The representative dK/dV are
    reduced into the small fp32 tables and then spread over the tokens each mean averaged.

    dQ has the same two choices VSA2 gives its own backward, and for the same reason -- the KV-owner
    kernel can produce dQ almost for free while it is already holding dS and K:

      variant 1 (default)  tier-A dQ comes from that kernel's fp32 atomics, and the query-owner
                           kernel walks only tiers B and C, adding its register dQ on top. One pass
                           over the tier-A pairs instead of two: measurably faster, and dQ's tier-A
                           part is summed in a nondeterministic fp32 order (VSA2's own default).
      variant 2            the query-owner kernel walks all three tiers and dQ is accumulated purely
                           in registers: deterministic, at the cost of recomputing the tier-A scores.

    ``add_*`` are optional per-block gradients (the coarse branch), applied as add/blk_cnt.
    """
    k_mid, v_mid, lc_mid, k_tail, v_tail, lc_tail = tab
    f32 = dict(device=q.device, dtype=torch.float32)
    dk_mid = torch.zeros(k_mid.shape, **f32)
    dv_mid = torch.zeros(v_mid.shape, **f32)
    dk_tail = torch.zeros(k_tail.shape, **f32)
    dv_tail = torch.zeros(v_tail.shape, **f32)
    split_dq = variant == 1 and bool(phases & PHASE_A) and bool(phases & (PHASE_B | PHASE_C))
    dq_acc = torch.zeros(q.shape, **f32) if split_dq else None
    if phases & PHASE_A:
        dk_a, dv_a = C.tier_a_bwd_kv(q, k, v, dout, lse, delta, sel.k2q_ptr, sel.k2q_idx, blk_cnt, dq_acc)
    else:
        dk_a = dv_a = None
    q_phases = (phases & ~PHASE_A) if split_dq else phases
    dq = C.rep_attn_bwd_q(q, k, v, k_mid, v_mid, lc_mid, k_tail, v_tail, lc_tail, dout, lse, delta,
                          sel.q2k_ptr, sel.q2k_idx, sel.bits_a, sel.bits_ab, blk_cnt, dk_mid, dv_mid,
                          dk_tail, dv_tail, mid_reps, q_phases, add_dq, dq_acc, 0)
    # The coarse branch's per-block gradients unpool exactly like a tail representative's: add/cnt
    # over every valid token of the block. Folding them in costs one add on an [B,H,NB,D] tensor.
    nb = k_tail.shape[2]
    if add_dk is not None:
        dk_tail += add_dk[:, :, :nb]
        dv_tail += add_dv[:, :, :nb]
    dk, dv = C.rep_unpool(dk_a, dv_a, dk_mid, dv_mid, dk_tail, dv_tail, blk_cnt, k, mid_reps)
    return dq, dk, dv


def tk_tables(C, k, v, blk_cnt, sel: TierSelection, mid_reps: int):
    """TK representative tables for one forward: rep_tk_build's (k_sw, v_sw, lc_mid, kt, vt, lc_tail) plus
    the tier-B log2 counts flattened per list entry (rep_tk_lcb), which both TK tier-B kernels read."""
    tab = C.rep_tk_build(k, v, blk_cnt, mid_reps)
    return tab + [C.rep_tk_lcb(tab[2], sel.b_ptr, sel.b_idx, k.shape[2] // 64, mid_reps)]


_SIDE_STREAM = None


def _side_stream():
    """One extra stream, created lazily; only the representative-table build uses it."""
    global _SIDE_STREAM
    if _SIDE_STREAM is None:
        _SIDE_STREAM = torch.cuda.Stream()
    return _SIDE_STREAM


def _overlap_tables() -> bool:
    """REP_TK_FWD_CFG=38: build the representative tables on a side stream while routing runs.

    The build depends only on K/V and blk_cnt; only the flattened tier-B log counts need the selection, so the
    expensive part can overlap route_tiers (a chain of eighteen small kernels that does not saturate the GPU).
    """
    import os
    return os.environ.get("REP_TK_FWD_CFG") == "38"


def _rep_fwd_tk_prebuilt(C, q, k, v, blk_cnt, sel: TierSelection, mid_reps: int, phases: int, tab0):
    """The TK forward with the selection-independent part of the tables already built on a side stream."""
    stream = torch.cuda.current_stream()
    stream.wait_stream(_side_stream())
    for t in tab0:
        t.record_stream(stream)
    sel = sel.with_tier_b()
    tab = tab0 + [C.rep_tk_lcb(tab0[2], sel.b_ptr, sel.b_idx, q.shape[2] // 64, mid_reps)]
    o, lse = C.rep_tk_fwd(q, k, v, tab, sel.q2k_ptr, sel.q2k_idx, sel.b_ptr, sel.b_idx, sel.bits_ab, blk_cnt,
                          mid_reps, phases, 0)
    return tab, o, lse


def _rep_fwd_tk(C, q, k, v, blk_cnt, sel: TierSelection, mid_reps: int, phases: int):
    """TK + TMA forward. tab = tk_tables(...), see csrc/rep_tk/rep_tk.cu."""
    sel = sel.with_tier_b()
    tab = tk_tables(C, k, v, blk_cnt, sel, mid_reps)
    o, lse = C.rep_tk_fwd(q, k, v, tab, sel.q2k_ptr, sel.q2k_idx, sel.b_ptr, sel.b_idx, sel.bits_ab, blk_cnt,
                          mid_reps, phases, 0)
    return tab, o, lse


def _rep_bwd_tk(C, q, k, v, tab, dout_m, lse, delta, blk_cnt, sel: TierSelection, mid_reps: int, phases: int,
                add_dq=None, add_dk=None, add_dv=None):
    """TK + TMA backward: three kernels, one shared fp32 dQ accumulator.

    tier A  VSA2's own TK backward on the tier-A k2q lists, unmodified (KV-owner; dQ by TMA reduce-add)
    tier B  query-owner; representative dK/dV by 1-D bulk reduce-add into the fp32 mid table
    tier C  KV-owner over 64 block means per CTA; their dK/dV in registers, dQ by TMA reduce-add
    ``dout_m`` must be the masked dO (0 on padded query rows), as for VSA2's TK backward.
    """
    sel = sel.with_tier_b()
    B, H, L, D = q.shape
    nb = L // 64
    f32 = dict(device=q.device, dtype=torch.float32)
    qg = torch.zeros(q.shape, **f32)
    kg = vg = None
    if phases & PHASE_A:
        # By default the tier-A kernel stores every kg/vg row exactly once (its single-owner path), so they
        # need no zero fill; REP_TK_FWD_CFG=37 (history) reduce-adds into zeroed accumulators.
        alloc = torch.empty if C.tk_bwd_store_kv() else torch.zeros
        kg, vg = alloc(q.shape, **f32), alloc(q.shape, **f32)
        C.tk_native_bwd(q, k, v, dout_m, lse, delta, sel.k2q_ptr, sel.k2q_idx, blk_cnt, qg, kg, vg)
    dk_mid = torch.zeros(B, H, nb * mid_reps, D, **f32)
    dv_mid = torch.zeros_like(dk_mid)
    if phases & PHASE_B:
        C.rep_tk_bwd_b(q, tab[0], tab[1], tab[6], dout_m, lse, delta, sel.b_ptr, sel.b_idx, qg, dk_mid, dv_mid,
                       mid_reps)
    kgt = torch.zeros(tab[3].shape, **f32)
    vgt = torch.zeros_like(kgt)
    if phases & PHASE_C:
        C.rep_tk_bwd_c(q, tab[3], tab[4], tab[5], dout_m, lse, delta, sel.bits_ab, qg, kgt, vgt)
    if add_dk is not None:  # coarse branch: unpools exactly like a block mean
        kgt[:, :, :nb] += add_dk[:, :, :nb]
        vgt[:, :, :nb] += add_dv[:, :, :nb]
    dq = C.finalize(qg, add_dq, blk_cnt, q)
    dk, dv = C.rep_unpool(kg, vg, dk_mid, dv_mid, kgt, vgt, blk_cnt, k, mid_reps)
    return dq, dk, dv


class _RepFineFunction(torch.autograd.Function):
    """Three-tier fine attention alone: no coarse branch, no gate, frozen selection."""

    @staticmethod
    def forward(ctx, q, k, v, blk_cnt, sel, mid_reps, phases, bwd_variant, use_tk):
        C = ext()
        if use_tk:
            tab, o, lse = _rep_fwd_tk(C, q, k, v, blk_cnt, sel, mid_reps, phases)
        else:
            tab, o, lse = _rep_fwd(C, q, k, v, blk_cnt, sel, mid_reps, phases)
        ctx.save_for_backward(q, k, v, blk_cnt, o, lse)
        ctx.tab, ctx.sel = tab, sel
        ctx.mid_reps, ctx.phases, ctx.bwd_variant, ctx.use_tk = mid_reps, phases, bwd_variant, use_tk
        ctx.mark_non_differentiable(lse)
        return o, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        C = ext()
        q, k, v, blk_cnt, o, lse = ctx.saved_tensors
        dout = dout.contiguous()
        if ctx.use_tk:
            dout = _row_mask(dout, blk_cnt).contiguous()
            delta = C.row_dot(dout, o, blk_cnt)
            dq, dk, dv = _rep_bwd_tk(C, q, k, v, ctx.tab, dout, lse, delta, blk_cnt, ctx.sel, ctx.mid_reps,
                                     ctx.phases)
        else:
            delta = C.row_dot(dout, o, blk_cnt)
            dq, dk, dv = _rep_bwd(C, q, k, v, ctx.tab, dout, lse, delta, blk_cnt, ctx.sel, ctx.mid_reps,
                                  ctx.phases, variant=ctx.bwd_variant)
        ctx.tab = ctx.sel = None
        return dq, dk, dv, None, None, None, None, None, None


def rep_fine_attention(q, k, v, blk_cnt, selection: TierSelection, *, mid_reps: int = 4,
                       phases: int = PHASE_ALL, bwd_variant: int = 1, backend: str = "generic"):
    """Three-tier fine attention on a frozen selection. Returns (O_fine, lse).

    ``phases`` selects which tiers run (PHASE_A | PHASE_B | PHASE_C). ``phases=PHASE_A`` is VSA2's
    native fine block-sparse attention on the tier-A lists, which makes it a direct cross-check.
    ``bwd_variant`` picks how dQ is produced (see _rep_bwd): 1 splits it between the two backward
    kernels and is faster, 2 keeps it entirely in registers and is deterministic (generic only).
    ``backend``: "generic" (cp.async + mma.sync, sm_90 and sm_120), "tk" (ThunderKittens wgmma + TMA,
    sm_90 bf16) or "auto". With "tk", padded query rows of O_fine are 0 but their lse is finite.
    """
    if bwd_variant not in (1, 2):
        raise ValueError(f"bwd_variant must be 1 or 2, got {bwd_variant!r}")
    if mid_reps not in MID_REPS_OK:
        raise ValueError(f"mid_reps must be one of {MID_REPS_OK}, got {mid_reps!r}")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous [B,H,L,D]")
    use_tk = resolve_backend(backend, q)
    return _RepFineFunction.apply(q, k, v, blk_cnt.to(torch.int32).contiguous(), selection, int(mid_reps),
                                  int(phases), int(bwd_variant), use_tk)


class _RepAttentionFunction(torch.autograd.Function):
    """Full operator: O = O_fine(three tiers) + act(gate) * repeat(O_coarse), VSA2's fusion kept."""

    @staticmethod
    def forward(ctx, q, k, v, gate, blk_cnt, R, tiers, gate_act, row_norm, max_score_bytes, sel_in, phases,
                bwd_variant, use_tk):
        C = ext()
        B, H, L, D = q.shape
        nb = L // 64
        pooled = use_tk and _pooled_tables_enabled() and 64 // tiers.mid_reps == R
        if pooled:
            qr, kr, vrl, qc, kc, vc = C.pool_fwd_vr(q, k, v, blk_cnt, R)
        else:
            qr, kr, qc, kc, vc = C.pool_fwd(q, k, v, blk_cnt, R)
        if _cached_coarse_meta_enabled():
            c_cnt, c_ptr, c_idx = _coarse_meta_cached(B, H, nb, qc.shape[2], q.device)
        else:
            c_cnt, c_ptr, c_idx = _coarse_meta(B, H, nb, qc.shape[2], q.device)
        oc, lse_c = C.attn_fwd(qc, kc, vc, c_ptr, c_idx, c_cnt)
        overlap = use_tk and sel_in is None and _overlap_tables()
        tab0 = None
        if overlap:  # the means and swizzled tables need only K/V and blk_cnt, so start them before routing
            side = _side_stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                tab0 = C.rep_tk_build(k, v, blk_cnt, tiers.mid_reps)
        if sel_in is None:
            ba, bab = tiers.budgets(nb)
            # row_norm arrives already resolved to its integer code (see rep_attention); do not remap it.
            outs = (_route_tiers_graphed(C, qr, kr, blk_cnt, R, ba, bab, row_norm, max_score_bytes)
                    if _route_graph_enabled() else None)
            if outs is None:
                outs = C.route_tiers(qr, kr, blk_cnt, R, ba, bab, row_norm, max_score_bytes)
            sel = TierSelection(*outs)
        else:
            sel = sel_in
        if use_tk:
            if pooled:
                sel = sel.with_tier_b()
                tab = C.rep_tk_build_pooled(kr, vrl, kc, vc, blk_cnt, tiers.mid_reps)
                tab = tab + [C.rep_tk_lcb(tab[2], sel.b_ptr, sel.b_idx, nb, tiers.mid_reps)]
                of, lse_f = C.rep_tk_fwd(q, k, v, tab, sel.q2k_ptr, sel.q2k_idx, sel.b_ptr, sel.b_idx,
                                         sel.bits_ab, blk_cnt, tiers.mid_reps, phases, 0)
            elif overlap:
                tab, of, lse_f = _rep_fwd_tk_prebuilt(C, q, k, v, blk_cnt, sel, tiers.mid_reps, phases, tab0)
            else:
                tab, of, lse_f = _rep_fwd_tk(C, q, k, v, blk_cnt, sel, tiers.mid_reps, phases)
        else:
            tab, of, lse_f = _rep_fwd(C, q, k, v, blk_cnt, sel, tiers.mid_reps, phases)
        o = C.fuse_fwd(of, oc, gate, blk_cnt, gate_act)
        ctx.save_for_backward(q, k, v, blk_cnt, of, lse_f, qc, kc, vc, oc, lse_c, c_cnt, c_ptr, c_idx,
                              *((gate,) if gate is not None else ()))
        ctx.tab, ctx.sel = tab, sel
        ctx.has_gate = gate is not None
        ctx.gate_act, ctx.mid_reps, ctx.phases = gate_act, tiers.mid_reps, phases
        ctx.bwd_variant, ctx.use_tk = bwd_variant, use_tk
        outs = (sel.q2k_ptr, sel.q2k_idx, sel.k2q_ptr, sel.k2q_idx, sel.bits_a, sel.bits_ab)
        ctx.mark_non_differentiable(*outs)
        return (o,) + outs

    @staticmethod
    def backward(ctx, dout, *unused):
        C = ext()
        s = ctx.saved_tensors
        q, k, v, blk_cnt, of, lse_f, qc, kc, vc, oc, lse_c, c_cnt, c_ptr, c_idx = s[:14]
        gate = s[14] if ctx.has_gate else None
        dout = dout.contiguous()
        # The generic fine kernels zero P on padded rows, so they take the unmasked dO; the TK ones need
        # fuse_bwd's masked dO_f, exactly as in VSA2.
        dout_f, delta_f, dout_c, delta_c, dgate = C.fuse_bwd(dout, of, oc, gate, blk_cnt, ctx.gate_act,
                                                             ctx.use_tk)
        dqc = torch.zeros(qc.shape, device=q.device, dtype=torch.float32)
        dkc, dvc = C.attn_bwd(qc, kc, vc, dout_c, lse_c, delta_c, c_ptr, c_idx, c_cnt, dqc, True)
        if ctx.use_tk:
            dq, dk, dv = _rep_bwd_tk(C, q, k, v, ctx.tab, dout_f, lse_f, delta_f, blk_cnt, ctx.sel,
                                     ctx.mid_reps, ctx.phases, add_dq=dqc, add_dk=dkc, add_dv=dvc)
        else:
            dq, dk, dv = _rep_bwd(C, q, k, v, ctx.tab, dout, lse_f, delta_f, blk_cnt, ctx.sel, ctx.mid_reps,
                                  ctx.phases, add_dq=dqc, add_dk=dkc, add_dv=dvc, variant=ctx.bwd_variant)
        ctx.tab = ctx.sel = None
        return (dq, dk, dv, dgate if ctx.has_gate else None) + (None,) * 10


def rep_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gate: Optional[torch.Tensor],
                  blk_cnt: torch.Tensor, *, R: int = 16, tiers: RepTiers = RepTiers(),
                  row_norm: str = "paper", gate_act: str = "tanh",
                  selection: Optional[TierSelection] = None, max_score_bytes: int = 2 << 30,
                  return_selection: bool = False, phases: int = PHASE_ALL, bwd_variant: int = 1,
                  backend: str = "generic"):
    """VSA2 attention with a three-tier representative KV budget.

    O = O_fine + act(gate) * repeat(O_coarse), with VSA2's coarse branch, gate fusion and router kept
    as they are; only the fine branch changes -- instead of dropping the unselected KV blocks it
    attends to them through representatives. With ``tiers=RepTiers(frac_full=f, frac_mid=0)`` and
    ``phases=PHASE_A`` the operator is numerically VSA2's own at ``topk = f * NB``.
    """
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous [B,H,L,D]")
    if gate is not None:
        gate = gate.contiguous()
    use_tk = resolve_backend(backend, q)
    out = _RepAttentionFunction.apply(q, k, v, gate, blk_cnt.to(torch.int32).contiguous(), int(R), tiers,
                                      GATE_ACTS[gate_act], ROW_NORMS[row_norm], int(max_score_bytes),
                                      selection, int(phases), int(bwd_variant), use_tk)
    if return_selection:
        return out[0], TierSelection(*out[1:])
    return out[0]
