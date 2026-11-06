"""Pure-PyTorch reference for the three-tier representative attention.

Written for clarity, not speed: it materializes the whole column set of every query block, so it is
only usable at test sizes. Columns are the real tokens (tier A), the mid representatives (tier B) and
the tail block means (tier C); a representative column carries ``+ ln(c)`` on its logit, which is the
natural-log form of the kernel's ``+ log2(c)``. The reference takes the tier assignment as the two
selection bitmaps the router produced, so it tests the attention math and the gradients independently
of the routing.

``round_reps`` rounds the representative tables to the token dtype, as the kernel stores them, which
is what makes a tight comparison possible.
"""

from __future__ import annotations

from typing import Tuple

import torch

from .ops import PHASE_A, PHASE_B, PHASE_C, PHASE_ALL, _unpack_bits


def build_representatives(k: torch.Tensor, v: torch.Tensor, blk_cnt: torch.Tensor, mid_reps: int,
                          dtype=torch.float64, round_to=None):
    """Plain group means over valid tokens. Returns (k_mid, v_mid, cg, k_tail, v_tail, cnt).

    k_mid / v_mid are [B, H, NB, RM, D], cg is [NB, RM] valid counts, k_tail / v_tail are [B, H, NB, D]
    and cnt is [NB]. Empty groups hold a zero mean and count 0.
    """
    B, H, L, D = k.shape
    RM = mid_reps
    NB, GS = L // 64, 64 // RM
    dev = k.device
    cnt = blk_cnt.to(dev).to(torch.int64)
    slot = torch.arange(64, device=dev).reshape(RM, GS)
    m = (slot[None] < cnt[:, None, None]).to(dtype)           # [NB, RM, GS]
    cg = m.sum(-1).to(torch.int64)                            # [NB, RM]
    kb = k.to(dtype).reshape(B, H, NB, RM, GS, D)
    vb = v.to(dtype).reshape(B, H, NB, RM, GS, D)
    w = m[None, None, :, :, :, None]
    inv = 1.0 / cg.clamp(min=1).to(dtype)[None, None, :, :, None]
    k_mid = (kb * w).sum(4) * inv
    v_mid = (vb * w).sum(4) * inv
    mk = m.reshape(NB, 64)[None, None, :, :, None]
    k_tail = (k.to(dtype).reshape(B, H, NB, 64, D) * mk).sum(3) / cnt.clamp(min=1).to(dtype)[None, None, :, None]
    v_tail = (v.to(dtype).reshape(B, H, NB, 64, D) * mk).sum(3) / cnt.clamp(min=1).to(dtype)[None, None, :, None]
    if round_to is not None:
        k_mid, v_mid = k_mid.to(round_to).to(dtype), v_mid.to(round_to).to(dtype)
        k_tail, v_tail = k_tail.to(round_to).to(dtype), v_tail.to(round_to).to(dtype)
    return k_mid, v_mid, cg, k_tail, v_tail, cnt


def rep_fine_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, blk_cnt: torch.Tensor,
                       bits_a: torch.Tensor, bits_ab: torch.Tensor, mid_reps: int = 4,
                       phases: int = PHASE_ALL, dtype=torch.float64,
                       round_reps: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """(O_fine, lse) of the three-tier attention. lse is natural; padded query rows give 0 and -inf."""
    B, H, L, D = q.shape
    RM = mid_reps
    NB = L // 64
    dev = q.device
    neg = torch.finfo(dtype).min
    round_to = q.dtype if round_reps else None
    k_mid, v_mid, cg, k_tail, v_tail, cnt = build_representatives(k, v, blk_cnt, RM, dtype, round_to)

    a = _unpack_bits(bits_a, NB)                 # [B, H, NB, NB]
    ab = _unpack_bits(bits_ab, NB)
    tier_a, tier_b, tier_c = a, ab & ~a, ~ab
    if not phases & PHASE_A:
        tier_a = torch.zeros_like(tier_a)
    if not phases & PHASE_B:
        tier_b = torch.zeros_like(tier_b)
    if not phases & PHASE_C:
        tier_c = torch.zeros_like(tier_c)

    valid_tok = (torch.arange(64, device=dev)[None, :] < cnt[:, None]).reshape(L)  # [L]
    lg = lambda x: torch.where(x > 0, x.to(dtype).clamp(min=1).log(), torch.full_like(x.to(dtype), neg))
    Kc = torch.cat([k.to(dtype), k_mid.reshape(B, H, NB * RM, D), k_tail], dim=2)
    Vc = torch.cat([v.to(dtype), v_mid.reshape(B, H, NB * RM, D), v_tail], dim=2)
    bias = torch.cat([torch.zeros(L, device=dev, dtype=dtype), lg(cg).reshape(NB * RM), lg(cnt)])

    vis = torch.cat([tier_a.repeat_interleave(64, dim=3) & valid_tok[None, None, None, :],
                     tier_b.repeat_interleave(RM, dim=3),
                     tier_c], dim=3)                                  # [B, H, NB, C]
    vis = vis.repeat_interleave(64, dim=2)                            # [B, H, L, C]

    logits = (q.to(dtype) @ Kc.transpose(-1, -2)) * (D ** -0.5) + bias
    logits = torch.where(vis, logits, torch.full_like(logits, neg))
    lse = torch.logsumexp(logits, dim=-1)
    p = torch.softmax(logits, dim=-1)
    o = p @ Vc
    row_ok = valid_tok[None, None, :] & vis.any(-1)
    o = torch.where(row_ok[..., None], o, torch.zeros_like(o))
    lse = torch.where(row_ok, lse, torch.full_like(lse, float("-inf")))
    return o, lse


def pack_bits(mask: torch.Tensor) -> torch.Tensor:
    """Dense bool [B, H, NB, NB] -> the kernels' [B, H, NB, ceil(NB/32)] int32 bitmap (inverse of
    _unpack_bits). Test utility: it lets a reference or a hand-made tier assignment drive the kernels."""
    B, H, NB, N2 = mask.shape
    assert NB == N2
    w = (NB + 31) // 32
    pad = torch.zeros(B, H, NB, w * 32 - NB, dtype=torch.bool, device=mask.device)
    m = torch.cat([mask, pad], dim=-1).reshape(B, H, NB, w, 32).to(torch.int64)
    bit = torch.arange(32, device=mask.device, dtype=torch.int64)
    packed = (m << bit).sum(-1)
    return (packed - (packed >> 31 & 1) * (1 << 32)).to(torch.int32).contiguous()
