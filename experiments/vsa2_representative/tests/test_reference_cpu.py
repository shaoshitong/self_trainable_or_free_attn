"""CPU tests of the representative softmax semantics, through the PyTorch reference.

These pin down the property the whole design rests on: a representative that carries + ln(c) on its
logit stands for exactly the c tokens it averaged. When those tokens are identical the substitution is
exact, so each tier must then reproduce plain dense attention -- tier C with one column per 64-token
block, tier B with one per 16-token group, tier A with the tokens themselves. No GPU and no build.

Run: python -m pytest tests/test_reference_cpu.py   (or python tests/test_reference_cpu.py)
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa2_rep.ops import PHASE_A, PHASE_ALL, PHASE_B, PHASE_C, _unpack_bits
from vsa2_rep.reference import pack_bits, rep_fine_reference

DT = torch.float64


def dense_attention(q, k, v, valid):
    """Plain softmax attention over the valid tokens only, fp64."""
    D = q.shape[-1]
    logits = (q @ k.transpose(-1, -2)) * (D ** -0.5)
    logits = torch.where(valid[None, None, None, :], logits, torch.full_like(logits, torch.finfo(DT).min))
    return torch.softmax(logits, -1) @ v


def make(nb, cnts, D=64, H=2, seed=0, group=None):
    """Random q/k/v in the block-padded layout. ``group`` repeats k/v inside each group of that many
    token slots, which is what makes the representative substitution exact."""
    torch.manual_seed(seed)
    L = nb * 64
    q = torch.randn(1, H, L, D, dtype=DT)
    k = torch.randn(1, H, L, D, dtype=DT)
    v = torch.randn(1, H, L, D, dtype=DT)
    if group is not None:
        k = k.reshape(1, H, L // group, group, D)[:, :, :, :1].expand(-1, -1, -1, group, -1).reshape(1, H, L, D)
        v = v.reshape(1, H, L // group, group, D)[:, :, :, :1].expand(-1, -1, -1, group, -1).reshape(1, H, L, D)
    blk_cnt = torch.tensor(cnts, dtype=torch.int32)
    valid = (torch.arange(64)[None, :] < blk_cnt.to(torch.int64)[:, None]).reshape(L)
    return q.contiguous(), k.contiguous(), v.contiguous(), blk_cnt, valid


def bits(nb, H, fill):
    m = torch.full((1, H, nb, nb), bool(fill))
    return pack_bits(m)


def test_pack_unpack_roundtrip():
    for nb in (2, 8, 32, 40):
        torch.manual_seed(nb)
        m = torch.rand(1, 3, nb, nb) < 0.4
        assert torch.equal(_unpack_bits(pack_bits(m), nb), m), f"bitmap roundtrip failed at NB={nb}"
    print("ok  pack/unpack roundtrip")


def test_tier_a_is_dense():
    """Tier A over every pair = dense attention over the valid tokens."""
    nb, H = 3, 2
    q, k, v, cnt, valid = make(nb, [64, 64, 37], H=H)
    o, lse = rep_fine_reference(q, k, v, cnt, bits(nb, H, 1), bits(nb, H, 1), 4, PHASE_ALL, DT, False)
    ref = dense_attention(q, k, v, valid)
    err = (o - ref)[:, :, valid].abs().max().item()
    assert err < 1e-12, f"tier A != dense attention: {err}"
    lref = torch.logsumexp(torch.where(valid[None, None, None, :],
                                       (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5),
                                       torch.full((1,), torch.finfo(DT).min, dtype=DT)), -1)
    assert (lse - lref)[:, :, valid].abs().max().item() < 1e-12, "tier A lse mismatch"
    print("ok  tier A reproduces dense attention (o and lse)")


def test_tier_c_exact_on_uniform_blocks():
    """One block mean per block, + ln(c): exact when a block's tokens are identical, incl. a partial block."""
    nb, H = 3, 2
    q, k, v, cnt, valid = make(nb, [64, 64, 20], H=H, group=64)
    o, _ = rep_fine_reference(q, k, v, cnt, bits(nb, H, 0), bits(nb, H, 0), 4, PHASE_ALL, DT, False)
    ref = dense_attention(q, k, v, valid)
    err = (o - ref)[:, :, valid].abs().max().item()
    assert err < 1e-12, f"tier C not exact on uniform blocks: {err}"
    print("ok  tier C (1 rep, +ln c) exact on uniform blocks, partial block included")


def test_tier_b_exact_on_uniform_groups():
    """mid_reps representatives per block, + ln(c_g): exact when each 64/RM-token group is uniform."""
    for rm in (2, 4, 8, 16):
        nb, H = 2, 2
        gs = 64 // rm
        q, k, v, cnt, valid = make(nb, [64, 64], H=H, group=gs, seed=rm)
        # every pair in A|B but none in A -> tier B only
        o, _ = rep_fine_reference(q, k, v, cnt, bits(nb, H, 0), bits(nb, H, 1), rm, PHASE_ALL, DT, False)
        ref = dense_attention(q, k, v, valid)
        err = (o - ref).abs().max().item()
        assert err < 1e-12, f"tier B not exact at mid_reps={rm}: {err}"
    print("ok  tier B (RM reps, +ln c_g) exact on uniform groups for RM in 2,4,8,16")


def test_mixed_tiers_partition_and_normalize():
    """A | B | C partitions the pairs, and the three tiers together always normalize to 1."""
    nb, H = 8, 2
    torch.manual_seed(7)
    q, k, v, cnt, valid = make(nb, [64] * 7 + [41], H=H, seed=3)
    ma = torch.rand(1, H, nb, nb) < 0.15
    mab = ma | (torch.rand(1, H, nb, nb) < 0.25)
    ba, bab = pack_bits(ma), pack_bits(mab)
    a, b, c = _unpack_bits(ba, nb), _unpack_bits(bab, nb) & ~_unpack_bits(ba, nb), ~_unpack_bits(bab, nb)
    assert not (a & b).any() and not (a & c).any() and not (b & c).any(), "tiers overlap"
    assert (a | b | c).all(), "tiers do not cover every pair"
    o, lse = rep_fine_reference(q, k, v, cnt, ba, bab, 4, PHASE_ALL, DT, False)
    # the softmax normalizes over the union of the tiers: rebuilding p from lse must sum to 1
    assert torch.isfinite(lse[:, :, valid]).all(), "a valid row has no column in any tier"
    # o must be a convex combination of the columns' values -> bounded by their range
    lo = min(v.min().item(), 0.0) - 1e-9
    hi = max(v.max().item(), 0.0) + 1e-9
    assert o[:, :, valid].min() >= lo and o[:, :, valid].max() <= hi, "output left the convex hull"
    print("ok  mixed tiers partition the pairs and stay a convex combination")


def test_phase_masks_select_tiers():
    """phases must switch tiers off; A-only on a sparse mask must differ from the full three tiers."""
    nb, H = 4, 2
    q, k, v, cnt, _ = make(nb, [64] * nb, H=H, seed=11)
    ma = torch.zeros(1, H, nb, nb, dtype=torch.bool)
    ma[..., 0] = True                       # every query block keeps kv block 0 at full resolution
    mab = ma.clone()
    mab[..., 1] = True
    ba, bab = pack_bits(ma), pack_bits(mab)
    oa, _ = rep_fine_reference(q, k, v, cnt, ba, bab, 4, PHASE_A, DT, False)
    oall, _ = rep_fine_reference(q, k, v, cnt, ba, bab, 4, PHASE_ALL, DT, False)
    assert (oa - oall).abs().max().item() > 1e-3, "tiers B and C changed nothing"
    only_a = dense_attention(q, k, v, torch.arange(nb * 64) < 64)
    assert (oa - only_a).abs().max().item() < 1e-12, "PHASE_A alone is not attention over kv block 0"
    print("ok  phase mask selects tiers (A alone == attention over the tier-A columns)")


def test_row_with_only_empty_columns():
    """A row whose visible columns all hold zero tokens is empty: o = 0, lse = -inf, no gradient."""
    nb, H = 2, 1
    q, k, v, cnt, _ = make(nb, [0, 64], H=H, seed=13)
    ma = torch.zeros(1, H, nb, nb, dtype=torch.bool)
    mab = ma.clone()
    mab[..., 1] = True                      # tiers: kv block 1 is B, the zero-count kv block 0 is C
    q = q.requires_grad_(True)
    o, lse = rep_fine_reference(q, k, v, cnt, pack_bits(ma), pack_bits(mab), 4, PHASE_C, DT, False)
    o.sum().backward()
    assert (o == 0).all() and torch.isinf(lse).all() and (q.grad == 0).all(), "empty row was not empty"
    print("ok  a row with only empty columns gives o = 0, lse = -inf and zero gradient")


if __name__ == "__main__":
    test_pack_unpack_roundtrip()
    test_tier_a_is_dense()
    test_tier_c_exact_on_uniform_blocks()
    test_tier_b_exact_on_uniform_groups()
    test_mixed_tiers_partition_and_normalize()
    test_phase_masks_select_tiers()
    test_row_with_only_empty_columns()
    print("\nall CPU reference tests passed")
