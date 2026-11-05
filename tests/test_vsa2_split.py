"""Experimental balanced split-KV scheduling (ops.SplitKV, csrc/vsa/split.cu) against the independent oracle.

Small chunks (1..3 kv blocks) force splitting at test sizes. Every case checks O and all q/k/v/gate gradients
against vsa_reference, on masks with zero-edge rows, token tails, hot rows and hot columns, and checks the plan
itself against a pure-Python planner.

Run:  pytest -q tests/test_vsa2_split.py
"""

import math

import pytest
import torch

import vsa_reference as ref
from echosparse.vsa import ops
from echosparse.vsa.ops import ext
from test_vsa2 import BACKENDS, DEV, _compare_full, check, counts, inputs, log_metric, random_mask

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not __import__("test_vsa2")._built(),
                                reason="needs a CUDA GPU and the ECHOSPARSE_VSA_BUILD=1 extension")


def skew_mask(B, H, nb, gen, kind):
    """Skewed frozen masks: rows / columns with many edges next to empty or single-edge ones."""
    if kind in ("random", "full", "diag"):
        return random_mask(B, H, nb, gen, kind)
    m = torch.zeros(B, H, nb, nb, dtype=torch.bool)
    if kind == "hotq":  # a few dense rows, one single-edge row, the rest empty
        m[:, :, 1] = True
        m[:, :, nb - 2] = torch.rand(B, H, nb, generator=gen) < 0.8
        m[:, :, 3, 5] = True
    elif kind == "hotkv":  # every row selects 2 random blocks plus the two hot columns
        m = torch.rand(B, H, nb, nb, generator=gen) < 2.0 / nb
        m[..., 0] = True
        m[..., nb - 1] = True
        m[:, :, 2] = False  # zero-edge row
    elif kind == "hotqkv":
        m[:, :, : nb // 4] = torch.rand(B, H, nb // 4, nb, generator=gen) < 0.7
        m[..., 1] = True
        m[:, :, nb // 2] = False
    else:
        raise ValueError(kind)
    return m.to(DEV)


MASKS = ["random", "full", "hotq", "hotkv", "hotqkv"]


@pytest.fixture(params=[0, 1, 2], ids=["bwdF2", "bwdWord", "bwdSplit"])
def bwd_variant(request):
    ops.set_bwd_variant(request.param)
    yield request.param
    ops.set_bwd_variant(None)


# --------------------------------------------------------------------------- plan


def plan_reference(ptr, chunk, max_slots, min_len=0, rel_mean=0.0, tail_frac=0.0, resident=1, budget=-1):
    """Pure-Python planner with the kernel's contract (split.cu, SplitPlanParams): (tasks, merges, info[8])."""
    B, H, n1 = ptr.shape
    nb = n1 - 1
    p = ptr.reshape(B * H, n1).cpu().tolist()
    W = p[-1][-1] - p[0][0]
    gthr = max(chunk + 1, min_len)
    if tail_frac > 0:
        gthr = max(gthr, math.ceil(tail_frac * W / resident))
    budget = 2**31 - 1 if budget < 0 else budget
    rows = []
    for bh in range(B * H):
        thr = gthr if rel_mean <= 0 else max(gthr, math.ceil(rel_mean * (p[bh][nb] - p[bh][0]) / nb))
        rows += [(bh * nb + i, p[bh][i], p[bh][i + 1] - p[bh][i], p[bh][i + 1] - p[bh][i] >= thr) for i in range(nb)]
    E = sum(e for *_, e in rows)
    ok = E <= budget
    split_tasks, direct, merges, used = [], [], [], 0
    for r, beg, n, e in rows:
        c = -(-n // chunk) if e else 0
        if ok and c and used + c <= max_slots:
            merges.append([r, used, c, 0])
            split_tasks += [[r, beg + j * n // c, beg + (j + 1) * n // c, used + j] for j in range(c)]
        else:
            direct.append([r, beg, beg + n, -1])
        used += c  # rows past the cap still advance the inclusive scan
    t_split = sum(m[2] for m in merges)
    return split_tasks + direct, merges, [len(split_tasks) + len(direct), len(merges), t_split, len(rows), E, int(ok),
                                          gthr, W]


def check_plan_against_reference(ptr, idx, chunk, slots, **policy):
    tasks, merges, info = ext().split_plan(ptr, chunk, slots, **policy)
    t_ref, m_ref, i_ref = plan_reference(ptr, chunk, slots, **policy)
    torch.cuda.synchronize()
    assert info.cpu().tolist() == i_ref
    assert tasks[: i_ref[0]].cpu().tolist() == t_ref
    assert merges[: i_ref[1]].cpu().tolist() == m_ref
    cover = torch.zeros(idx.numel(), dtype=torch.int32)  # every edge covered exactly once; no edge cut
    for _, b, e, _ in t_ref:
        cover[b:e] += 1
    assert (cover == 1).all()
    return i_ref, m_ref


@pytest.mark.parametrize("kind", MASKS)
@pytest.mark.parametrize("chunk", [1, 2, 3, 64])
@pytest.mark.parametrize("cap", [0, 3, 10, None])
def test_split_plan_matches_reference(kind, chunk, cap):
    gen = torch.Generator().manual_seed(chunk)
    nb = 13
    sel = ops.mask_to_selection(skew_mask(2, 2, nb, gen, kind))
    for ptr, idx in ((sel.q2k_ptr, sel.q2k_idx), (sel.k2q_ptr, sel.k2q_idx)):
        slots = ops.split_slots(idx.numel(), chunk, 1, 1 << 40) if cap is None else cap
        i_ref, _ = check_plan_against_reference(ptr, idx, chunk, slots)
        assert ext().split_plan(ptr, chunk, slots)[0].shape[0] == 2 * 2 * nb + slots
        assert i_ref[2] <= slots


# --------------------------------------------------------------------------- fine only


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("mask_kind", MASKS)
@pytest.mark.parametrize("cnt_kind", ["tail", "mixed"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("chunk", [1, 3])
def test_split_fine_frozen(D, mask_kind, cnt_kind, dtype, chunk, bwd_variant):
    gen = torch.Generator().manual_seed(D + chunk)
    B, H, nb = 1, 2, 11
    cnt = counts(nb, cnt_kind, gen)
    q, k, v, _ = inputs(B, H, nb, D, dtype, cnt, seed=1)
    mask = skew_mask(B, H, nb, gen, mask_kind)
    sel = ops.mask_to_selection(mask)
    split = ops.SplitKV(chunk=chunk)
    qq, kk, vv = (x.clone().requires_grad_(True) for x in (q, k, v))
    of, lse = ops.fine_block_sparse_attention(qq, kk, vv, cnt, sel, backend="generic", split_kv=split)
    L = nb * 64
    valid = ref.token_valid(cnt, L)
    dout = torch.randn_like(q) * valid[None, None, :, None].to(dtype)
    dq, dk, dv = torch.autograd.grad(of, (qq, kk, vv), dout)
    qf, kf, vf = (x.float().clone().requires_grad_(True) for x in (q, k, v))
    of_r, lse_r = ref.fine_attention(qf, kf, vf, cnt, mask)
    dq_r, dk_r, dv_r = torch.autograd.grad(of_r, (qf, kf, vf), dout.float())
    rt = 2e-2 if dtype == torch.bfloat16 else 1e-2
    meta = dict(D=D, mask=mask_kind, cnt=cnt_kind, dtype=str(dtype), chunk=chunk, bwd=bwd_variant,
                max_row=int(mask.sum(-1).max()), max_col=int(mask.sum(-2).max()))
    rows = valid.nonzero().squeeze(1)
    check("split_fine_o", of, of_r, rt, rows, **meta)
    finite = torch.isfinite(lse_r[..., rows])
    assert torch.equal(torch.isfinite(lse[..., rows]), finite), "lse sentinel mismatch"
    check("split_fine_lse", torch.where(finite, lse[..., rows], 0), torch.where(finite, lse_r[..., rows], 0), 1e-2, **meta)
    check("split_fine_dq", dq, dq_r, 2 * rt, **meta)
    check("split_fine_dk", dk, dk_r, 2 * rt, **meta)
    check("split_fine_dv", dv, dv_r, 2 * rt, **meta)
    pad = (~valid).nonzero().squeeze(1)
    if pad.numel():
        assert (of[..., pad, :] == 0).all() and not torch.isfinite(lse[..., pad]).any()
        for t in (dq, dk, dv):
            assert (t[..., pad, :] == 0).all(), "padded tokens must get zero gradient"
    # same operator as the baseline schedule
    of_b, lse_b = ops.fine_block_sparse_attention(q, k, v, cnt, sel, backend="generic")
    check("split_vs_base_o", of, of_b, rt, rows, **meta)


@pytest.mark.parametrize("fwd_mt", [1, 2])
def test_split_fine_fwd_variants(fwd_mt):
    C = ext()
    old = C.get_fwd_variant()
    C.set_fwd_variant(fwd_mt)
    try:
        gen = torch.Generator().manual_seed(fwd_mt)
        B, H, nb, D = 1, 2, 12, 128
        cnt = counts(nb, "mixed", gen)
        q, k, v, _ = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=2)
        mask = skew_mask(B, H, nb, gen, "hotq")
        of, lse = ops.fine_block_sparse_attention(q, k, v, cnt, ops.mask_to_selection(mask), backend="generic",
                                                  split_kv=ops.SplitKV(chunk=2))
        of_r, _ = ref.fine_attention(q, k, v, cnt, mask)
        check("split_fwd_mt", of, of_r, 2e-2, ref.token_valid(cnt, nb * 64).nonzero().squeeze(1), fwd_mt=fwd_mt)
    finally:
        C.set_fwd_variant(old)


# --------------------------------------------------------------------------- full operator


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("gate_mode", ["token", "elem", "none"])
@pytest.mark.parametrize("mask_kind", ["random", "hotq", "hotkv", "hotqkv"])
@pytest.mark.parametrize("parts", ["both", "fwd", "bwd"])
def test_split_full_frozen(D, gate_mode, mask_kind, parts, bwd_variant):
    gen = torch.Generator().manual_seed(D + 3)
    B, H, nb = 2, 2, 10
    dtype = torch.bfloat16
    cnt = counts(nb, "mixed", gen)
    q, k, v, gate = inputs(B, H, nb, D, dtype, cnt, seed=2, gate_mode=gate_mode)
    mask = skew_mask(B, H, nb, gen, mask_kind)
    split = ops.SplitKV(chunk=2, fwd=parts != "bwd", bwd=parts != "fwd")
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v)] + ([gate.clone().requires_grad_(True)] if gate is not None else [])
    o = ops.vsa2_attention(leaves[0], leaves[1], leaves[2], leaves[3] if gate is not None else None, cnt, R=16,
                           selection=ops.mask_to_selection(mask), backend="generic", split_kv=split)
    dout = torch.randn_like(q)
    grads = torch.autograd.grad(o, leaves, dout)
    _compare_full(o, grads, leaves, cnt, mask, "split_full", dout=dout, D=D, gate=gate_mode, mask_kind=mask_kind, parts=parts,
                  bwd=bwd_variant)


@pytest.mark.parametrize("cap_slots", [0, 2, 5])
def test_split_scratch_cap_falls_back(cap_slots, bwd_variant):
    """A partial-slot cap smaller than the plan leaves later long rows on the direct path; results unchanged."""
    gen = torch.Generator().manual_seed(40 + cap_slots)
    B, H, nb, D = 1, 2, 12, 64
    cnt = counts(nb, "tail", gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=4)
    mask = skew_mask(B, H, nb, gen, "hotqkv")
    sel = ops.mask_to_selection(mask)
    # cap in forward slots (64*(D+1) fp32); the backward stages get fewer slots from the same byte cap
    split = ops.SplitKV(chunk=2, max_scratch_bytes=cap_slots * 64 * (D + 1) * 4)
    _, _, info = ext().split_plan(sel.q2k_ptr, 2, ops.split_slots(sel.q2k_idx.numel(), 2, 64 * (D + 1) * 4,
                                                                   split.max_scratch_bytes))
    long_rows = int((mask.sum(-1) > 2).sum())
    split_rows = int(info[1])
    log_metric("split_cap", cap_slots=cap_slots, split_rows=split_rows, long_rows=long_rows)
    assert split_rows <= cap_slots // 2 and split_rows < long_rows
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
    o = ops.vsa2_attention(*leaves, cnt, selection=sel, backend="generic", split_kv=split)
    dout = torch.randn_like(q)
    grads = torch.autograd.grad(o, leaves, dout)
    _compare_full(o, grads, leaves, cnt, mask, "split_cap", dout=dout, cap=cap_slots, bwd=bwd_variant)


@pytest.mark.parametrize("R", [8, 16])
def test_split_routed_matches_baseline(R, bwd_variant):
    """Same router selection with and without split-KV; both against the oracle on that selection."""
    gen = torch.Generator().manual_seed(R)
    B, H, nb, D, topk = 1, 3, 24, 128, 3
    cnt = counts(nb, "mixed", gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=7)
    dout = torch.randn_like(q)
    outs = []
    for split in (None, ops.SplitKV(chunk=2)):
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
        o, sel = ops.vsa2_attention(*leaves, cnt, R=R, topk=topk, backend="generic", return_selection=True,
                                    split_kv=split)
        grads = torch.autograd.grad(o, leaves, dout)
        outs.append((o, sel.to_mask(nb), grads, leaves))
    assert torch.equal(outs[0][1], outs[1][1]), "split_kv must not change the selection"
    o, mask, grads, leaves = outs[1]
    _compare_full(o, grads, leaves, cnt, mask, "split_routed", dout=dout, R=R, bwd=bwd_variant,
                  max_row=int(mask.sum(-1).max()))
    check("split_routed_vs_base_o", o, outs[0][0], 2e-2, R=R)
    for n, a, b in zip(["dq", "dk", "dv", "dgate"], grads, outs[0][2]):
        check(f"split_routed_vs_base_{n}", a, b, 4e-2, R=R, bwd=bwd_variant)


def test_split_original_vsa():
    gen = torch.Generator().manual_seed(9)
    B, H, nb, D, topk = 1, 2, 16, 64, 5
    cnt = counts(nb, "tail", gen)
    q, k, v, _ = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=9)
    o0 = ops.vsa_attention(q, k, v, cnt, topk, backend="generic")
    o1 = ops.vsa_attention(q, k, v, cnt, topk, backend="generic", split_kv=ops.SplitKV(chunk=2))
    check("split_vsa_o", o1, o0, 2e-2)


def test_split_deterministic_rerun():
    """Split partials are merged in a fixed order: O, dK, dV, dGate rerun bit-identically (variant 2: dQ too,
    up to the coarse branch's atomic dQ_c)."""
    gen = torch.Generator().manual_seed(19)
    B, H, nb, D = 1, 2, 18, 128
    cnt = counts(nb, "mixed", gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=19)
    sel = ops.mask_to_selection(skew_mask(B, H, nb, gen, "hotqkv"))
    dout = torch.randn_like(q)
    runs = []
    for _ in range(2):
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
        o = ops.vsa2_attention(*leaves, cnt, selection=sel, backend="generic", bwd_variant=2,
                               split_kv=ops.SplitKV(chunk=2))
        runs.append((o.detach(), torch.autograd.grad(o, leaves, dout)))
    assert torch.equal(runs[0][0], runs[1][0])
    for i in (1, 2, 3):
        assert torch.equal(runs[0][1][i], runs[1][1][i])
    check("split_deterministic_dq", runs[0][1][0], runs[1][1][0], 1e-3)


def test_split_option_checks():
    leaves_q = torch.zeros(1, 1, 64, 64, device=DEV, dtype=torch.bfloat16)
    cnt = torch.full((1,), 64, dtype=torch.int32, device=DEV)
    with pytest.raises(ValueError):
        ops.SplitKV(chunk=0)
    with pytest.raises(ValueError):
        ops.SplitKV(chunk=(1 << 30) + 1)
    with pytest.raises(TypeError):
        ops.vsa2_attention(leaves_q, leaves_q, leaves_q, None, cnt, topk=1, backend="generic", split_kv=64)
    ptr = torch.zeros(1, 1, 2, dtype=torch.int32, device=DEV)
    with pytest.raises(RuntimeError):
        ext().split_plan(ptr, (1 << 30) + 1, 0)  # binding bound, independent of the Python check
    with pytest.raises(RuntimeError):
        ext().split_plan(ptr, 0, 0)


@pytest.mark.parametrize("mask_kind", ["random", "hotq", "hotkv", "hotqkv"])
@pytest.mark.parametrize("variant", [0, 2])
def test_split_generic_vs_tk_h100(mask_kind, variant):
    """H100: balanced generic split path vs the production TK path on one frozen skewed mask; both vs the oracle."""
    if "tk" not in BACKENDS:
        pytest.skip("needs the SM90 TK backend")
    gen = torch.Generator().manual_seed(61)
    B, H, nb, D = 1, 2, 20, 128
    cnt = counts(nb, "mixed", gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=61)
    mask = skew_mask(B, H, nb, gen, mask_kind)
    sel = ops.mask_to_selection(mask)
    dout = torch.randn_like(q)
    res = {}
    for name, kw in (("tk", dict(backend="tk")),
                     ("split", dict(backend="generic", bwd_variant=variant, split_kv=ops.SplitKV(chunk=3))),
                     ("tk_split", dict(backend="tk", split_kv=ops.SplitKV(chunk=3)))):
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
        o = ops.vsa2_attention(*leaves, cnt, selection=sel, **kw)
        grads = torch.autograd.grad(o, leaves, dout)
        _compare_full(o, grads, leaves, cnt, mask, f"h100_{name}", dout=dout, mask_kind=mask_kind, bwd=variant)
        res[name] = (o, *grads)
    for n, a, b in zip(["o", "dq", "dk", "dv", "dgate"], res["split"], res["tk"]):
        check(f"h100_split_vs_tk_{n}", a, b, 4e-2, mask=mask_kind, bwd=variant)
    for n, a, b in zip(["o", "dq", "dk", "dv", "dgate"], res["tk_split"], res["tk"]):
        check(f"h100_tk_split_vs_tk_{n}", a, b, 4e-2, mask=mask_kind)


# --------------------------------------------------------------------------- TK split (H100)


def _need_tk():
    if "tk" not in BACKENDS:
        pytest.skip("needs the SM90 TK backend")


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("mask_kind", MASKS)
@pytest.mark.parametrize("cnt_kind", ["tail", "mixed"])
@pytest.mark.parametrize("chunk", [1, 3])
@pytest.mark.parametrize("parts", ["both", "fwd", "bwd"])
def test_tk_split_fine_frozen(D, mask_kind, cnt_kind, chunk, parts):
    """TK split vs the oracle and vs TK unsplit on the same frozen mask (fine branch only)."""
    _need_tk()
    gen = torch.Generator().manual_seed(D + chunk + 7)
    B, H, nb = 1, 2, 11
    cnt = counts(nb, cnt_kind, gen)
    q, k, v, _ = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=1)
    mask = skew_mask(B, H, nb, gen, mask_kind)
    sel = ops.mask_to_selection(mask)
    split = ops.SplitKV(chunk=chunk, fwd=parts != "bwd", bwd=parts != "fwd")
    L = nb * 64
    valid = ref.token_valid(cnt, L)
    dout = torch.randn_like(q) * valid[None, None, :, None].to(q.dtype)
    res = {}
    for name, sk in (("split", split), ("unsplit", None)):
        qq, kk, vv = (x.clone().requires_grad_(True) for x in (q, k, v))
        of, lse = ops.fine_block_sparse_attention(qq, kk, vv, cnt, sel, backend="tk", split_kv=sk)
        res[name] = (of, lse, *torch.autograd.grad(of, (qq, kk, vv), dout))
    qf, kf, vf = (x.float().clone().requires_grad_(True) for x in (q, k, v))
    of_r, lse_r = ref.fine_attention(qf, kf, vf, cnt, mask)
    dq_r, dk_r, dv_r = torch.autograd.grad(of_r, (qf, kf, vf), dout.float())
    meta = dict(D=D, mask_kind=mask_kind, cnt=cnt_kind, chunk=chunk, parts=parts, max_row=int(mask.sum(-1).max()),
                max_col=int(mask.sum(-2).max()))
    rows = valid.nonzero().squeeze(1)
    of, lse, dq, dk, dv = res["split"]
    check("tk_split_fine_o", of, of_r, 2e-2, rows, **meta)
    finite = torch.isfinite(lse_r[..., rows])
    assert torch.equal(torch.isfinite(lse[..., rows]), finite), "lse sentinel mismatch on valid rows"
    check("tk_split_fine_lse", torch.where(finite, lse[..., rows], 0), torch.where(finite, lse_r[..., rows], 0), 1e-2,
          **meta)
    # TK keeps padded rows of nonempty query blocks finite (its backward needs it); the split merge must too
    assert torch.equal(torch.isfinite(lse), torch.isfinite(res["unsplit"][1])), "TK lse sentinel convention changed"
    check("tk_split_fine_dq", dq, dq_r, 4e-2, **meta)
    check("tk_split_fine_dk", dk, dk_r, 4e-2, **meta)
    check("tk_split_fine_dv", dv, dv_r, 4e-2, **meta)
    for t in (dq, dk, dv):
        assert torch.isfinite(t.float()).all()
    pad = (~valid).nonzero().squeeze(1)
    if pad.numel():
        for t in (dq, dk, dv):
            assert (t[..., pad, :] == 0).all(), "padded tokens must get zero gradient"
    for n, a, b in zip(["o", "dq", "dk", "dv"], (of, dq, dk, dv), res["unsplit"][:1] + res["unsplit"][2:]):
        check(f"tk_split_vs_tk_{n}", a if n != "o" else a[..., rows, :], b if n != "o" else b[..., rows, :],
              2e-2 if n == "o" else 4e-2, **meta)


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("gate_mode", ["token", "none"])
@pytest.mark.parametrize("mask_kind", ["random", "hotq", "hotkv", "hotqkv"])
@pytest.mark.parametrize("parts", ["both", "fwd", "bwd"])
def test_tk_split_full_frozen(D, gate_mode, mask_kind, parts):
    """Complete operator, TK split: O and q/k/v/gate grads vs the oracle (coarse branch and gate once)."""
    _need_tk()
    gen = torch.Generator().manual_seed(D + 5)
    B, H, nb = 2, 2, 10
    cnt = counts(nb, "mixed", gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=2, gate_mode=gate_mode)
    mask = skew_mask(B, H, nb, gen, mask_kind)
    split = ops.SplitKV(chunk=2, fwd=parts != "bwd", bwd=parts != "fwd")
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v)] + ([gate.clone().requires_grad_(True)] if gate is not None else [])
    o = ops.vsa2_attention(leaves[0], leaves[1], leaves[2], leaves[3] if gate is not None else None, cnt, R=16,
                           selection=ops.mask_to_selection(mask), backend="tk", split_kv=split)
    dout = torch.randn_like(q)
    grads = torch.autograd.grad(o, leaves, dout)
    _compare_full(o, grads, leaves, cnt, mask, "tk_split_full", dout=dout, D=D, gate=gate_mode, mask_kind=mask_kind,
                  parts=parts)


@pytest.mark.parametrize("cap_slots", [0, 2, 5])
def test_tk_split_scratch_cap_falls_back(cap_slots):
    _need_tk()
    gen = torch.Generator().manual_seed(40 + cap_slots)
    B, H, nb, D = 1, 2, 12, 64
    cnt = counts(nb, "tail", gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=4)
    mask = skew_mask(B, H, nb, gen, "hotqkv")
    split = ops.SplitKV(chunk=2, max_scratch_bytes=cap_slots * 64 * (D + 1) * 4)
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
    o = ops.vsa2_attention(*leaves, cnt, selection=ops.mask_to_selection(mask), backend="tk", split_kv=split)
    dout = torch.randn_like(q)
    grads = torch.autograd.grad(o, leaves, dout)
    _compare_full(o, grads, leaves, cnt, mask, "tk_split_cap", dout=dout, cap=cap_slots)


@pytest.mark.parametrize("R", [8, 16])
def test_tk_split_routed_matches_tk(R):
    """Routed TK split vs routed TK unsplit: identical selection; O and all grads close; both vs the oracle."""
    _need_tk()
    gen = torch.Generator().manual_seed(R)
    B, H, nb, D, topk = 1, 3, 24, 128, 3
    cnt = counts(nb, "mixed", gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=7)
    dout = torch.randn_like(q)
    outs = []
    for split in (None, ops.SplitKV(chunk=2)):
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
        o, sel = ops.vsa2_attention(*leaves, cnt, R=R, topk=topk, backend="tk", return_selection=True, split_kv=split)
        grads = torch.autograd.grad(o, leaves, dout)
        outs.append((o, sel.to_mask(nb), grads, leaves))
    assert torch.equal(outs[0][1], outs[1][1]), "split_kv must not change the selection"
    o, mask, grads, leaves = outs[1]
    _compare_full(o, grads, leaves, cnt, mask, "tk_split_routed", dout=dout, R=R, max_row=int(mask.sum(-1).max()))
    check("tk_split_routed_vs_tk_o", o, outs[0][0], 2e-2, R=R)
    for n, a, b in zip(["dq", "dk", "dv", "dgate"], grads, outs[0][2]):
        check(f"tk_split_routed_vs_tk_{n}", a, b, 4e-2, R=R)


def test_tk_split_forward_deterministic():
    """The TK split forward merges chunks in a fixed order: O reruns bit-identically. (TK dQ, and dK/dV of
    split kv blocks, use fp32 reduce-add, so only closeness is checked for gradients.)"""
    _need_tk()
    gen = torch.Generator().manual_seed(19)
    B, H, nb, D = 1, 2, 18, 128
    cnt = counts(nb, "mixed", gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=19)
    sel = ops.mask_to_selection(skew_mask(B, H, nb, gen, "hotqkv"))
    dout = torch.randn_like(q)
    runs = []
    for _ in range(2):
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
        o = ops.vsa2_attention(*leaves, cnt, selection=sel, backend="tk", split_kv=ops.SplitKV(chunk=2))
        runs.append((o.detach(), torch.autograd.grad(o, leaves, dout)))
    assert torch.equal(runs[0][0], runs[1][0])
    assert torch.equal(runs[0][1][3], runs[1][1][3]), "dgate does not depend on the fine backward"
    for n, i in (("dq", 0), ("dk", 1), ("dv", 2)):
        check(f"tk_split_rerun_{n}", runs[0][1][i], runs[1][1][i], 1e-2)


# --------------------------------------------------------------------------- TK backward options (H100)


@pytest.mark.parametrize("opts", [1, 2, 3])
@pytest.mark.parametrize("mask_kind", ["random", "hotkv", "hotqkv"])
@pytest.mark.parametrize("chunk", [None, 2])
@pytest.mark.parametrize("D", [64, 128])
def test_tk_bwd_opts(opts, mask_kind, chunk, D):
    """dQ read-wait / rotated k2q walk (unsplit and split): same operator; O and all grads vs the oracle and vs
    opts=0 on the same mask."""
    _need_tk()
    gen = torch.Generator().manual_seed(71 + D)
    B, H, nb = 2, 2, 12
    cnt = counts(nb, "mixed", gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=71)
    mask = skew_mask(B, H, nb, gen, mask_kind)
    sel = ops.mask_to_selection(mask)
    split = None if chunk is None else ops.SplitKV(chunk=chunk)
    dout = torch.randn_like(q)
    res = {}
    for o_ in (0, opts):
        ops.set_tk_bwd_opts(o_)
        try:
            leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
            o = ops.vsa2_attention(*leaves, cnt, selection=sel, backend="tk", split_kv=split)
            assert o.grad_fn.tk_bwd_opts == o_
        finally:
            ops.set_tk_bwd_opts(None)
        grads = torch.autograd.grad(o, leaves, dout)  # options were fixed at forward
        res[o_] = (o, grads, leaves)
    o, grads, leaves = res[opts]
    _compare_full(o, grads, leaves, cnt, mask, "tk_opts", dout=dout, opts=opts, mask_kind=mask_kind, chunk=chunk, D=D)
    for n, a, b in zip(["dq", "dk", "dv", "dgate"], grads, res[0][1]):
        check(f"tk_opts_vs_opts0_{n}", a, b, 2e-2, opts=opts, mask_kind=mask_kind, chunk=chunk, D=D)
    assert torch.equal(o, res[0][0]), "the options change only the backward"


def test_tk_bwd_opts_precedence_and_checks(monkeypatch):
    _need_tk()
    monkeypatch.delenv(ops.TK_BWD_OPTS_ENV, raising=False)
    assert ops.resolve_tk_bwd_opts() == 0
    monkeypatch.setenv(ops.TK_BWD_OPTS_ENV, "3")
    assert ops.resolve_tk_bwd_opts() == 3
    try:
        ops.set_tk_bwd_opts(1)
        assert ops.resolve_tk_bwd_opts() == 1
    finally:
        ops.set_tk_bwd_opts(None)
    for bad in ("4", "7", "x"):
        monkeypatch.setenv(ops.TK_BWD_OPTS_ENV, bad)
        with pytest.raises(ValueError):
            ops.resolve_tk_bwd_opts()
    with pytest.raises(ValueError):
        ops.set_tk_bwd_opts(4)  # skip-dQ is a raw-binding diagnostic only


def test_tk_bwd_skip_dq_diagnostic_binding():
    """opts=4 leaves dQ untouched and keeps dK/dV; opts=8 is rejected."""
    _need_tk()
    gen = torch.Generator().manual_seed(5)
    B, H, nb, D = 1, 2, 8, 128
    cnt = counts(nb, "tail", gen)
    q, k, v, _ = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=5)
    sel = ops.mask_to_selection(skew_mask(B, H, nb, gen, "hotkv"))
    C = ext()
    of, lse = C.attn_fwd(q, k, v, sel.q2k_ptr, sel.q2k_idx, cnt, True)
    nbc_pad = (nb + 63) // 64 * 64
    oc0 = torch.zeros(B, H, nbc_pad, D, device=DEV, dtype=q.dtype)
    dout = torch.randn_like(q)
    dout_f, delta, _, _, _ = C.fuse_bwd(dout, of, oc0, None, cnt, 0, True)
    outs = {}
    for o_ in (0, 4):
        acc = [torch.zeros(q.shape, device=DEV, dtype=torch.float32) for _ in range(3)]
        C.tk_fine_bwd(q, k, v, dout_f, lse, delta, sel.k2q_ptr, sel.k2q_idx, cnt, *acc, o_)
        outs[o_] = acc
    assert outs[0][0].abs().max() > 0 and (outs[4][0] == 0).all()
    assert torch.equal(outs[0][1], outs[4][1]) and torch.equal(outs[0][2], outs[4][2])
    with pytest.raises(RuntimeError):
        C.tk_fine_bwd(q, k, v, dout_f, lse, delta, sel.k2q_ptr, sel.k2q_idx, cnt, *outs[0], 8)


# --------------------------------------------------------------------------- automatic policy (SplitPolicy)


def policy_mask(B, H, nb, kind, gen):
    """Base: 2 random kv blocks per query block (8 for 'uniform', K > chunk). hotq: the last query block (a tail
    row) attends to everything; hotkv: kv blocks 0 and 5 are selected by every query block; both: hotq + hotkv;
    one_hot_head: hotq in head 0 only; empty: base plus zero-edge rows and an all-empty head."""
    K = 8 if kind == "uniform" else 2
    m = torch.zeros(B, H, nb, nb, dtype=torch.bool)
    m.scatter_(-1, torch.rand(B, H, nb, nb, generator=gen).argsort(-1)[..., :K], True)
    if kind in ("hotq", "both"):
        m[:, :, nb - 1] = True
    if kind in ("hotkv", "both"):
        m[..., [0, 5]] = True
    if kind == "one_hot_head":
        m[:, 0, nb - 1] = True
    if kind == "empty":
        m[:, :, 3] = False
        m[:, -1] = False
        m[:, 0, nb - 1] = True
    return m.to(DEV)


# kind -> (forward lists expected split, kv-owner lists expected split)
POLICY_EXPECT = {"uniform": (False, False), "hotq": (True, False), "hotkv": (False, True), "both": (True, True),
                 "one_hot_head": (True, False), "empty": (True, False)}
TEST_POLICY = dict(min_len=4, rel_mean=4.0, tail_frac=0.05, resident=4)  # chunk 2: min_len = max(3, 2 * chunk)


@pytest.mark.parametrize("kind", list(POLICY_EXPECT))
def test_split_policy_plan(kind):
    """The GPU policy matches the Python planner exactly, and picks the expected direction(s)."""
    gen = torch.Generator().manual_seed(len(kind))
    B, H, nb, chunk = 1, 3, 24, 2
    sel = ops.mask_to_selection(policy_mask(B, H, nb, kind, gen))
    for (ptr, idx), expect in zip(((sel.q2k_ptr, sel.q2k_idx), (sel.k2q_ptr, sel.k2q_idx)), POLICY_EXPECT[kind]):
        slots = ops.split_slots(idx.numel(), chunk, 1, 1 << 40)
        info, merges = check_plan_against_reference(ptr, idx, chunk, slots, budget=16, **TEST_POLICY)
        assert (info[1] > 0) == expect, (kind, info)
        if kind in ("one_hot_head", "empty") and expect:
            assert {r // nb for r, *_ in merges} == {0}, "only head 0 has a long list"
    if kind == "uniform":  # K=8 > chunk=2 everywhere: manual splits every list, the policy none
        assert check_plan_against_reference(sel.q2k_ptr, sel.q2k_idx, chunk, 10**6)[0][1] == B * H * nb


def test_split_policy_budget_and_cap():
    """Too many eligible lists -> no list split (all direct); a small slot cap splits a prefix, the rest direct."""
    gen = torch.Generator().manual_seed(3)
    B, H, nb, chunk = 1, 4, 32, 2
    m = policy_mask(B, H, nb, "uniform", gen).cpu() & False
    m.scatter_(-1, torch.rand(B, H, nb, nb, generator=gen).argsort(-1)[..., :2], True)
    m[:, :, :6] = True  # 6 full rows per head: 24 eligible lists
    sel = ops.mask_to_selection(m.to(DEV))
    big = ops.split_slots(sel.q2k_idx.numel(), chunk, 1, 1 << 40)
    info, merges = check_plan_against_reference(sel.q2k_ptr, sel.q2k_idx, chunk, big, budget=23, **TEST_POLICY)
    assert info[4] == 24 and info[5] == 0 and info[1] == 0 and info[0] == B * H * nb, "over budget: all direct"
    info, merges = check_plan_against_reference(sel.q2k_ptr, sel.q2k_idx, chunk, big, budget=24, **TEST_POLICY)
    assert info[5] == 1 and info[1] == 24
    info, merges = check_plan_against_reference(sel.q2k_ptr, sel.q2k_idx, chunk, 40, budget=24, **TEST_POLICY)
    assert 0 < info[1] < 24 and info[2] <= 40, "cap exhaustion: a prefix of the eligible lists splits"


def _auto(chunk=2):
    return ops.SplitKV.auto(chunk=chunk, policy=ops.SplitPolicy(resident_ctas=4, fwd_waves=4, bwd_waves=4))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kind", list(POLICY_EXPECT))
@pytest.mark.parametrize("D", [64, 128])
def test_split_auto_operator(backend, kind, D):
    """SplitKV.auto: O and q/k/v/gate grads vs the oracle and vs unsplit; the plan splits the expected directions."""
    gen = torch.Generator().manual_seed(D + len(kind))
    B, H, nb = 1, 3, 24
    cnt = counts(nb, "mixed", gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=81)
    mask = policy_mask(B, H, nb, kind, gen)
    sel = ops.mask_to_selection(mask)
    split = _auto()
    plan = ops.describe_split(q, sel, split, backend=backend, bwd_variant=0)
    want_f, want_b = POLICY_EXPECT[kind]
    assert (plan["fwd"]["info"]["split_rows"] > 0) == want_f, plan["fwd"]
    assert (plan["bwd_kv"]["info"]["split_rows"] > 0) == want_b, plan["bwd_kv"]
    if kind in ("one_hot_head", "empty"):
        assert plan["fwd"]["split_rows_per_head"][1:] == [0] * (B * H - 1)
    dout = torch.randn_like(q)
    res = {}
    for name, sk in (("auto", split), ("unsplit", None)):
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
        o = ops.vsa2_attention(*leaves, cnt, selection=sel, backend=backend, split_kv=sk)
        res[name] = (o, torch.autograd.grad(o, leaves, dout), leaves)
    o, grads, leaves = res["auto"]
    meta = dict(backend=backend, kind=kind, D=D, fwd_split_rows=plan["fwd"]["info"]["split_rows"],
                bwd_split_rows=plan["bwd_kv"]["info"]["split_rows"])
    _compare_full(o, grads, leaves, cnt, mask, "split_auto", dout=dout, **meta)
    check("split_auto_vs_unsplit_o", o, res["unsplit"][0], 2e-2, **meta)
    for n, a, b in zip(["dq", "dk", "dv", "dgate"], grads, res["unsplit"][1]):
        check(f"split_auto_vs_unsplit_{n}", a, b, 4e-2, **meta)
    if not want_f:  # nothing split forward: the forward is the direct task path of the same kernels
        assert plan["fwd"]["info"]["tasks"] == B * H * nb


def test_split_auto_string_and_defaults():
    assert ops._check_split("auto", True) == ops.SplitKV.auto()
    a = ops.SplitKV.auto()
    assert a.policy == ops.SplitPolicy() and a.fwd and a.bwd and a.max_scratch_bytes == 64 << 20
    assert ops.SplitKV().policy is None, "manual SplitKV stays the original rule"
    with pytest.raises(ValueError):
        ops.SplitPolicy(rel_mean=-1)
    with pytest.raises(TypeError):
        ops.vsa2_attention(*(torch.zeros(1, 1, 64, 64, device=DEV, dtype=torch.bfloat16),) * 3, None,
                           torch.full((1,), 64, dtype=torch.int32, device=DEV), topk=1, split_kv="fast")


def test_split_policy_config_validation():
    """SplitPolicy rejects non-finite / negative / non-real thresholds and non-int / bool / < 1 counts."""
    assert ops.SplitPolicy() == ops.SplitPolicy(rel_mean=4.0, tail_frac=0.05, min_chunks=2, fwd_waves=1.0,
                                                bwd_waves=8.0, max_task_waves=64, resident_ctas=None)
    ops.SplitPolicy(rel_mean=4, tail_frac=0, fwd_waves=2, bwd_waves=0.5, min_chunks=3, max_task_waves=1,
                    resident_ctas=264)  # valid ints / floats / zeros
    for name in ("rel_mean", "tail_frac", "fwd_waves", "bwd_waves"):
        for bad in (float("inf"), float("-inf"), float("nan"), -1.0):
            with pytest.raises(ValueError):
                ops.SplitPolicy(**{name: bad})
        for bad in (True, "4", None, [1.0]):
            with pytest.raises(TypeError):
                ops.SplitPolicy(**{name: bad})
    for name in ("min_chunks", "max_task_waves", "resident_ctas"):
        for bad in (0, -2):
            with pytest.raises(ValueError):
                ops.SplitPolicy(**{name: bad})
        for bad in (2.0, 2.5, float("inf"), float("nan"), True, "2"):
            with pytest.raises(TypeError):
                ops.SplitPolicy(**{name: bad})
    with pytest.raises(TypeError):
        ops.SplitPolicy(min_chunks=None)
    with pytest.raises(ValueError):
        ops.SplitKV.auto(policy=ops.SplitPolicy(bwd_waves=float("nan")))
