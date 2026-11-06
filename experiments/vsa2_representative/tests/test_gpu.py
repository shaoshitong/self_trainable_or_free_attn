"""GPU tests of the three-tier representative kernels against the PyTorch reference.

Needs the built extension. Each test states what it pins down:

  routing      the two budgets really produce nested selections with the exact VSA2 counts
  rep_build    the native representative tables equal the reference group means and log2 counts
  forward      o and lse of each tier in isolation, and of the three together, match the reference
  tier_A_only  phases=PHASE_A reproduces VSA2's own fine block-sparse attention, kernel to kernel
  backward     dQ, dK, dV match the reference's autograd
  unpool       a representative's gradient is spread as drep / c over the tokens it averaged
  partial      a partial last block (blk_cnt < 64) is handled in every tier and in the gradients

Run: /opt/venv/bin/python tests/test_gpu.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa2_rep.ops import (PHASE_A, PHASE_ALL, PHASE_B, PHASE_C, RepTiers, TierSelection, ext,
                          rep_fine_attention, route_tiers, _unpack_bits)
from vsa2_rep.reference import build_representatives, pack_bits, rep_fine_reference

DEV = "cuda"
DT = torch.bfloat16
FAILED = []


def check(name, got, want, tol, extra=""):
    err = (got - want).abs().max().item()
    den = want.abs().max().item() or 1.0
    ok = err <= tol * max(den, 1.0)
    print(f"{'ok ' if ok else 'FAIL'} {name:46s} max|err| = {err:.3e}  (tol {tol:.1e} rel, scale {den:.3g}) {extra}")
    if not ok:
        FAILED.append(name)
    return ok


def make(nb, cnts, H=2, D=64, seed=0, B=1):
    torch.manual_seed(seed)
    L = nb * 64
    q, k, v = (torch.randn(B, H, L, D, device=DEV, dtype=DT).contiguous() for _ in range(3))
    blk_cnt = torch.tensor(cnts, dtype=torch.int32, device=DEV)
    # the layout contract: padded slots hold finite values and are never pooled
    valid = (torch.arange(64, device=DEV)[None, :] < blk_cnt.to(torch.int64)[:, None]).reshape(L)
    for t in (q, k, v):
        t[:, :, ~valid] = 0
    return q, k, v, blk_cnt, valid


def rand_sel(nb, H, B, pa, pb, seed):
    """A hand-made nested tier assignment, so the attention math is tested apart from the router."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    ma = torch.rand(B, H, nb, nb, device=DEV, generator=g) < pa
    mab = ma | (torch.rand(B, H, nb, nb, device=DEV, generator=g) < pb)
    ba, bab = pack_bits(ma), pack_bits(mab)
    # tier A also needs its CSR lists (the kernels walk them, and the KV-owner backward inverts them)
    qp, qi = _csr(ma)
    kp, ki = _csr(ma.transpose(-1, -2).contiguous())
    return TierSelection(qp, qi, kp, ki, ba, bab)


def _csr(m):
    B, H, N, _ = m.shape
    cnt = m.reshape(-1, N).sum(-1).to(torch.int64)
    incl = torch.cumsum(cnt, 0)
    ptr = torch.cat([(incl - cnt).reshape(B * H, N), incl.reshape(B * H, N)[:, -1:]], dim=1)
    idx = m.reshape(-1, N).nonzero()[:, 1]
    return ptr.reshape(B, H, N + 1).to(torch.int32).contiguous(), idx.to(torch.int32).contiguous()


def test_routing():
    """VSA2's exact global top-k at two budgets: nested, and exactly `budget` pairs per (b, h)."""
    nb, H, B = 16, 2, 1
    q, k, v, cnt, _ = make(nb, [64] * (nb - 1) + [40], H=H, D=64, seed=1, B=B)
    tiers = RepTiers(frac_full=0.10, frac_mid=0.20, mid_reps=4)
    sel = route_tiers(q, k, cnt, tiers, R=16)
    ba, bab = tiers.budgets(nb)
    a, b, c = sel.tier_masks(nb)
    na = a.reshape(B * H, -1).sum(-1)
    nab = (a | b).reshape(B * H, -1).sum(-1)
    ok = (na == ba).all().item() and (nab == bab).all().item()
    print(f"{'ok ' if ok else 'FAIL'} routing: |A| = {na.tolist()} (want {ba}), |A|B| = {nab.tolist()} (want {bab})")
    if not ok:
        FAILED.append("routing counts")
    nested = (a & ~(a | b)).sum().item() == 0
    partition = ((a.int() + b.int() + c.int()) == 1).all().item()
    print(f"{'ok ' if nested and partition else 'FAIL'} routing: A nested in A|B, and A|B|C partitions the pairs")
    if not (nested and partition):
        FAILED.append("routing partition")
    # the CSR tier-A list must agree with the tier-A bitmap
    C = ext()
    qp, qi = _csr(a)
    same = torch.equal(qp, sel.q2k_ptr) and torch.equal(qi, sel.q2k_idx)
    print(f"{'ok ' if same else 'FAIL'} routing: tier-A CSR list matches the tier-A bitmap")
    if not same:
        FAILED.append("routing CSR vs bitmap")


def test_rep_build():
    """Native representative tables vs the reference group means and log2 counts."""
    C = ext()
    for rm in (1, 2, 4, 8, 16):
        nb, H, D = 4, 2, 128
        q, k, v, cnt, _ = make(nb, [64, 64, 64, 23], H=H, D=D, seed=rm)
        k_mid, v_mid, lc_mid, k_tail, v_tail, lc_tail = C.rep_build(k, v, cnt, rm)
        rk, rv, cg, rkt, rvt, rc = build_representatives(k, v, cnt, rm, torch.float32)
        check(f"rep_build k_mid (RM={rm})", k_mid.float(), rk.reshape(k_mid.shape).to(DT).float(), 1e-6)
        check(f"rep_build v_tail (RM={rm})", v_tail.float(), rvt.to(DT).float(), 1e-6)
        want_lc = torch.where(cg > 0, cg.float().log2(), torch.tensor(float("-inf"), device=DEV))
        got = lc_mid[0, 0].reshape(nb, rm)
        ok = torch.equal(got, want_lc.contiguous())
        print(f"{'ok ' if ok else 'FAIL'} rep_build log2 counts (RM={rm}): {got[-1].tolist()} "
              f"for blk_cnt=23 (want {want_lc[-1].tolist()})")
        if not ok:
            FAILED.append(f"rep_build lc RM={rm}")


def test_forward_per_tier():
    """o and lse of each tier alone and of all three, kernel vs fp64 reference."""
    for D in (64, 128):
        nb, H, B = 8, 2, 1
        q, k, v, cnt, valid = make(nb, [64] * (nb - 1) + [37], H=H, D=D, seed=D, B=B)
        sel = rand_sel(nb, H, B, 0.12, 0.25, seed=D)
        for name, ph in (("A", PHASE_A), ("B", PHASE_B), ("C", PHASE_C), ("A|B|C", PHASE_ALL)):
            o, lse = rep_fine_attention(q, k, v, cnt, sel, mid_reps=4, phases=ph)
            ro, rlse = rep_fine_reference(q, k, v, cnt, sel.bits_a, sel.bits_ab, 4, ph, torch.float64)
            m = valid[None, None, :, None].expand_as(o)
            check(f"fwd o   D={D} phases={name}", o.float()[m], ro.float().to(DT).float()[m], 3e-2)
            fin = torch.isfinite(rlse) & valid[None, None, :]
            check(f"fwd lse D={D} phases={name}", lse[fin], rlse.float()[fin], 3e-3)


def test_tier_a_matches_vsa2():
    """phases=PHASE_A must equal VSA2's own fine kernel on the same tier-A lists (kernel vs kernel)."""
    C = ext()
    nb, H, B, D = 8, 2, 1, 64
    q, k, v, cnt, valid = make(nb, [64] * (nb - 1) + [51], H=H, D=D, seed=5, B=B)
    sel = rand_sel(nb, H, B, 0.2, 0.2, seed=5)
    o, lse = rep_fine_attention(q, k, v, cnt, sel, mid_reps=4, phases=PHASE_A)
    o2, lse2 = C.attn_fwd(q, k, v, sel.q2k_ptr, sel.q2k_idx, cnt)
    m = valid[None, None, :, None].expand_as(o)
    check("tier A == VSA2 attn_fwd (o)", o.float()[m], o2.float()[m], 0.0)
    fin = torch.isfinite(lse2) & valid[None, None, :]
    check("tier A == VSA2 attn_fwd (lse)", lse[fin], lse2[fin], 0.0)


def test_backward():
    """dQ, dK, dV of the three tiers against the reference's autograd."""
    for D, rm in ((64, 4), (128, 4), (64, 2)):
        nb, H, B = 6, 2, 1
        q, k, v, cnt, valid = make(nb, [64] * (nb - 1) + [45], H=H, D=D, seed=D + rm, B=B)
        sel = rand_sel(nb, H, B, 0.15, 0.3, seed=D + rm)
        torch.manual_seed(99)
        dout = torch.randn_like(q)
        dout[:, :, ~valid] = 0

        grads = {}
        for var in (1, 2):
            qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
            o, _ = rep_fine_attention(qg, kg, vg, cnt, sel, mid_reps=rm, phases=PHASE_ALL, bwd_variant=var)
            o.backward(dout)
            grads[var] = (qg.grad, kg.grad, vg.grad)
        qg, kg, vg = (type("G", (), {"grad": g})() for g in grads[1])

        qr, kr, vr = (t.detach().double().clone().requires_grad_(True) for t in (q, k, v))
        ro, _ = rep_fine_reference(qr, kr, vr, cnt, sel.bits_a, sel.bits_ab, rm, PHASE_ALL, torch.float64,
                                   round_reps=False)
        ro.backward(dout.double())

        m = valid[None, None, :, None].expand_as(q)
        for var in (1, 2):
            tag = f"D={D} RM={rm} v{var}"
            gq, gk, gv = grads[var]
            check(f"bwd dQ {tag}", gq.float()[m], qr.grad.float()[m], 4e-2)
            check(f"bwd dK {tag}", gk.float()[m], gk.float()[m] * 0 + kr.grad.float()[m], 4e-2)
            check(f"bwd dV {tag}", gv.float()[m], vr.grad.float()[m], 4e-2)
        # the two dQ paths must agree with each other, not just with the reference
        check(f"bwd dQ v1 == v2 D={D} RM={rm}", grads[1][0].float()[m], grads[2][0].float()[m], 2e-2)


def test_unpool_spreads_representative_gradient():
    """A representative's gradient must reach every token it averaged, scaled by 1/c -- including in
    a partial block, where c is the number of valid tokens, not 64."""
    C = ext()
    nb, H, B, D, rm = 3, 1, 1, 64, 4
    cnts = [64, 64, 20]
    _, k, _, cnt, valid = make(nb, cnts, H=H, D=D, seed=3, B=B)
    f32 = dict(device=DEV, dtype=torch.float32)
    dk_mid = torch.zeros(B, H, nb * rm, D, **f32)
    dv_mid = torch.zeros_like(dk_mid)
    dk_tail = torch.zeros(B, H, nb, D, **f32)
    dv_tail = torch.zeros_like(dk_tail)
    dk_tail[0, 0, 2] = 1.0            # one unit of gradient on the last block's mean (20 valid tokens)
    dk_mid[0, 0, 2 * rm + 1] = 1.0    # and on its second 16-token group (tokens 16..19 valid -> c = 4)
    dk, dv = C.rep_unpool(None, None, dk_mid, dv_mid, dk_tail, dv_tail, cnt, k, rm)
    blk2 = dk[0, 0, 128:192, 0].float()
    want = torch.zeros(64, device=DEV)
    want[:20] += 1.0 / 20.0           # tail mean over 20 valid tokens
    want[16:20] += 1.0 / 4.0          # group 1 holds slots 16..31, of which 16..19 are valid
    # dK/dV come back in the token dtype (VSA2's convention), so the bound is bf16 resolution
    check("unpool drep/c over averaged tokens", blk2, want, 4e-3,
          extra="(tail c=20, group c=4, padding 0; bf16 out)")
    ok = (dk[0, 0, 128 + 20:192] == 0).all().item()
    print(f"{'ok ' if ok else 'FAIL'} unpool: padded slots of a partial block get zero gradient")
    if not ok:
        FAILED.append("unpool padding")


def test_end_to_end_operator():
    """The full operator (router + coarse + gate + three-tier fine) runs and backprops finitely."""
    from vsa2_rep.ops import rep_attention
    nb, H, B, D = 16, 2, 1, 64
    q, k, v, cnt, valid = make(nb, [64] * (nb - 1) + [33], H=H, D=D, seed=21, B=B)
    gate = torch.zeros(B, H, nb * 64, device=DEV, dtype=DT).requires_grad_(True)
    qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    o, sel = rep_attention(qg, kg, vg, gate, cnt, R=16, tiers=RepTiers(0.1, 0.2, 4), return_selection=True)
    o.sum().backward()
    grads = {"dQ": qg.grad, "dK": kg.grad, "dV": vg.grad, "dgate": gate.grad}
    bad = [n for n, g in grads.items() if g is None or not torch.isfinite(g.float()).all()]
    nz = [n for n, g in grads.items() if g is not None and g.float().abs().max().item() > 0]
    ok = not bad and len(nz) == 4
    print(f"{'ok ' if ok else 'FAIL'} end-to-end operator: finite nonzero {sorted(nz)}"
          + (f" BAD {bad}" if bad else ""))
    if not ok:
        FAILED.append("end-to-end operator")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "these tests need a GPU"
    print(f"device: {torch.cuda.get_device_name(0)}  capability {torch.cuda.get_device_capability(0)}")
    print(f"torch {torch.__version__}\n")
    for fn in (test_routing, test_rep_build, test_forward_per_tier, test_tier_a_matches_vsa2,
               test_backward, test_unpool_spreads_representative_gradient, test_end_to_end_operator):
        print(f"--- {fn.__name__}")
        fn()
        torch.cuda.synchronize()
        print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        sys.exit(1)
    print("all GPU tests passed")
