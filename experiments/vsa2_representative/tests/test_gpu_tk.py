"""GPU tests of the ThunderKittens + TMA backend against the fp64 reference, the generic backend and
VSA2's native TK kernel. Needs a VSA2_REP_TK=1 build on an sm_90 GPU.

  tables      pre-swizzled mid table un-swizzles to the plain group means; padded tail rows
  forward     o and lse per tier and all three, vs the fp64 reference (D = 64, 128; RM = 2, 4, 8)
  native      phases=A vs VSA2's own TK forward on the same lists
  generic     TK vs generic backend, same selection, all tiers
  backward    dQ, dK, dV vs the reference's autograd, and vs the generic backend
  routed      the full operator on a routed selection, TK vs generic

Run: /opt/venv/bin/python tests/test_gpu_tk.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa2_rep.ops import (PHASE_A, PHASE_ALL, PHASE_B, PHASE_C, RepTiers, ext, rep_attention,
                          rep_fine_attention, route_tiers)
from vsa2_rep.reference import build_representatives, rep_fine_reference

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_gpu import FAILED, check, make, rand_sel  # noqa: E402

DEV = "cuda"


def unswizzle(sw, rm, nb, D):
    """[B,H,NP,NB+1,D/64,RM,64] pre-swizzled records (phase 0) -> plain [B,H,NB,RM,D]."""
    rec = sw[:, :, 0, :nb]                                  # phase 0: tile rows 0..RM-1 of an atom
    B, H = rec.shape[:2]
    out = torch.empty(B, H, nb, rm, D, dtype=sw.dtype, device=sw.device)
    for g in range(rm):
        for p in range(D // 64):
            row = rec[:, :, :, p, g].reshape(B, H, nb, 8, 8)     # 8 chunks of 8 elements
            sw_ = g % 8
            perm = torch.tensor([c ^ sw_ for c in range(8)], device=sw.device)
            out[:, :, :, g, p * 64:(p + 1) * 64] = row[:, :, :, perm].reshape(B, H, nb, 64)
    return out


def test_tables():
    C = ext()
    for rm in (2, 4, 8, 16):
        nb, H, D = 5, 2, 128
        q, k, v, cnt, _ = make(nb, [64, 64, 64, 64, 23], H=H, D=D, seed=rm)
        k_sw, v_sw, lc_mid, kt, vt, lc_tail = C.rep_tk_build(k, v, cnt, rm)
        rk, rv, cg, rkt, rvt, rc = build_representatives(k, v, cnt, rm, torch.float32)
        check(f"tk tables: k_mid un-swizzled (RM={rm})", unswizzle(k_sw, rm, nb, D).float(),
              rk.to(torch.bfloat16).float(), 0.0)
        check(f"tk tables: tail means (RM={rm})", kt[:, :, :nb].float(), rkt.to(torch.bfloat16).float(), 0.0)
        ok = (kt[:, :, nb:] == 0).all().item() and torch.isinf(lc_tail[:, :, nb:]).all().item() \
            and (k_sw[:, :, :, nb] == 0).all().item()
        print(f"{'ok ' if ok else 'FAIL'} tk tables: padded tail rows 0/-inf and zero record (RM={rm})")
        if not ok:
            FAILED.append(f"tk tables padding RM={rm}")


def test_forward():
    for D, rm in ((64, 4), (128, 4), (128, 2), (64, 8)):
        nb, H, B = 8, 2, 1
        q, k, v, cnt, valid = make(nb, [64] * (nb - 1) + [37], H=H, D=D, seed=D + rm, B=B)
        sel = rand_sel(nb, H, B, 0.12, 0.25, seed=D + rm)
        for name, ph in (("A", PHASE_A), ("B", PHASE_B), ("C", PHASE_C), ("A|B|C", PHASE_ALL)):
            o, lse = rep_fine_attention(q, k, v, cnt, sel, mid_reps=rm, phases=ph, backend="tk")
            ro, rlse = rep_fine_reference(q, k, v, cnt, sel.bits_a, sel.bits_ab, rm, ph, torch.float64)
            m = valid[None, None, :, None].expand_as(o)
            check(f"tk fwd o   D={D} RM={rm} {name}", o.float()[m], ro.float().to(torch.bfloat16).float()[m], 3e-2)
            fin = torch.isfinite(rlse) & valid[None, None, :]
            check(f"tk fwd lse D={D} RM={rm} {name}", lse[fin], rlse.float()[fin], 3e-3)


def test_native_and_generic():
    C = ext()
    nb, H, B, D = 16, 2, 1, 128
    q, k, v, cnt, valid = make(nb, [64] * (nb - 1) + [51], H=H, D=D, seed=5, B=B)
    sel = rand_sel(nb, H, B, 0.15, 0.25, seed=5)
    o, lse = rep_fine_attention(q, k, v, cnt, sel, mid_reps=4, phases=PHASE_A, backend="tk")
    on, lsen = C.tk_native_fwd(q, k, v, sel.q2k_ptr, sel.q2k_idx, cnt)
    m = valid[None, None, :, None].expand_as(o)
    check("tk A-only vs native TK fwd (o)", o.float()[m], on.float()[m], 1e-2)
    # query blocks with no tier-A block give lse = -inf in both kernels; compare the finite rows, and
    # require the two kernels to agree on which rows those are
    fin = torch.isfinite(lsen) & valid[None, None]
    same = torch.equal(torch.isfinite(lse) & valid[None, None], fin)
    print(f"{'ok ' if same else 'FAIL'} tk A-only vs native TK fwd: same empty (-inf) rows "
          f"({int((~torch.isfinite(lsen) & valid[None, None]).sum())} of them)")
    if not same:
        FAILED.append("tk vs native empty rows")
    check("tk A-only vs native TK fwd (lse)", lse[fin], lsen[fin], 1e-4)
    og, lseg = rep_fine_attention(q, k, v, cnt, sel, mid_reps=4, backend="generic")
    ot, lset = rep_fine_attention(q, k, v, cnt, sel, mid_reps=4, backend="tk")
    check("tk vs generic, A|B|C (o)", ot.float()[m], og.float()[m], 1e-2)
    vm = valid[None, None].expand_as(lset)
    check("tk vs generic, A|B|C (lse)", lset[vm], lseg[vm], 1e-4)


def test_backward():
    for D, rm in ((64, 4), (128, 4), (128, 2)):
        nb, H, B = 6, 2, 1
        q, k, v, cnt, valid = make(nb, [64] * (nb - 1) + [45], H=H, D=D, seed=D + rm, B=B)
        sel = rand_sel(nb, H, B, 0.15, 0.3, seed=D + rm)
        torch.manual_seed(99)
        dout = torch.randn_like(q)
        dout[:, :, ~valid] = 0
        grads = {}
        for be in ("tk", "generic"):
            qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
            o, _ = rep_fine_attention(qg, kg, vg, cnt, sel, mid_reps=rm, backend=be)
            o.backward(dout)
            grads[be] = (qg.grad, kg.grad, vg.grad)
        qr, kr, vr = (t.detach().double().clone().requires_grad_(True) for t in (q, k, v))
        ro, _ = rep_fine_reference(qr, kr, vr, cnt, sel.bits_a, sel.bits_ab, rm, PHASE_ALL, torch.float64,
                                   round_reps=False)
        ro.backward(dout.double())
        m = valid[None, None, :, None].expand_as(q)
        ref = (qr.grad, kr.grad, vr.grad)
        for n, i in (("dQ", 0), ("dK", 1), ("dV", 2)):
            check(f"tk bwd {n} D={D} RM={rm} vs reference", grads["tk"][i].float()[m], ref[i].float()[m], 4e-2)
            check(f"tk bwd {n} D={D} RM={rm} vs generic", grads["tk"][i].float()[m],
                  grads["generic"][i].float()[m], 2e-2)


def test_routed_operator():
    nb, H, B, D = 32, 2, 1, 128
    q, k, v, cnt, valid = make(nb, [64] * (nb - 1) + [33], H=H, D=D, seed=21, B=B)
    gate = (0.1 * torch.randn(B, H, nb * 64, device=DEV)).to(torch.bfloat16)
    torch.manual_seed(3)
    dout = torch.randn_like(q)
    res = {}
    for be in ("tk", "generic"):
        qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        gg = gate.detach().clone().requires_grad_(True)
        o = rep_attention(qg, kg, vg, gg, cnt, R=16, tiers=RepTiers(0.1, 0.2, 4), backend=be)
        o.backward(dout)
        res[be] = (o, qg.grad, kg.grad, vg.grad, gg.grad)
    m = valid[None, None, :, None].expand_as(q)
    for n, i in (("o", 0), ("dQ", 1), ("dK", 2), ("dV", 3)):
        check(f"routed operator {n}: tk vs generic", res["tk"][i].float()[m], res["generic"][i].float()[m], 2e-2)
    mg = valid[None, None].expand_as(gate)
    check("routed operator dgate: tk vs generic", res["tk"][4].float()[mg], res["generic"][4].float()[mg], 2e-2)


if __name__ == "__main__":
    assert ext().has_tk(), "needs a VSA2_REP_TK=1 build"
    print(f"device: {torch.cuda.get_device_name(0)}\n")
    for fn in (test_tables, test_forward, test_native_and_generic, test_backward, test_routed_operator):
        print(f"--- {fn.__name__}")
        fn()
        torch.cuda.synchronize()
        print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        sys.exit(1)
    print("all TK GPU tests passed")
