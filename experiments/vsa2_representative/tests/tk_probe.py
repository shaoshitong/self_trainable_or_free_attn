"""Run one TK kernel on a tiny problem and report; used to localise hangs. Usage: tk_probe.py <what>"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vsa2_rep.ops import PHASE_A, PHASE_B, PHASE_C, PHASE_ALL, ext, tk_tables
from test_gpu import make, rand_sel
what = sys.argv[1]
C = ext()
nb, H, D = int(os.environ.get("NB", 8)), 2, int(os.environ.get("D", 128))
q, k, v, cnt, valid = make(nb, [64] * nb, H=H, D=D, seed=1)
sel = rand_sel(nb, H, 1, 0.12, 0.25, seed=1).with_tier_b()
print("built inputs", flush=True)
tab = tk_tables(C, k, v, cnt, sel.with_tier_b(), 4); torch.cuda.synchronize(); print("rep_tk_build ok", flush=True)
ph = {"fa": PHASE_A, "fb": PHASE_B, "fc": PHASE_C, "fall": PHASE_ALL}
if what in ph:
    o, lse = C.rep_tk_fwd(q, k, v, tab, sel.q2k_ptr, sel.q2k_idx, sel.b_ptr, sel.b_idx, sel.bits_ab, cnt, 4, ph[what], 0)
    torch.cuda.synchronize(); print(what, "ok", o.float().abs().max().item(), flush=True)
elif what == "native":
    o, lse = C.tk_native_fwd(q, k, v, sel.q2k_ptr, sel.q2k_idx, cnt); torch.cuda.synchronize(); print("native ok", flush=True)
else:
    o, lse = C.rep_tk_fwd(q, k, v, tab, sel.q2k_ptr, sel.q2k_idx, sel.b_ptr, sel.b_idx, sel.bits_ab, cnt, 4, PHASE_ALL, 0)
    dout = torch.randn_like(q)
    delta = C.row_dot(dout, o, cnt)
    f32 = dict(device="cuda", dtype=torch.float32)
    qg = torch.zeros(q.shape, **f32)
    torch.cuda.synchronize(); print("fwd for bwd ok", flush=True)
    if what == "bnative":
        kg, vg = torch.zeros(q.shape, **f32), torch.zeros(q.shape, **f32)
        C.tk_native_bwd(q, k, v, dout, lse, delta, sel.k2q_ptr, sel.k2q_idx, cnt, qg, kg, vg)
    elif what == "bb":
        dkm = torch.zeros(1, H, nb * 4, D, **f32); dvm = torch.zeros_like(dkm)
        C.rep_tk_bwd_b(q, tab[0], tab[1], tab[6], dout, lse, delta, sel.b_ptr, sel.b_idx, qg, dkm, dvm, 4)
    elif what == "bc":
        kgt = torch.zeros(tab[3].shape, **f32); vgt = torch.zeros_like(kgt)
        C.rep_tk_bwd_c(q, tab[3], tab[4], tab[5], dout, lse, delta, sel.bits_ab, qg, kgt, vgt)
    torch.cuda.synchronize(); print(what, "ok", qg.abs().max().item(), flush=True)
