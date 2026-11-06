"""One launch of one kernel, for ncu. Usage: one_launch.py <fwd_a|fwd_abc|fwd_native|bq_a|bq_bc|bq_abc>"""
import sys
import torch
from vsa2_rep.ops import PHASE_A, PHASE_ALL, PHASE_B, PHASE_C, RepTiers, ext, route_tiers

which = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
dev, dt = "cuda", torch.bfloat16
B, H, L, D, rm = 1, 16, 16384, 128, 4
nb = L // 64
torch.manual_seed(0)
q, k, v = (torch.randn(B, H, L, D, device=dev, dtype=dt).contiguous() for _ in range(3))
blk_cnt = torch.full((nb,), 64, dtype=torch.int32, device=dev)
C = ext()
sel = route_tiers(q, k, blk_cnt, RepTiers(0.10, 0.20, rm), R=16)
tab = C.rep_build(k, v, blk_cnt, rm)
qp, qi, kp, ki = sel.q2k_ptr, sel.q2k_idx, sel.k2q_ptr, sel.k2q_idx
o, lse = C.rep_attn_fwd(q, k, v, *tab, qp, qi, sel.bits_a, sel.bits_ab, blk_cnt, rm, PHASE_ALL, 0)
dout = torch.randn_like(q)
delta = C.row_dot(dout, o, blk_cnt)
f32 = dict(device=dev, dtype=torch.float32)
zm = [torch.zeros(t.shape, **f32) for t in (tab[0], tab[1], tab[3], tab[4])]
PH = {"a": PHASE_A, "abc": PHASE_ALL, "bc": PHASE_B | PHASE_C}
fns = {
    "fwd_native": lambda: C.attn_fwd(q, k, v, qp, qi, blk_cnt),
    "fwd_a":   lambda: C.rep_attn_fwd(q, k, v, *tab, qp, qi, sel.bits_a, sel.bits_ab, blk_cnt, rm, PH["a"], 0),
    "fwd_abc": lambda: C.rep_attn_fwd(q, k, v, *tab, qp, qi, sel.bits_a, sel.bits_ab, blk_cnt, rm, PH["abc"], 0),
    "bwd_kv":  lambda: C.tier_a_bwd_kv(q, k, v, dout, lse, delta, kp, ki, blk_cnt, None),
    "bq_a":    lambda: C.rep_attn_bwd_q(q, k, v, *tab, dout, lse, delta, qp, qi, sel.bits_a, sel.bits_ab,
                                        blk_cnt, *zm, rm, PH["a"], None, None),
    "bq_bc":   lambda: C.rep_attn_bwd_q(q, k, v, *tab, dout, lse, delta, qp, qi, sel.bits_a, sel.bits_ab,
                                        blk_cnt, *zm, rm, PH["bc"], None, None),
    "bq_abc":  lambda: C.rep_attn_bwd_q(q, k, v, *tab, dout, lse, delta, qp, qi, sel.bits_a, sel.bits_ab,
                                        blk_cnt, *zm, rm, PH["abc"], None, None),
}
torch.cuda.synchronize()
for _ in range(reps):
    fns[which]()
torch.cuda.synchronize()
