"""bwd_c time vs REP_TK_BWDC_SPLITS (set in the environment); one value per process (cached nowhere)."""
import os, torch
from vsa2_rep.ops import PHASE_ALL, RepTiers, ext, route_tiers, tk_tables
C = ext(); dev = "cuda"; B, H, L, D, rm = 1, 16, 16384, 128, 4; nb = L // 64
torch.manual_seed(0)
q, k, v = (torch.randn(B, H, L, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
cnt = torch.full((nb,), 64, dtype=torch.int32, device=dev)
sel = route_tiers(q, k, cnt, RepTiers(0.1, 0.2, rm), R=16)
tab = tk_tables(C, k, v, cnt, sel.with_tier_b(), rm)
o, lse = C.rep_tk_fwd(q, k, v, tab, sel.q2k_ptr, sel.q2k_idx, sel.b_ptr, sel.b_idx, sel.bits_ab, cnt, rm, PHASE_ALL, 0)
dout = torch.randn_like(q); d = C.row_dot(dout, o, cnt)
f32 = dict(device=dev, dtype=torch.float32)
qg = torch.zeros(q.shape, **f32); kgt = torch.zeros(tab[3].shape, **f32); vgt = torch.zeros_like(kgt)
def T(fn, n=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
    for _ in range(n): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n
t = T(lambda: C.rep_tk_bwd_c(q, tab[3], tab[4], tab[5], dout, lse, d, sel.bits_ab, qg, kgt, vgt))
print(f"splits={os.environ.get('REP_TK_BWDC_SPLITS', 'auto'):>4s}  bwd_c {t:.3f} ms")
