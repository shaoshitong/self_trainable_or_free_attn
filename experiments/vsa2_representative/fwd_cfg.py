"""Forward-only timing of the TK kernel for one REP_TK_FWD_CFG (set in the environment) vs native TK.

Diagnostic text output, not a comparison: bench_compare.py runs the same-build pairs with a recorded result.
CFG 4/5/7 run tier A whatever `phases` says, write no lse and do not mask padded output; CFG 2/3 run their
tier-A path only for phases=A. Only the tier-A row is printed for them, never a B or A|B|C row. The diag=
rows change the work (no bias loads, a different load path); they locate cost and cannot establish numerical
or end-to-end acceleration.
"""
import torch
from bench_compare import active_fwd_cfg
from vsa2_rep.ops import PHASE_A, PHASE_ALL, RepTiers, ext, route_tiers, tk_tables
cfg_name, cfg_kind, cfg_meaning = active_fwd_cfg()  # before any launch reads and caches it
C = ext(); dev = "cuda"
B, H, L, D, rm = 1, 16, 16384, 128, 4
nb = L // 64
torch.manual_seed(0)
q, k, v = (torch.randn(B, H, L, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
cnt = torch.full((nb,), 64, dtype=torch.int32, device=dev)
sel = route_tiers(q, k, cnt, RepTiers(0.1, 0.2, rm), R=16)
tab = tk_tables(C, k, v, cnt, sel.with_tier_b(), rm)
def T(fn, n=50):
    for _ in range(10): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
    for _ in range(n): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n
nat = T(lambda: C.tk_native_fwd(q, k, v, sel.q2k_ptr, sel.q2k_idx, cnt))
fa = T(lambda: C.rep_tk_fwd(q, k, v, tab, sel.q2k_ptr, sel.q2k_idx, sel.b_ptr, sel.b_idx, sel.bits_ab, cnt, rm, PHASE_A, 0))
if cfg_kind != "operator":
    print(f"cfg={cfg_name}  native {nat:.3f}  A-only {fa:.3f} ({fa/nat:.3f}x)  {cfg_kind.upper()}, tier A: {cfg_meaning}")
    raise SystemExit
fall = T(lambda: C.rep_tk_fwd(q, k, v, tab, sel.q2k_ptr, sel.q2k_idx, sel.b_ptr, sel.b_idx, sel.bits_ab, cnt, rm, PHASE_ALL, 0))
print(f"cfg={cfg_name}  native {nat:.3f}  A-only {fa:.3f} ({fa/nat:.3f}x)  A|B|C {fall:.3f} ({fall/nat:.3f}x)")
from vsa2_rep.ops import PHASE_B, PHASE_C
fb = T(lambda: C.rep_tk_fwd(q, k, v, tab, sel.q2k_ptr, sel.q2k_idx, sel.b_ptr, sel.b_idx, sel.bits_ab, cnt, rm, PHASE_B, 0))
fc = T(lambda: C.rep_tk_fwd(q, k, v, tab, sel.q2k_ptr, sel.q2k_idx, sel.b_ptr, sel.b_idx, sel.bits_ab, cnt, rm, PHASE_C, 0))
nbt = ((sel.b_ptr[..., 1:] - sel.b_ptr[..., :-1]).float() / (64 // rm)).ceil().mean().item()
print(f"  B only {fb:.3f} ms ({nbt:.2f} tiles/qblock -> {fb/nbt*1e3:.1f} us per tile-per-qblock)"
      f"   C only {fc:.3f} ms (4 tiles -> {fc/4*1e3:.1f})   A tile {fa/25.6*1e3:.1f}")
if cfg_name in ("11", "24"):
    print("  diag rows skipped: the K-double-buffered kernel does not read diag")
    raise SystemExit
BULK_GATHER = ("0", "6", "8", "9", "20")  # diag 2 and 4 select bulk-copy variants, which apply only to these
dg_rows = ((1, "B: no bias loads"),)
if cfg_name in BULK_GATHER:
    dg_rows += ((2, "B: one tensor-TMA tile, no bulk"), (3, "B: neither"))
for dg, what in dg_rows:
    t = T(lambda: C.rep_tk_fwd(q, k, v, tab, sel.q2k_ptr, sel.q2k_idx, sel.b_ptr, sel.b_idx, sel.bits_ab, cnt, rm, PHASE_B, dg))
    print(f"  diag={dg} DIAGNOSTIC, changes work: {what:34s} B only {t:.3f} ms  (full {fb:.3f}: this component {fb - t:+.3f} ms)")
if cfg_name not in BULK_GATHER:
    print("  diag=2, 3, 4 skipped: they ablate the per-slot bulk gather, which this configuration does not run")
    raise SystemExit
t4 = T(lambda: C.rep_tk_fwd(q, k, v, tab, sel.q2k_ptr, sel.q2k_idx, sel.b_ptr, sel.b_idx, sel.bits_ab, cnt, rm, PHASE_B, 4))
print(f"  diag=4 DIAGNOSTIC: B copies one per lane           B only {t4:.3f} ms  (vs slot-per-lane {fb:.3f})")
