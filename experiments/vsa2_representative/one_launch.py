"""One kernel after warmup, for a profiler to select by name. Usage: one_launch.py <stage> [reps]

  generic  fwd_native fwd_a fwd_abc bwd_kv bq_a bq_bc bq_abc
  TK       tk_fwd_abc tk_fwd_b tk_fwd_c tk_bwd_a tk_bwd_b tk_bwd_c

Deterministic 16K/H16/D128/RM4 inputs, as in bench_compare.py. Tables and intermediates are built first,
and a TK forward stage launches no forward kernel before its warmup, so the printed invocation numbers
hold (counted per kernel name). REP_TK_FWD_CFG and REP_TK_BWDC_SPLITS come from the environment; tk_fwd_b and tk_fwd_c are partial
work. This is kernel-scope diagnosis: no timing, not a correctness check and not a full-operator benchmark.
"""
import json
import os
import sys
import torch
from bench_compare import active_fwd_cfg
from vsa2_rep.ops import PHASE_A, PHASE_ALL, PHASE_B, PHASE_C, RepTiers, ext, route_tiers, tk_tables

STAGES = ("fwd_native", "fwd_a", "fwd_abc", "bwd_kv", "bq_a", "bq_bc", "bq_abc",
          "tk_fwd_abc", "tk_fwd_b", "tk_fwd_c", "tk_bwd_a", "tk_bwd_b", "tk_bwd_c")
which = sys.argv[1] if len(sys.argv) > 1 else None
if which not in STAGES:
    sys.exit(f"stage must be one of {STAGES}")
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
warmup = 10
dev, dt = "cuda", torch.bfloat16
B, H, L, D, rm = 1, 16, 16384, 128, 4
nb = L // 64
torch.manual_seed(0)
q, k, v = (torch.randn(B, H, L, D, device=dev, dtype=dt).contiguous() for _ in range(3))
dout = torch.randn_like(q)
blk_cnt = torch.full((nb,), 64, dtype=torch.int32, device=dev)
C = ext()
sel = route_tiers(q, k, blk_cnt, RepTiers(0.10, 0.20, rm), R=16)
if which.startswith("tk_"):
    cfg, kind, meaning = active_fwd_cfg()
    if kind != "operator":
        sys.exit(f"REP_TK_FWD_CFG={cfg} does not run all tiers: {meaning}")
    f32 = dict(device=dev, dtype=torch.float32)
    tab = tk_tables(C, k, v, blk_cnt, sel, rm)
    qp, qi = sel.q2k_ptr, sel.q2k_idx

    def fwd(ph):
        return lambda: C.rep_tk_fwd(q, k, v, tab, qp, qi, sel.b_ptr, sel.b_idx, sel.bits_ab, blk_cnt, rm, ph, 0)

    if which in ("tk_bwd_b", "tk_bwd_c"):
        o, lse = fwd(PHASE_ALL)()
        delta = C.row_dot(dout, o, blk_cnt)
        qg = torch.zeros(q.shape, **f32)
    if which == "tk_bwd_a":  # VSA2's TK backward on the tier-A lists, with its own forward's lse
        o, lse = C.tk_native_fwd(q, k, v, qp, qi, blk_cnt)
        delta = C.row_dot(dout, o, blk_cnt)
        qg, kg, vg = (torch.zeros(q.shape, **f32) for _ in range(3))
        fn = lambda: C.tk_native_bwd(q, k, v, dout, lse, delta, sel.k2q_ptr, sel.k2q_idx, blk_cnt, qg, kg, vg)
    elif which == "tk_bwd_b":
        dkm, dvm = (torch.zeros(B, H, nb * rm, D, **f32) for _ in range(2))
        fn = lambda: C.rep_tk_bwd_b(q, tab[0], tab[1], tab[6], dout, lse, delta, sel.b_ptr, sel.b_idx, qg, dkm, dvm, rm)
    elif which == "tk_bwd_c":
        kgt, vgt = (torch.zeros(tab[3].shape, **f32) for _ in range(2))
        fn = lambda: C.rep_tk_bwd_c(q, tab[3], tab[4], tab[5], dout, lse, delta, sel.bits_ab, qg, kgt, vgt)
    else:
        fn = fwd({"tk_fwd_abc": PHASE_ALL, "tk_fwd_b": PHASE_B, "tk_fwd_c": PHASE_C}[which])
    kernel = {"tk_bwd_a": "bwd_attend_ker", "tk_bwd_b": "bwd_b_ker", "tk_bwd_c": "bwd_c_ker"}.get(which, "fwd_ker")
else:
    cfg, kernel = None, None
    tab = C.rep_build(k, v, blk_cnt, rm)
    qp, qi, kp, ki = sel.q2k_ptr, sel.q2k_idx, sel.k2q_ptr, sel.k2q_idx
    o, lse = C.rep_attn_fwd(q, k, v, *tab, qp, qi, sel.bits_a, sel.bits_ab, blk_cnt, rm, PHASE_ALL, 0)
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
                                            blk_cnt, *zm, rm, PH["a"], None, None, 0),
        "bq_bc":   lambda: C.rep_attn_bwd_q(q, k, v, *tab, dout, lse, delta, qp, qi, sel.bits_a, sel.bits_ab,
                                            blk_cnt, *zm, rm, PH["bc"], None, None, 0),
        "bq_abc":  lambda: C.rep_attn_bwd_q(q, k, v, *tab, dout, lse, delta, qp, qi, sel.bits_a, sel.bits_ab,
                                            blk_cnt, *zm, rm, PH["abc"], None, None, 0),
    }
    fn = fns[which]
for _ in range(warmup):
    fn()
torch.cuda.synchronize()
for _ in range(reps):
    fn()
torch.cuda.synchronize()
print(json.dumps(dict(stage=which, kernel=kernel, invocations=[warmup + 1, warmup + reps] if kernel else None, warmup=warmup,
                      reps=reps, B=B, H=H, L=L, D=D, RM=rm, seed=0, tables_prebuilt=True, fwd_cfg=cfg,
                      bwdc_splits=os.environ.get("REP_TK_BWDC_SPLITS", "auto"))))
