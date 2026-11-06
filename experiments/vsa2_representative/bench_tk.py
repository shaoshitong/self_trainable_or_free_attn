"""Generic vs ThunderKittens+TMA vs native VSA2 TK, on identical inputs, selection and timing boundaries.

Every row is timed the same way: CUDA events around a loop of `iters` calls after `warmup` calls, one
synchronize at the end, nothing logged or synchronized per call. Two scopes are reported separately so
that no row silently includes work another one omits:

  kernel   the attention kernel(s) alone, representative tables prebuilt
  op       what one forward / backward of the autograd op costs (includes table build, zero-inits,
           delta, unpool, finalize); the backward is (fwd + bwd) - fwd

The native TK rows are VSA2's own fine kernels on the tier-A lists -- the 10% budget VSA2 keeps, with
the other 90% of the pairs dropped. They are the baseline every three-tier row is compared against.
"""
import argparse
import sys

import torch

from vsa2_rep.ops import tk_tables, PHASE_A, PHASE_ALL, PHASE_B, PHASE_C, RepTiers, ext, rep_fine_attention, route_tiers


def timeit(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int, default=16384)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--dim", type=int, default=128, choices=(64, 128))
    ap.add_argument("--mid-reps", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args()
    dev, dt = "cuda", torch.bfloat16
    B, H, L, D, nb, rm = 1, a.heads, a.len, a.dim, a.len // 64, a.mid_reps
    T = lambda fn: timeit(fn, a.iters, a.warmup)
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, L, D, device=dev, dtype=dt).contiguous() for _ in range(3))
    blk_cnt = torch.full((nb,), 64, dtype=torch.int32, device=dev)
    C = ext()
    sel = route_tiers(q, k, blk_cnt, RepTiers(0.10, 0.20, rm), R=16)
    qp, qi, kp, ki = sel.q2k_ptr, sel.q2k_idx, sel.k2q_ptr, sel.k2q_idx
    f32 = dict(device=dev, dtype=torch.float32)
    print(f"{torch.cuda.get_device_name(0)}  B={B} H={H} L={L} D={D} NB={nb} RM={rm} bf16  "
          f"tiers 10%/20%/70%  iters={a.iters} warmup={a.warmup}\n")

    # ------------------------------------------------------------ kernel scope, forward
    gtab = C.rep_build(k, v, blk_cnt, rm)
    ttab = tk_tables(C, k, v, blk_cnt, sel.with_tier_b(), rm)
    rows = []
    rows.append(("native TK attn_fwd (tier A only)", T(lambda: C.tk_native_fwd(q, k, v, qp, qi, blk_cnt))))
    rows.append(("native generic attn_fwd (tier A only)", T(lambda: C.attn_fwd(q, k, v, qp, qi, blk_cnt))))
    for name, ph in (("A", PHASE_A), ("A|B", PHASE_A | PHASE_B), ("A|B|C", PHASE_ALL), ("B|C", PHASE_B | PHASE_C)):
        rows.append((f"generic rep_attn_fwd {name}",
                     T(lambda ph=ph: C.rep_attn_fwd(q, k, v, *gtab, qp, qi, sel.bits_a, sel.bits_ab, blk_cnt, rm,
                                                    ph, 0))))
        rows.append((f"TK      rep_tk_fwd   {name}",
                     T(lambda ph=ph: C.rep_tk_fwd(q, k, v, ttab, qp, qi, sel.b_ptr, sel.b_idx, sel.bits_ab, blk_cnt,
                                                  rm, ph, 0))))
    rows.append(("generic rep_build", T(lambda: C.rep_build(k, v, blk_cnt, rm))))
    rows.append(("TK      tk_tables (build + lcb)", T(lambda: tk_tables(C, k, v, blk_cnt, sel, rm))))
    base = rows[0][1]
    print("--- forward, kernel scope (tables prebuilt)                 ms     vs native TK")
    for n, t in rows:
        print(f"  {n:52s} {t:8.3f}   {t/base:6.3f}x")

    # ------------------------------------------------------------ kernel scope, backward
    o_t, lse_t = C.rep_tk_fwd(q, k, v, ttab, qp, qi, sel.b_ptr, sel.b_idx, sel.bits_ab, blk_cnt, rm, PHASE_ALL, 0)
    o_g, lse_g = C.rep_attn_fwd(q, k, v, *gtab, qp, qi, sel.bits_a, sel.bits_ab, blk_cnt, rm, PHASE_ALL, 0)
    dout = torch.randn_like(q)
    d_t = C.row_dot(dout, o_t, blk_cnt)
    d_g = C.row_dot(dout, o_g, blk_cnt)
    qg, kg, vg = (torch.zeros(q.shape, **f32) for _ in range(3))
    dkm, dvm = torch.zeros(B, H, nb * rm, D, **f32), torch.zeros(B, H, nb * rm, D, **f32)
    kgt, vgt = torch.zeros(ttab[3].shape, **f32), torch.zeros(ttab[3].shape, **f32)
    zm = [torch.zeros(t.shape, **f32) for t in (gtab[0], gtab[1], gtab[3], gtab[4])]
    o_n, lse_n = C.tk_native_fwd(q, k, v, qp, qi, blk_cnt)
    d_n = C.row_dot(dout, o_n, blk_cnt)
    brows = [
        ("native TK bwd (tier A only; = TK tier A)",
         T(lambda: C.tk_native_bwd(q, k, v, dout, lse_n, d_n, kp, ki, blk_cnt, qg, kg, vg))),
        ("TK tier B  (query-owner, bulk reduce)",
         T(lambda: C.rep_tk_bwd_b(q, ttab[0], ttab[1], ttab[6], dout, lse_t, d_t, sel.b_ptr, sel.b_idx, qg, dkm,
                                  dvm, rm))),
        ("TK tier C  (KV-owner over block means)",
         T(lambda: C.rep_tk_bwd_c(q, ttab[3], ttab[4], ttab[5], dout, lse_t, d_t, sel.bits_ab, qg, kgt, vgt))),
        ("generic tier A KV-owner + atomic dQ",
         T(lambda: C.tier_a_bwd_kv(q, k, v, dout, lse_g, d_g, kp, ki, blk_cnt, qg))),
        ("generic B|C query-owner",
         T(lambda: C.rep_attn_bwd_q(q, k, v, *gtab, dout, lse_g, d_g, qp, qi, sel.bits_a, sel.bits_ab, blk_cnt,
                                    *zm, rm, PHASE_B | PHASE_C, None, qg, 0))),
    ]
    print("\n--- backward, kernel scope                                    ms")
    for n, t in brows:
        print(f"  {n:52s} {t:8.3f}")
    tk_b = brows[0][1] + brows[1][1] + brows[2][1]
    ge_b = brows[3][1] + brows[4][1]
    print(f"  {'=> TK three tiers (A native + B + C)':52s} {tk_b:8.3f}   {tk_b/brows[0][1]:.3f}x native TK")
    print(f"  {'=> generic three tiers (variant 1)':52s} {ge_b:8.3f}   {ge_b/brows[0][1]:.3f}x native TK")
    print(f"  {'   TK / generic':52s} {tk_b/ge_b:8.3f}x")

    # ------------------------------------------------------------ op scope
    print("\n--- op scope (rep_fine_attention autograd op, frozen selection)   fwd ms    bwd ms")
    qq, kk, vv = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    op = {}
    for be in ("generic", "tk"):
        for name, ph in (("A only", PHASE_A), ("A|B|C", PHASE_ALL)):
            f = T(lambda: rep_fine_attention(q, k, v, blk_cnt, sel, mid_reps=rm, phases=ph, backend=be))

            def fb():
                o, _ = rep_fine_attention(qq, kk, vv, blk_cnt, sel, mid_reps=rm, phases=ph, backend=be)
                o.backward(dout)
                qq.grad = kk.grad = vv.grad = None
            op[(be, name)] = (f, T(fb) - f)
            print(f"  {be:8s} {name:8s}{'':40s} {f:8.3f}  {op[(be, name)][1]:8.3f}")

    def native_op_fb():
        o, lse = C.tk_native_fwd(q, k, v, qp, qi, blk_cnt)
        dd = C.row_dot(dout, o, blk_cnt)
        a1, a2, a3 = (torch.zeros(q.shape, **f32) for _ in range(3))
        C.tk_native_bwd(q, k, v, dout, lse, dd, kp, ki, blk_cnt, a1, a2, a3)
        return C.finalize(a1, None, blk_cnt, q), C.finalize(a2, None, blk_cnt, k), C.finalize(a3, None, blk_cnt, v)
    nf = T(lambda: C.tk_native_fwd(q, k, v, qp, qi, blk_cnt))
    nb_ = T(native_op_fb) - nf
    print(f"  {'native TK fine (tier A only, VSA2 kernels + zero-init/delta/finalize)':58s} {nf:8.3f}  {nb_:8.3f}")
    gf, gb = op[("generic", "A|B|C")]
    tf, tb = op[("tk", "A|B|C")]
    print(f"\n  three tiers, TK vs generic:   forward {tf/gf:.3f}x   backward {tb/gb:.3f}x   "
          f"fwd+bwd {(tf+tb)/(gf+gb):.3f}x")
    print(f"  three tiers TK vs native TK (tier A only): forward {tf/nf:.3f}x   backward {tb/nb_:.3f}x   "
          f"fwd+bwd {(tf+tb)/(nf+nb_):.3f}x")


if __name__ == "__main__":
    if not torch.cuda.is_available() or not ext().has_tk():
        sys.exit("needs a GPU and a VSA2_REP_TK=1 build")
    main()
