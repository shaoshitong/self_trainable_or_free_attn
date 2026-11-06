"""Per-kernel cost breakdown, so every number has a measured scope.

The headline bench times an autograd op; this times each native launch on its own, with the
representative tables prebuilt, so the three-tier attention kernel can be compared against the VSA2
kernel it extends on equal terms (same inputs, same tier-A lists, same selection).

Sections:
  1. forward kernels, isolated: rep_build, rep_attn_fwd per phase mask, VSA2 attn_fwd
  2. backward kernels, isolated: rep_attn_bwd_q per phase mask, tier_a_bwd_kv, rep_unpool, row_dot
  3. wrapper overhead: the autograd op minus the sum of its kernels
  4. torch profiler, per-kernel self CUDA time over one full forward+backward

Timing is CUDA events around a loop; nothing logs or synchronizes per call.
"""
import argparse
import sys

import torch

from vsa2_rep.ops import (PHASE_A, PHASE_ALL, PHASE_B, PHASE_C, RepTiers, ext, rep_fine_attention,
                          route_tiers, _rep_bwd, _rep_fwd)


def timeit(fn, iters, warmup=10):
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
    ap.add_argument("--profiler", action="store_true", help="also run the torch profiler breakdown")
    a = ap.parse_args()

    dev, dt = "cuda", torch.bfloat16
    B, H, L, D, nb, rm = 1, a.heads, a.len, a.dim, a.len // 64, a.mid_reps
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, L, D, device=dev, dtype=dt).contiguous() for _ in range(3))
    blk_cnt = torch.full((nb,), 64, dtype=torch.int32, device=dev)
    tiers = RepTiers(0.10, 0.20, rm)
    C = ext()
    sel = route_tiers(q, k, blk_cnt, tiers, R=16)
    ba, bab = tiers.budgets(nb)
    # Budgets are pairs per (b,h); per query block that is /nb. A tile packs 64 columns, and a tier's
    # last tile is only partly filled, so the honest work unit is the *integer* tile count: tier A is
    # one block per tile, tier B packs 64/RM blocks and tier C packs 64.
    import math
    tiles_a = ba / nb
    tiles_b = math.ceil((bab - ba) / nb * rm / 64.0)
    tiles_c = math.ceil((nb * nb - bab) / nb / 64.0)
    print(f"{torch.cuda.get_device_name(0)}  B={B} H={H} L={L} D={D} NB={nb} RM={rm} bf16, {a.iters} iters")
    print(f"tiles per query block: A {tiles_a:.1f}  B {tiles_b:.2f}  C {tiles_c:.2f}  "
          f"(total {(tiles_a+tiles_b+tiles_c)/tiles_a:.3f}x tier A)\n")

    tab = C.rep_build(k, v, blk_cnt, rm)
    kp, ki, qp, qi = sel.k2q_ptr, sel.k2q_idx, sel.q2k_ptr, sel.q2k_idx
    o, lse = C.rep_attn_fwd(q, k, v, *tab, qp, qi, sel.bits_a, sel.bits_ab, blk_cnt, rm, PHASE_ALL, 0)
    dout = torch.randn_like(q)
    delta = C.row_dot(dout, o, blk_cnt)
    f32 = dict(device=dev, dtype=torch.float32)
    zm = [torch.zeros(t.shape, **f32) for t in (tab[0], tab[1], tab[3], tab[4])]

    print("--- 0. preparation / routing (outside the attention timings below)")
    t_pool = timeit(lambda: C.pool_fwd(q, k, v, blk_cnt, 16), a.iters)
    t_r1 = timeit(lambda: route_tiers(q, k, blk_cnt, RepTiers(0.10, 0.0, rm), R=16), a.iters)
    t_r2 = timeit(lambda: route_tiers(q, k, blk_cnt, RepTiers(0.10, 0.20, rm), R=16), a.iters)
    print(f"  pool_fwd (router pools + block means, shared) {t_pool:8.3f} ms")
    print(f"  route_tiers ONE budget (frac_mid=0, = VSA2's own routing) {t_r1:8.3f} ms [incl. pool_fwd]")
    print(f"  route_tiers TWO budgets (frac_mid=0.20, this work)        {t_r2:8.3f} ms [incl. pool_fwd]")
    print(f"  -> the second global top-k pass the tiers need: {t_r2-t_r1:+.3f} ms")

    print("\n--- 1. forward kernels, isolated (tables prebuilt)")
    t_native = timeit(lambda: C.attn_fwd(q, k, v, qp, qi, blk_cnt), a.iters)
    t_build = timeit(lambda: C.rep_build(k, v, blk_cnt, rm), a.iters)
    rows = []
    for name, ph in (("A", PHASE_A), ("A|B", PHASE_A | PHASE_B), ("A|B|C", PHASE_ALL),
                     ("B|C", PHASE_B | PHASE_C)):
        rows.append((name, timeit(lambda ph=ph: C.rep_attn_fwd(q, k, v, *tab, qp, qi, sel.bits_a,
                                                               sel.bits_ab, blk_cnt, rm, ph, 0), a.iters)))
    print(f"  VSA2 attn_fwd (native, tier-A lists)      {t_native:8.3f} ms   1.000x")
    ntiles = {"A": tiles_a, "A|B": tiles_a + tiles_b, "A|B|C": tiles_a + tiles_b + tiles_c,
              "B|C": tiles_b + tiles_c}
    for name, t in rows:
        exp = ntiles[name] / tiles_a
        print(f"  rep_attn_fwd phases={name:5s}                 {t:8.3f} ms   {t/t_native:.3f}x"
              f"   ({ntiles[name]:5.1f} tiles, predict {exp:.3f}x)")
    per_a = rows[0][1] / tiles_a
    per_bc = rows[3][1] / (tiles_b + tiles_c)
    print(f"  per-tile: contiguous (A) {per_a*1e3:.1f} us, gathered (B|C) {per_bc*1e3:.1f} us"
          f"  -> a gathered tile costs {per_bc/per_a:.2f}x a contiguous one")
    print(f"  rep_build (tables, once per forward)      {t_build:8.3f} ms   "
          f"{t_build/t_native:.3f}x of native forward")

    print("\n--- 1b. what a gathered tile pays over a contiguous one (diagnostic; results wrong)")
    BC = PHASE_B | PHASE_C
    d = {}
    for diag, what in ((0, "full (scattered gather + log2 bias)"), (1, "gather, no bias reads"),
                       (2, "contiguous rows, no bias"),
                       (3, "+ contiguous loader (no index read)")):
        d[diag] = timeit(lambda g=diag: C.rep_attn_fwd(q, k, v, *tab, qp, qi, sel.bits_a, sel.bits_ab,
                                                       blk_cnt, rm, BC, g), a.iters)
        print(f"  phases=B|C  {what:36s} {d[diag]:8.3f} ms  ({d[diag]/(tiles_b+tiles_c)*1e3:5.1f} us/tile)")
    ptile_a = rows[0][1] / tiles_a
    print(f"  tier-A contiguous tile, for comparison:              {ptile_a*1e3:5.1f} us/tile")
    print(f"  -> softmax bias reads from shared memory : {d[0]-d[1]:+.3f} ms "
          f"({(d[0]-d[1])/d[0]:.0%} of the B|C pass)")
    print(f"  -> scattered row gather + expand_tile    : {d[1]-d[2]:+.3f} ms "
          f"({(d[1]-d[2])/d[0]:.0%})")
    print(f"  -> the gather loader's per-chunk index read: {d[2]-d[3]:+.3f} ms "
          f"({(d[2]-d[3])/d[0]:.0%})")
    print(f"  -> everything else (MMA, softmax, stream, bitmap scan): {d[3]:.3f} ms "
          f"({d[3]/d[0]:.0%}), i.e. {d[3]/(tiles_b+tiles_c)/ptile_a:.2f}x a tier-A tile -- NOT further"
          f"\n     decomposed: skipping the tier scan changes the tile count, so it cannot be ablated.")
    # Fit incremental per-tile costs, which is the scope-correct comparison: adding tiers B|C to a
    # tier-A stream costs (A|B|C - A), spread over the gathered tiles it added.
    inc = rows[2][1] - rows[0][1]
    print(f"  incremental: adding B|C to an A stream costs {inc:.3f} ms over {tiles_b+tiles_c:.0f} tiles"
          f" = {inc/(tiles_b+tiles_c)*1e3:.1f} us/tile, {inc/(tiles_b+tiles_c)/ptile_a:.2f}x a tier-A tile")

    print("\n--- 2. backward kernels, isolated")
    t_rowdot = timeit(lambda: C.row_dot(dout, o, blk_cnt), a.iters)
    t_tiera = timeit(lambda: C.tier_a_bwd_kv(q, k, v, dout, lse, delta, kp, ki, blk_cnt, None), a.iters)
    dka, dva = C.tier_a_bwd_kv(q, k, v, dout, lse, delta, kp, ki, blk_cnt, None)
    t_unpool = timeit(lambda: C.rep_unpool(dka, dva, *zm, blk_cnt, k, rm), a.iters)
    bq = []
    for name, ph in (("A", PHASE_A), ("A|B", PHASE_A | PHASE_B), ("A|B|C", PHASE_ALL)):
        bq.append((name, timeit(lambda ph=ph: C.rep_attn_bwd_q(q, k, v, *tab, dout, lse, delta, qp, qi,
                                                               sel.bits_a, sel.bits_ab, blk_cnt, *zm, rm,
                                                               ph, None, None, 0), a.iters)))
    for name, t in bq:
        print(f"  rep_attn_bwd_q phases={name:5s} (dQ + rep dK/dV) {t:8.3f} ms")
    print(f"  tier_a_bwd_kv (VSA2 attn_bwd, exact dK/dV)     {t_tiera:8.3f} ms")
    print(f"  rep_unpool                                     {t_unpool:8.3f} ms")
    print(f"  row_dot (delta)                                {t_rowdot:8.3f} ms")
    dqa = torch.zeros(q.shape, **f32)
    t_tiera_dq = timeit(lambda: C.tier_a_bwd_kv(q, k, v, dout, lse, delta, kp, ki, blk_cnt, dqa), a.iters)
    t_bq_bc = timeit(lambda: C.rep_attn_bwd_q(q, k, v, *tab, dout, lse, delta, qp, qi, sel.bits_a,
                                              sel.bits_ab, blk_cnt, *zm, rm, PHASE_B | PHASE_C, None,
                                              dqa, 0), a.iters)
    print(f"  tier_a_bwd_kv + atomic tier-A dQ                {t_tiera_dq:8.3f} ms")
    print(f"  rep_attn_bwd_q phases=B|C  (+ adds that dQ)     {t_bq_bc:8.3f} ms")
    v2 = bq[2][1] + t_tiera + t_unpool + t_rowdot
    v1 = t_tiera_dq + t_bq_bc + t_unpool + t_rowdot
    print(f"  => backward variant 2 (dQ in registers, all tiers): {v2:7.3f} ms")
    print(f"  => backward variant 1 (tier-A dQ via atomics)     : {v1:7.3f} ms   {v1/v2:.3f}x")

    print("\n--- 2b. where the gathered tiers' backward time goes (diagnostic flag; dK/dV wrong)")
    bcs = {}
    for diag, what in ((0, "full (MMAs + atomic scatter)"), (1, "rep MMAs, no atomic scatter"),
                       (2, "neither (dQ only)")):
        bcs[diag] = timeit(lambda d=diag: C.rep_attn_bwd_q(q, k, v, *tab, dout, lse, delta, qp, qi,
                                                           sel.bits_a, sel.bits_ab, blk_cnt, *zm, rm,
                                                           PHASE_B | PHASE_C, None, None, d), a.iters)
        print(f"  phases=B|C  {what:32s} {bcs[diag]:8.3f} ms")
    print(f"  -> atomic scatter of the rep tables: {bcs[0]-bcs[1]:+.3f} ms "
          f"({(bcs[0]-bcs[1])/bcs[0]:.0%} of the B|C pass)")
    print(f"  -> the two representative MMAs     : {bcs[1]-bcs[2]:+.3f} ms "
          f"({(bcs[1]-bcs[2])/bcs[0]:.0%})")
    print(f"  -> dQ over the same gathered tiles  : {bcs[2]:.3f} ms "
          f"({bcs[2]/bcs[0]:.0%})")

    print("\n--- 3. wrapper overhead (autograd op vs the sum of its kernels)")
    for name, ph, kern in (("A", PHASE_A, t_build + rows[0][1]), ("A|B|C", PHASE_ALL, t_build + rows[2][1])):
        tot = timeit(lambda ph=ph: rep_fine_attention(q, k, v, blk_cnt, sel, mid_reps=rm, phases=ph), a.iters)
        print(f"  rep_fine_attention phases={name:5s}  {tot:8.3f} ms   kernels {kern:7.3f} ms   "
              f"python/alloc {tot-kern:+7.3f} ms")

    print("\n--- 3b. scopes, forward per call (routing included or not)")
    o_fine = timeit(lambda: rep_fine_attention(q, k, v, blk_cnt, sel, mid_reps=rm), a.iters)
    print(f"  fine-only, frozen selection (no routing)       {o_fine:8.3f} ms")
    print(f"  fine-only + two-budget routing                 {o_fine + t_r2:8.3f} ms"
          f"  (routing is {t_r2/(o_fine+t_r2):.0%})")
    from vsa2_rep.ops import rep_attention
    gate = torch.zeros(B, H, L, device=dev, dtype=dt)
    t_full = timeit(lambda: rep_attention(q, k, v, gate, blk_cnt, R=16, tiers=tiers), a.iters)
    print(f"  full operator (routing + coarse + gate + fine) {t_full:8.3f} ms"
          f"  -> router + coarse + gate = {t_full-o_fine:+.3f} ms")

    if a.profiler:
        print("\n--- 4. torch profiler, self CUDA time over one forward+backward (phases=A|B|C)")
        qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        for _ in range(3):
            oo, _ = rep_fine_attention(qg, kg, vg, blk_cnt, sel, mid_reps=rm)
            oo.backward(dout)
            qg.grad = kg.grad = vg.grad = None
        torch.cuda.synchronize()
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            oo, _ = rep_fine_attention(qg, kg, vg, blk_cnt, sel, mid_reps=rm)
            oo.backward(dout)
            torch.cuda.synchronize()
        # Only device kernels: the autograd-function and profiler-internal rows carry the device time
        # of their children, so summing every row would double count.
        ka = [e for e in prof.key_averages() if e.self_device_time_total > 0 and e.key.startswith("void ")]
        tot = sum(e.self_device_time_total for e in ka)
        print(f"  {'kernel':62s} {'self CUDA':>11s}  {'share':>7s}")
        for e in sorted(ka, key=lambda e: -e.self_device_time_total):
            print(f"  {e.key[:62]:62s} {e.self_device_time_total/1e3:8.3f} ms "
                  f"{e.self_device_time_total/tot:7.1%}")
        print(f"  {'sum of device kernels (fwd + bwd)':62s} {tot/1e3:8.3f} ms")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("needs a GPU")
    main()
