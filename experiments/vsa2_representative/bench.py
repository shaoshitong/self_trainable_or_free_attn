"""What the representative tiers cost over VSA2's drop-the-rest baseline.

Compares, on one frozen selection so only the attention work differs:
  A only           this implementation with tiers B and C switched off -- the same KV budget VSA2
                   keeps (10% of the pairs, the other 90% dropped). It is still the rep_fine_attention
                   autograd op, NOT VSA2's own kernel; the native forward is reported separately below.
  A + B            + mid_reps representatives for the next 20% of the pairs
  A + B + C        + one block mean for every remaining pair -- every query block now sees the whole
                   sequence, at three resolutions

Tier A does 64 columns per selected block, tier B does mid_reps and tier C does 1, so the added
columns are a small fraction of tier A's: at NB blocks and the default fractions,
  tier A  0.10 * NB^2 * 64 columns,  tier B  0.20 * NB^2 * 4,  tier C  0.70 * NB^2 * 1
i.e. tier B adds 12.5% and tier C 10.9% of tier A's columns.

Timing uses CUDA events around a whole loop; nothing logs or synchronizes per call.
"""
import argparse
import sys

import torch

from vsa2_rep.ops import PHASE_A, PHASE_ALL, PHASE_B, RepTiers, ext, rep_fine_attention, route_tiers


def timeit(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int, default=32768, help="padded sequence length (multiple of 64)")
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--dim", type=int, default=128, choices=(64, 128))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--frac-full", type=float, default=0.10)
    ap.add_argument("--frac-mid", type=float, default=0.20)
    ap.add_argument("--mid-reps", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()

    dev, dt = "cuda", torch.bfloat16
    B, H, L, D, nb = a.batch, a.heads, a.len, a.dim, a.len // 64
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, L, D, device=dev, dtype=dt).contiguous() for _ in range(3))
    blk_cnt = torch.full((nb,), 64, dtype=torch.int32, device=dev)
    tiers = RepTiers(a.frac_full, a.frac_mid, a.mid_reps)
    C = ext()

    print(f"{torch.cuda.get_device_name(0)}  B={B} H={H} L={L} D={D} NB={nb}  bf16")
    print(f"tiers: full {a.frac_full:.0%} (64 tok) | mid {a.frac_mid:.0%} ({a.mid_reps} reps) | "
          f"tail {1 - a.frac_full - a.frac_mid:.0%} (1 rep)\n")

    sel = route_tiers(q, k, blk_cnt, tiers, R=16)
    ba, bab = tiers.budgets(nb)
    cols_a, cols_b, cols_c = ba * 64, (bab - ba) * a.mid_reps, (nb * nb - bab) * 1
    tot = cols_a + cols_b + cols_c
    print(f"columns per (b,h):  A {cols_a:,} ({cols_a/tot:.1%})  B {cols_b:,} ({cols_b/tot:.1%})  "
          f"C {cols_c:,} ({cols_c/tot:.1%})   total {tot/cols_a:.3f}x tier A")
    print(f"KV blocks reachable per query block: tier A {ba/nb:.1f}, all tiers {nb} (the whole sequence)\n")

    qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    dout = torch.randn_like(q)
    rows = []
    for name, ph in (("this impl, A only (VSA2's KV budget)", PHASE_A), ("A + B", PHASE_A | PHASE_B),
                     ("A + B + C (full sequence)", PHASE_ALL)):
        f = timeit(lambda: rep_fine_attention(q, k, v, blk_cnt, sel, mid_reps=a.mid_reps, phases=ph), a.iters)
        bwd = []
        for var in (1, 2):
            def fb(var=var, ph=ph):
                o, _ = rep_fine_attention(qg, kg, vg, blk_cnt, sel, mid_reps=a.mid_reps, phases=ph,
                                          bwd_variant=var)
                o.backward(dout)
                qg.grad = kg.grad = vg.grad = None
            bwd.append(timeit(fb, a.iters) - f)
        rows.append((name, f, bwd[0], bwd[1]))

    base_f, base_b = rows[0][1], rows[0][2]
    print(f"{'':38s} {'forward':>11s} {'bwd v1':>10s} {'bwd v2':>10s} {'fwd x':>7s} {'bwd x':>7s}")
    for name, f, b1, b2 in rows:
        print(f"{name:38s} {f:8.3f} ms {b1:7.3f} ms {b2:7.3f} ms {f/base_f:6.2f}x {b1/base_b:6.2f}x")

    # absolute reference points: the native VSA2 kernels on the same tier-A lists
    nat = timeit(lambda: C.attn_fwd(q, k, v, sel.q2k_ptr, sel.q2k_idx, blk_cnt), a.iters)
    print(f"\nNATIVE VSA2 reference, same tier-A lists (kernel only, no autograd wrapper):")
    print(f"  {'attn_fwd (forward)':36s} {nat:8.3f} ms"
          f"   -> three-tier forward (incl. rep_build) is {rows[2][1]/nat:.2f}x this")
    print(f"  {'of which rep_build':36s} "
          f"{timeit(lambda: C.rep_build(k, v, blk_cnt, a.mid_reps), a.iters):8.3f} ms")
    print("  A full native backward is NOT measured here (VSA2's attn_bwd_q is not bound); prof.py\n"
          "  does measure VSA2's attn_bwd on the tier-A lists, with and without its atomic dQ.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("needs a GPU")
    main()
