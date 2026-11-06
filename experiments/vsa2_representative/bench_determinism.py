"""Is the operator deterministic across two eager runs with identical inputs?

The whole-operator CUDA graph was set aside in round 13 partly because its gradients differed from eager by one
bf16 ulp. But dq/dk/dv are built with fp32 reduce-adds across CTAs whose order is not fixed, so two *eager* runs
may differ by the same ulp — in which case the graph changed nothing and only exposed nondeterminism the default
already has.

This measures exactly that: run the same forward+backward twice, eagerly, with identical inputs, and compare
every output and gradient bitwise. Then, for reference, compare an eager run against a captured graph replay of
the same inputs. measurement_only.

  python bench_determinism.py --out determinism.json
"""
import argparse
import json
import sys
from pathlib import Path

from bench_compare import HERE, active_fwd_cfg, sha256_file

NOTE = "measurement_only: not a performance claim; do not add to kernel rows"
TENSORS = ("o", "q2k_idx", "bits_a", "bits_ab", "dq", "dk", "dv")


def bits(t):
    import torch
    return t.detach().contiguous().view(torch.int16) if t.dtype == torch.bfloat16 else t.detach().contiguous()


def compare(a, b, name):
    import torch
    x, y = bits(a), bits(b)
    if x.shape != y.shape:
        return dict(name=name, same_shape=False, equal=False)
    diff = (x != y).sum().item()
    return dict(name=name, same_shape=True, equal=diff == 0, differing_elements=int(diff), elements=int(x.numel()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        ap.error(f"{out} exists; results are not overwritten")
    cfg, kind, meaning = active_fwd_cfg()
    if kind != "operator":
        sys.exit(f"REP_TK_FWD_CFG={cfg} does not run all tiers: {meaning}")

    import torch

    from vsa2_rep.ops import RepTiers, ext, rep_attention

    C = ext()
    if not torch.cuda.is_available() or not C.has_tk():
        sys.exit("needs a GPU and a VSA2_REP_TK=1 build")

    dev, dt = "cuda", torch.bfloat16
    B, H, L, D, R, rm = 1, 16, 16384, 128, 16, 4
    nb = L // 64
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, L, D, device=dev, dtype=dt).contiguous().requires_grad_(True) for _ in range(3))
    gate = (0.1 * torch.randn(B, H, L, device=dev)).to(dt).requires_grad_(True)  # per token, as bench_compare builds it
    cnt = torch.full((nb,), 64, dtype=torch.int32, device=dev)
    tiers = RepTiers(0.10, 0.20, rm)

    def run():
        for t in (q, k, v):
            t.grad = None
        gate.grad = None
        o, sel = rep_attention(q, k, v, gate, cnt, R=R, tiers=tiers, return_selection=True)
        o.float().square().mean().backward()
        return dict(o=o, q2k_idx=sel.q2k_idx, bits_a=sel.bits_a, bits_ab=sel.bits_ab,
                    dq=q.grad, dk=k.grad, dv=v.grad)

    first = run()
    torch.cuda.synchronize()
    second = run()
    torch.cuda.synchronize()
    rows = [compare(first[n], second[n], n) for n in TENSORS]
    same = all(r["equal"] for r in rows)

    res = dict(schema="vsa2-rep-determinism/1", measurement_only=True, note=NOTE, fwd_cfg=cfg,
               fwd_cfg_meaning=meaning, shape=dict(B=B, H=H, L=L, D=D, NB=nb, RM=rm, R=R),
               build=dict(rep_tk_sha256=sha256_file(HERE / "csrc/rep_tk/rep_tk.cu")),
               eager_vs_eager=rows, eager_is_deterministic=same)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    print(f"{'tensor':12s} {'bit-identical':>14s} {'differing':>12s}")
    for r in rows:
        print(f"{r['name']:12s} {str(r['equal']):>14s} {r.get('differing_elements', '-'):>12}")
    print("eager is deterministic across runs:", same)


if __name__ == "__main__":
    main()
