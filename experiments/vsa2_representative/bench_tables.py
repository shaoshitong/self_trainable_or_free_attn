"""Measurement-only: how does `fine.tables` (0.070 ms) split between table build and the tier-B log counts?

The reuse idea — feed the router's pooled K/V into the representative-table build instead of re-reading
K and V (128 MiB) — is only worth a kernel change if the build dominates that row. `rep_tk_build` reads K and
V and writes the swizzled tables; `rep_tk_lcb` only gathers log2 counts through the tier-B list. This times
them apart on the same build, with the same inputs as `bench_stages.py`.

  python bench_tables.py --out tables.json
  python bench_tables.py --compare tables-cfg1.json tables-cfg42.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

from bench_compare import HERE, active_fwd_cfg, sha256_file

NOTE = "measurement_only: not a performance claim; do not add to kernel rows"


def row(name, ms, note=None):
    return dict(id=name, label="measurement_only", note=note or NOTE, measured=True, available=True,
                ms=ms, batch_ms=ms, spread_ms=max(ms) - min(ms))


def compare(a, b):
    for key in ("build", "shape", "timing"):
        if a[key] != b[key]:
            raise ValueError(f"runs differ in {key}")
    med = {side: {r["id"]: statistics.median(r["ms"]) for r in run["rows"]} for side, run in (("a", a), ("b", b))}
    return [dict(id=n, a_ms=med["a"].get(n), b_ms=med["b"].get(n),
                 b_over_a=(med["b"][n] / med["a"][n]) if med["a"].get(n) else None) for n in med["a"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    a = ap.parse_args()
    if a.compare:
        x, y = (json.loads(Path(p).read_text()) for p in a.compare)
        try:
            rows = compare(x, y)
        except ValueError as e:
            sys.exit(str(e))
        print(f"{NOTE}\nCFG{x['fwd_cfg']} vs CFG{y['fwd_cfg']}, same build")
        for r in rows:
            print(f"  {r['id']:24s} {r['a_ms']:.4f} -> {r['b_ms']:.4f}  {r['b_over_a']:.4f}" if r["b_over_a"]
                  else f"  {r['id']:24s} {r['a_ms']:.4f} -> -")
        return
    if not a.out:
        ap.error("--out is required")
    out = Path(a.out)
    if out.exists():
        ap.error(f"{out} exists; results are not overwritten")
    cfg, kind, meaning = active_fwd_cfg()
    if kind != "operator":
        sys.exit(f"REP_TK_FWD_CFG={cfg} does not run all tiers: {meaning}")

    import torch

    from vsa2_rep.ops import RepTiers, ext, route_tiers, tk_tables

    def loop(fn):
        for _ in range(a.warmup):
            fn()
        ms = []
        for _ in range(a.batches):
            torch.cuda.synchronize()
            t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
            t0.record()
            for _ in range(a.iters):
                fn()
            t1.record()
            torch.cuda.synchronize()
            ms.append(t0.elapsed_time(t1) / a.iters)
        return ms

    C = ext()
    if not torch.cuda.is_available() or not C.has_tk():
        sys.exit("needs a GPU and a VSA2_REP_TK=1 build")
    if torch.cuda.device_count() != 1:
        sys.exit(f"needs exactly one visible GPU, got {torch.cuda.device_count()}")

    dev, dt, seed = "cuda", torch.bfloat16, 0
    B, H, L, D, rm, R = 1, 16, 16384, 128, 4, 16
    nb = L // 64
    torch.manual_seed(seed)  # draw order as in bench_compare.py
    q, k, v = (torch.randn(B, H, L, D, device=dev, dtype=dt).contiguous() for _ in range(3))
    cnt = torch.full((nb,), 64, dtype=torch.int32, device=dev)
    sel = route_tiers(q, k, cnt, RepTiers(0.10, 0.20, rm), R=R).with_tier_b()

    tab = C.rep_tk_build(k, v, cnt, rm)
    rows = [
        row("build", loop(lambda: C.rep_tk_build(k, v, cnt, rm)),
            note="rep_tk_build: reads K and V (128 MiB), writes the swizzled tables"),
        row("lcb", loop(lambda: C.rep_tk_lcb(tab[2], sel.b_ptr, sel.b_idx, nb, rm)),
            note="rep_tk_lcb: gathers the flattened tier-B log2 counts through b_idx"),
        row("tables.total", loop(lambda: tk_tables(C, k, v, cnt, sel, rm)),
            note="tk_tables: build + lcb, i.e. the fine.tables row of bench_stages.py"),
    ]
    res = dict(schema="vsa2-rep-tables/1", measurement_only=True, note=NOTE, fwd_cfg=cfg, fwd_cfg_meaning=meaning,
               shape=dict(B=B, H=H, L=L, D=D, NB=nb, RM=rm, R=R),
               timing=dict(warmup=a.warmup, iters=a.iters, batches=a.batches),
               build=dict(rep_tk_sha256=sha256_file(HERE / "csrc/rep_tk/rep_tk.cu")), rows=rows)
    res["attribution"] = dict(measured=False, note=NOTE,
                              build_share=statistics.median(rows[0]["ms"]) / statistics.median(rows[2]["ms"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    for r in rows:
        print(f"{r['id']:20s} {statistics.median(r['ms']):8.4f} ms  spread {r['spread_ms']:.4f}")
    print(f"build share of tk_tables: {res['attribution']['build_share']:.3f}")


if __name__ == "__main__":
    main()
