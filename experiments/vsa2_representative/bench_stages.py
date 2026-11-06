"""Measurement-only breakdown of the full operator's forward, one REP_TK_FWD_CFG per run.

Every row runs the native call _RepAttentionFunction.forward makes, on intermediates built before the
timed region, with CUDA events around a loop and no synchronize inside it. Inputs are bench_compare.py's
deterministic 16K/H16/D128/RM4 draws; the recorded digests can be checked against its result.

  pool         pool_fwd. One native call makes both the router pools and the coarse means, so it belongs
               to neither `route` nor `coarse`.
  route        route_tiers on the pools: router lse and scores, top-k at two budgets and tier-B lists. One
               native call; its parts cannot be timed apart without changing code.
  coarse.meta  _coarse_meta, the coarse branch's CSR.
  coarse.attn  attn_fwd on the coarse means.
  fine         rep_fine_attention on the built selection. This is where the representative tables and
               the flattened tier-B log counts are built, so they are not part of `route`.
  fuse         fuse_fwd, gate and fusion. The forward has no finalize; that runs in the backward.
  total        rep_attention end to end, first and last, so its spread is this run's own A/A floor.

Rows with in_sum=false overlap the parts: fine.tables and fine.kernel are inside `fine`, and route.one_budget
does different work (frac_mid=0: one top-k, no tier-B lists). The operator's own autograd wrapper cannot be
isolated, so it lands in the residual, and `fine` carries a wrapper the operator's path does not.

Stage times are not a performance claim and must not be added to bench_compare's kernel rows. Isolated loops
need not sum to the complete call: stages overlap differently on the stream when they run back to back.

Under every configuration each row keeps its meaning. reads_fwd_cfg says whether it can change with
REP_TK_FWD_CFG. A row with false, such as pool or route, should match another configuration within the floor.
CFG13 and CFG36 are aliases and CFG21/22/23/37 change only the backward, so all their rows are labelled as a repeat of the
CFG1 forward. Compare runs from
one build only, with --compare; numbers from another build, or from CFG1 before its gather changed, do not qualify.

  REP_TK_FWD_CFG=11 <gpu-runner> python bench_stages.py --out stages-cfg11.json
  python bench_stages.py --compare stages-cfg1.json stages-cfg11.json
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

from bench_compare import FORWARD_EQUAL, HERE, active_fwd_cfg, sha256_file

NOTE = "measurement_only: not a performance claim; do not add to kernel rows"
READS_FWD_CFG = {"total", "total.repeat", "fine", "fine.kernel"}
BACKWARD_ONLY_NOTE = "this configuration changes only the backward: this forward row repeats CFG1's"


def compare_stages(a, b):
    """Per-row ratios of two runs, or an error if they are not one build, input, shape and timing setup."""
    for key in ("build", "inputs", "shape", "timing", "hardware"):
        if a[key] != b[key]:
            raise ValueError(f"runs differ in {key}; stage times are only comparable within one build")
    med = {side: {r["id"]: statistics.median(r["ms"]) for r in run["rows"]} for side, run in (("a", a), ("b", b))}
    reads = {r["id"]: r.get("reads_fwd_cfg") for r in a["rows"] + b["rows"]}
    ids = [r["id"] for r in a["rows"]] + [r["id"] for r in b["rows"] if r["id"] not in med["a"]]
    rows = []
    for n in ids:  # a row only one configuration has reads as zero in the other, never as a dropped row
        x, y = med["a"].get(n, 0.0), med["b"].get(n, 0.0)
        rows.append(dict(id=n, reads_fwd_cfg=reads[n], a_ms=x, b_ms=y, b_over_a=y / x if x else None))
    return dict(measured_rows=True, note=NOTE, fwd_cfgs=[a["fwd_cfg"], b["fwd_cfg"]], rows=rows)


def attribute(rows):
    """Derived figures, all labelled unmeasured. A residual outside the floor is reported, never hidden.

    `sum_of_parts` mixes rows from the fine-only and the routed paths, so since the routed operator builds its
    tables from the pooled means it is reported, never asserted.
    """
    median = {r["id"]: statistics.median(r["ms"]) for r in rows}
    totals = [x for r in rows if r["id"].startswith("total") for x in r["ms"]]
    total, parts = statistics.median(totals), [r["id"] for r in rows if r["in_sum"]]
    sum_parts = sum(median[p] for p in parts)
    floor = max(totals) - min(totals)
    return dict(measured=False, note=NOTE, total_ms=total, parts=parts, sum_of_parts_ms=sum_parts,
                residual_ms=total - sum_parts, floor_ms=floor, sum_within_floor=abs(total - sum_parts) <= floor,
                outside_fine_ms=total - median["fine"], floor_method="max - min of the total batches of this run")


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
            res = compare_stages(x, y)
        except ValueError as e:
            sys.exit(str(e))
        print(f"{NOTE}\nCFG{res['fwd_cfgs'][0]} vs CFG{res['fwd_cfgs'][1]}, same build")
        for r in res["rows"]:
            tag = "" if r["reads_fwd_cfg"] else "  does not read the config"
            ratio = "n/a" if r["b_over_a"] is None else f"{r['b_over_a']:.3f}"
            print(f"{r['id']:20s} {r['a_ms']:9.3f} {r['b_ms']:9.3f} {ratio:>7s}{tag}")
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

    from vsa2_rep.ops import (GATE_ACTS, PHASE_ALL, ROW_NORMS, RepTiers, _coarse_meta, ext, rep_attention,
                              rep_fine_attention, route_tiers, tk_tables)

    def digest(t):
        data = t.float() if t.dtype == torch.bfloat16 else t
        return hashlib.sha256(str(t.dtype).encode() + str(tuple(t.shape)).encode()
                              + data.detach().cpu().numpy().tobytes()).hexdigest()

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
    B, H, L, D, rm, R, nb = 1, 16, 16384, 128, 4, 16, 16384 // 64
    torch.manual_seed(seed)  # draw order as in bench_compare.py
    q, k, v = (torch.randn(B, H, L, D, device=dev, dtype=dt).contiguous() for _ in range(3))
    dout = torch.randn_like(q)
    gate = (0.1 * torch.randn(B, H, L, device=dev)).to(dt)
    cnt = torch.full((nb,), 64, dtype=torch.int32, device=dev)
    tiers = RepTiers(0.10, 0.20, rm)
    row_norm, gate_act, max_score = ROW_NORMS["paper"], GATE_ACTS["tanh"], 2 << 30

    # intermediates, all before any timed region
    sel = route_tiers(q, k, cnt, tiers, R=R)
    qr, kr, qc, kc, vc = C.pool_fwd(q, k, v, cnt, R)
    c_cnt, c_ptr, c_idx = _coarse_meta(B, H, nb, qc.shape[2], q.device)
    tab = tk_tables(C, k, v, cnt, sel, rm)
    of, _ = C.rep_tk_fwd(q, k, v, tab, sel.q2k_ptr, sel.q2k_idx, sel.b_ptr, sel.b_idx, sel.bits_ab, cnt, rm, PHASE_ALL, 0)
    oc, _ = C.attn_fwd(qc, kc, vc, c_ptr, c_idx, c_cnt)
    qq, kk, vv, gg = (t.detach().clone().requires_grad_(True) for t in (q, k, v, gate))
    ba, bab = tiers.budgets(nb)
    one, _ = RepTiers(0.10, 0.0, rm).budgets(nb)

    total = lambda: rep_attention(qq, kk, vv, gg, cnt, R=R, tiers=tiers, backend="tk")
    stages = (
        ("total", False, None, total),
        ("pool", True, None, lambda: C.pool_fwd(q, k, v, cnt, R)),
        ("route", True, None, lambda: C.route_tiers(qr, kr, cnt, R, ba, bab, row_norm, max_score)),
        ("route.one_budget", False, "different work: one top-k and no tier-B lists; not a part of route",
         lambda: C.route_tiers(qr, kr, cnt, R, one, one, row_norm, max_score)),
        ("coarse.meta", True, None, lambda: _coarse_meta(B, H, nb, qc.shape[2], q.device)),
        ("coarse.attn", True, None, lambda: C.attn_fwd(qc, kc, vc, c_ptr, c_idx, c_cnt)),
        ("fine", True, None, lambda: rep_fine_attention(qq, kk, vv, cnt, sel, mid_reps=rm, backend="tk")),
        ("fine.tables", False, "inside fine: representative tables and flattened tier-B log counts",
         lambda: tk_tables(C, k, v, cnt, sel, rm)),
        ("fine.kernel", False, "inside fine: rep_tk_fwd with tables prebuilt",
         lambda: C.rep_tk_fwd(q, k, v, tab, sel.q2k_ptr, sel.q2k_idx, sel.b_ptr, sel.b_idx, sel.bits_ab, cnt, rm,
                              PHASE_ALL, 0)),
        ("fuse", True, None, lambda: C.fuse_fwd(of, oc, gate, cnt, gate_act)),
        ("total.repeat", False, "second measurement of total, for the floor", total),
    )
    # Which API each row describes. Since the routed operator builds its representative tables from the pooled
    # means (CFG42 is the default), `fine` and `fine.tables` describe the fine-only entry point and no longer
    # belong to a decomposition of the routed operator; only `total` does. Summing them overstates the operator.
    path = {"fine": "fine_only", "fine.tables": "fine_only", "fine.kernel": "fine_only",
            "total": "routed", "total.repeat": "routed"}
    rows = [dict(id=n, ms=loop(fn), in_sum=in_sum, label="measurement_only", measured=True, note=note or NOTE,
                 reads_fwd_cfg=n in READS_FWD_CFG, describes=path.get(n, "both"),
                 **({"config_note": BACKWARD_ONLY_NOTE} if cfg in FORWARD_EQUAL - {"1", "10"} else {}))
            for n, in_sum, note, fn in stages]
    res = dict(schema="vsa2-rep-stages/1", measurement_only=True, note=NOTE, fwd_cfg=cfg, fwd_cfg_meaning=meaning,
               shape=dict(B=B, H=H, L=L, D=D, NB=nb, RM=rm, R=R, dtype="bf16", blk_cnt="full"),
               timing=dict(warmup=a.warmup, iters=a.iters, batches=a.batches, clock="torch.cuda.Event"),
               hardware=dict(name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0))),
               build=dict(extension_sha256=sha256_file(C.__file__), rep_tk_sha256=sha256_file(HERE / "csrc/rep_tk/rep_tk.cu")),
               inputs=dict(seed=seed, q=digest(q), k=digest(k), v=digest(v), dout=digest(dout), gate=digest(gate)),
               rows=rows, attribution=attribute(rows))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"{NOTE}\nREP_TK_FWD_CFG={cfg}: {meaning}\n{'stage':20s} {'median ms':>10s}  in sum")
    for r in rows:
        print(f"{r['id']:20s} {statistics.median(r['ms']):10.3f}  {'yes' if r['in_sum'] else '-'}")
    t = res["attribution"]
    print(f"derived, not timed: sum of parts {t['sum_of_parts_ms']:.3f}  total {t['total_ms']:.3f}  residual "
          f"{t['residual_ms']:+.3f}  floor {t['floor_ms']:.3f}  within floor: {t['sum_within_floor']}\nwrote {out}")


if __name__ == "__main__":
    main()
