"""Measurement-only: can `route_tiers` alone be captured in a CUDA graph, and does it pay?

`route` is 0.262 ms of the 1.29 ms full forward, and it is a chain of eighteen small kernels over a score
matrix of only 4 MiB — launch latency, not arithmetic. The full-operator graph probe showed that copying fresh
q/k/v into the captured buffers costs more than the graph saves (3 x 64 MiB). Routing's inputs are the *pooled*
q/k, which are 1/16 the size, so a graph that covers only the route chain may pay where the whole-operator one
did not.

Eager and replay are compared on the same pooled tensors; the replay rows copy fresh pooled inputs into the
captured buffers first, which is what a caller with new data does. Every row is measurement_only.

  python bench_route_graph.py --out route-graph.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from bench_compare import HERE, active_fwd_cfg, sha256_file

NOTE = "measurement_only: not a performance claim; do not add to kernel rows"

ROWS = ("route.eager", "route.replay", "route.replay_with_copy", "route.graph_host_call")


def row(name, ms=None, reason=None, note=None):
    base = dict(id=name, label="measurement_only", note=note or NOTE, measured=ms is not None, available=ms is not None)
    if ms is None:
        return dict(base, ms=[], batch_ms=[], spread_ms=None, unavailable_reason=reason)
    return dict(base, ms=ms, batch_ms=ms, spread_ms=max(ms) - min(ms))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        ap.error(f"{out} exists; results are not overwritten")
    cfg, kind, meaning = active_fwd_cfg()
    if kind != "operator":
        sys.exit(f"REP_TK_FWD_CFG={cfg} does not run all tiers: {meaning}")

    import torch

    from vsa2_rep.ops import ROW_NORMS, RepTiers, ext

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
    cnt = torch.full((nb,), 64, dtype=torch.int32, device=dev)
    qr, kr, _, _, _ = C.pool_fwd(q, k, v, cnt, R)
    qr2, kr2, _, _, _ = C.pool_fwd(q.flip(2).contiguous(), k.flip(2).contiguous(), v, cnt, R)  # different pooled input
    ba, bab = RepTiers(0.10, 0.20, rm).budgets(nb)
    norm = ROW_NORMS["paper"]

    def run_route(x, y):
        return C.route_tiers(x, y, cnt, R, ba, bab, norm, 2 << 30)

    eager_out = run_route(qr, kr)
    torch.cuda.synchronize()
    rows = [row("route.eager", loop(lambda: run_route(qr, kr)), note="production path, default stream")]

    stream = torch.cuda.Stream()
    graph_rows, reason = None, None
    try:
        capture_out = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            for _ in range(3):
                run_route(qr, kr)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            capture_out = run_route(qr, kr)
        replay_out = [t.detach() for t in capture_out]
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.synchronize()

        def diffs(xs, ys, names):
            return {n: float((x.float() - y.float()).abs().max().item()) for n, x, y in zip(names, xs, ys)}

        names = ("q2k_ptr", "q2k_idx", "k2q_ptr", "k2q_idx", "bits_a", "bits_ab", "b_ptr", "b_idx")
        d_replay = diffs(replay_out, eager_out, names)

        with torch.cuda.stream(stream):
            replay_ms = loop(graph.replay)

            def replay_with_copy():
                qr.copy_(qr2)
                kr.copy_(kr2)
                graph.replay()

            copy_ms = loop(replay_with_copy)
            host = []
            for _ in range(a.batches):
                hs = []
                for _ in range(a.iters):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    graph.replay()
                    hs.append((time.perf_counter() - t) * 1e3)
                    torch.cuda.synchronize()
                host.append(statistics.median(hs))
        torch.cuda.synchronize()
        graph_rows = [row("route.replay", replay_ms, note="replays in a loop, same boundaries as eager"),
                      row("route.replay_with_copy", copy_ms,
                          note="fresh pooled inputs (8 MiB) copied into the captured buffers each step, then replay"),
                      row("route.graph_host_call", host, note="host wall time of one replay call")]

        # bitmaps must be bit-identical between eager and replay, and the replay must track changed inputs
        qr.copy_(qr2)
        kr.copy_(kr2)
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.synchronize()
        changed = [t.detach() for t in capture_out]
        eager_changed = run_route(qr2, kr2)
        torch.cuda.synchronize()
        d_changed = diffs(changed, eager_changed, names)
    except Exception as e:  # a capture failure is itself the answer this script was asked for
        reason = f"{type(e).__name__}: {e}"
        print(f"route graph unavailable: {reason}")
        graph_rows = [row(n, reason=reason) for n in ROWS[1:]]
        d_replay = d_changed = None
    rows += graph_rows

    by = {r["id"]: r for r in rows}
    eager_ms = statistics.median(by["route.eager"]["ms"])
    floor = max(by["route.eager"]["ms"]) - min(by["route.eager"]["ms"])
    res = dict(schema="vsa2-rep-route-graph/1", measurement_only=True, note=NOTE, fwd_cfg=cfg,
               fwd_cfg_meaning=meaning,
               shape=dict(B=B, H=H, L=L, D=D, NB=nb, RM=rm, R=R), timing=dict(warmup=a.warmup, iters=a.iters,
                                                                             batches=a.batches),
               build=dict(rep_tk_sha256=sha256_file(HERE / "csrc/rep_tk/rep_tk.cu")),
               rows=rows, outputs={"replay_vs_eager": d_replay, "replay_vs_eager_changed_input": d_changed},
               attribution=dict(measured=False, note=NOTE, eager_ms=eager_ms, floor_ms=floor,
                                graph_available=reason is None, unavailable_reason=reason,
                                replay_ms=(statistics.median(by["route.replay"]["ms"])
                                           if by["route.replay"]["available"] else None),
                                replay_with_copy_ms=(statistics.median(by["route.replay_with_copy"]["ms"])
                                                     if by["route.replay_with_copy"]["available"] else None),
                                floor_method="max - min of the eager batches of this run"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    print(f"{'row':28s} {'median ms':>10s} {'spread':>8s}")
    for r in rows:
        if r["available"]:
            print(f"{r['id']:28s} {statistics.median(r['ms']):10.3f} {r['spread_ms']:8.3f}")
        else:
            print(f"{r['id']:28s} unavailable: {r['unavailable_reason']}")
    if d_replay:
        print("replay vs eager max|diff|:", {k: f"{v:.3g}" for k, v in d_replay.items()})
    if d_changed:
        print("replay vs eager (changed input) max|diff|:", {k: f"{v:.3g}" for k, v in d_changed.items()})
    print(json.dumps({k: v for k, v in res["attribution"].items() if k not in ("note",)}, indent=1))


if __name__ == "__main__":
    main()
