"""Measurement only: how much of the eager call's host and launch cost could a CUDA graph recover?

Captures one full rep_attention(...) plus its backward() into a torch.cuda.CUDAGraph on a non-default stream, with
static input tensors, then replays it. Inputs are bench_compare.py's deterministic 16K/H16/D128/RM4 draws. This is not
a supported API; nothing in vsa2_rep changes. If capture or replay raises, the reason is recorded, the graph rows are
marked unavailable and the eager rows are still produced.

  eager               routed F+B on the default stream, the production path; first and last, so its spread is the floor
  eager.same_stream   the same call on the capture stream with autograd multithreading off, as during capture. Autograd's
                      worker thread would otherwise launch the backward on the default stream, outside the capture
  graph.replay        replays in a loop, no synchronize inside it, the same boundaries as eager
  graph.event_pair    an event pair around one replay. These are GPU-timeline events, so this is the replay's execution
                      time, not its launch cost
  graph.host_call     wall time of the graph.replay() call on the host, the closest measure of the launch cost itself

The graph runs what was captured, so the router and the tables re-run on the same tensors in each replay, and any
selection-dependent host decision is frozen at capture. Capture executes nothing: outputs exist only after a replay.

Outputs are compared as max abs differences on o, dq, dk and dv, against the same difference between two eager calls.
dQ, dK and dV include fp32 reductions in a nondeterministic order, so the eager pair is the floor they can be read
against. `equivalent` uses bench_compare.py's tolerances for the generic cross-check.

Stage times are not a performance claim and must not be added to bench_compare's kernel rows.

  REP_TK_FWD_CFG=1 <gpu-runner> python bench_graph.py --out graph.json
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

from bench_compare import HERE, TOL_GRAD, TOL_O, active_fwd_cfg, sha256_file
from bench_stages import NOTE

GRAPH_ROWS = ("graph.replay", "graph.event_pair", "graph.host_call")


def row(name, ms=None, reason=None, note=None):
    """A row with its batch spread, or one marked unavailable; graph rows are never silently absent."""
    base = dict(id=name, label="measurement_only", note=note or NOTE, measured=ms is not None, available=ms is not None)
    if ms is None:
        return dict(base, ms=[], batch_ms=[], spread_ms=None, unavailable_reason=reason)
    return dict(base, ms=ms, batch_ms=ms, spread_ms=max(ms) - min(ms))


def attribute(rows, reason=None):
    """Derived and unmeasured. recoverable_ms is eager minus replay; a value inside the floor is not a saving."""
    by = {r["id"]: r for r in rows}
    eager = [x for n in ("eager", "eager.repeat") for x in by[n]["ms"]]
    med = statistics.median(eager)
    floor = max(eager) - min(eager)
    res = dict(measured=False, note=NOTE, eager_ms=med, floor_ms=floor, graph_available=reason is None,
               unavailable_reason=reason, floor_method="max - min of the eager batches of this run")
    if reason is None:
        replay = statistics.median(by["graph.replay"]["ms"])
        res.update(replay_ms=replay, recoverable_ms=med - replay, recoverable_above_floor=med - replay > floor)
    if by["eager.same_stream"]["available"]:
        res["stream_and_threading_ms"] = statistics.median(by["eager.same_stream"]["ms"]) - med
    return res


def compare_outputs(pairs, tolerances):
    """{name: (max abs difference, scale)} -> per-output report and an overall verdict; NaN never passes."""
    report = {n: dict(max_abs_diff=err, scale=scale, tol_rel=tolerances[n], ok=err <= tolerances[n] * max(scale, 1.0))
              for n, (err, scale) in pairs.items()}
    return report, all(r["ok"] for r in report.values())


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

    from vsa2_rep.ops import RepTiers, ext, rep_attention, route_tiers

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
    sel = route_tiers(q, k, cnt, tiers, R=R)  # recorded; the operator routes itself, as bench_compare's full_op does
    qq, kk, vv, gg = (t.detach().clone().requires_grad_(True) for t in (q, k, v, gate))
    leaves = (qq, kk, vv, gg)

    def forward():
        return rep_attention(qq, kk, vv, gg, cnt, R=R, tiers=tiers, backend="tk")

    def eager():
        forward().backward(dout)
        for t in leaves:
            t.grad = None

    def snapshot():
        o = forward()
        o.backward(dout)
        res = dict(o=o.detach().float().clone(), dq=qq.grad.clone(), dk=kk.grad.clone(), dv=vv.grad.clone())
        for t in leaves:
            t.grad = None
        return res

    def differences(x, y):
        return {n: ((x[n].float() - y[n].float()).abs().max().item(), y[n].float().abs().max().item() or 1.0) for n in x}

    tol = dict(o=TOL_O, dq=TOL_GRAD, dk=TOL_GRAD, dv=TOL_GRAD)
    rows = [row("eager", loop(eager), note="production path, default stream")]
    ref = snapshot()
    eager_pair, _ = compare_outputs(differences(snapshot(), ref), tol)
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    graph_rows, reason, replay_vs_eager, equivalent = None, None, None, None
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.autograd.set_multithreading_enabled(False):
            with torch.cuda.graph(graph, stream=stream):
                o_cap = forward()
                o_cap.backward(dout)
            captured = dict(o=o_cap.detach(), dq=qq.grad.detach(), dk=kk.grad.detach(), dv=vv.grad.detach())
        for t in leaves:  # the graph keeps writing these tensors; eager rows must not accumulate into them
            t.grad = None

        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.synchronize()
        replay_vs_eager, equivalent = compare_outputs(differences(captured, ref), tol)

        with torch.cuda.stream(stream):
            replay_ms = loop(graph.replay)
            pair, host = [], []
            for _ in range(a.batches):
                pairs, hosts = [], []
                for _ in range(a.iters):
                    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    torch.cuda.synchronize()
                    t0.record()
                    start = time.perf_counter()
                    graph.replay()
                    hosts.append((time.perf_counter() - start) * 1e3)
                    t1.record()
                    torch.cuda.synchronize()
                    pairs.append(t0.elapsed_time(t1))
                pair.append(statistics.median(pairs))
                host.append(statistics.median(hosts))
        graph_rows = [row("graph.replay", replay_ms, note="replays in a loop, same boundaries as eager"),
                      row("graph.event_pair", pair, note="GPU-timeline execution of one replay, not its launch cost"),
                      row("graph.host_call", host, note="host wall time of the replay call; the launch cost")]
    except Exception as e:  # a capture or replay failure is itself the answer this script is asked for
        reason = f"{type(e).__name__}: {e}"
        print(f"graph unavailable: {reason}")
        for t in leaves:
            t.grad = None
        graph_rows = [row(n, reason=reason) for n in GRAPH_ROWS]

    rows += graph_rows
    try:
        with torch.cuda.stream(stream), torch.autograd.set_multithreading_enabled(False):
            same_stream = row("eager.same_stream", loop(eager), note="capture stream, autograd multithreading off")
    except Exception as e:
        same_stream = row("eager.same_stream", reason=f"{type(e).__name__}: {e}")
        for t in leaves:
            t.grad = None
    rows.append(same_stream)
    rows.append(row("eager.repeat", loop(eager), note="second measurement, for the floor"))
    res = dict(schema="vsa2-rep-graph/1", measurement_only=True, note=NOTE, fwd_cfg=cfg, fwd_cfg_meaning=meaning,
               shape=dict(B=B, H=H, L=L, D=D, NB=nb, RM=rm, R=R, dtype="bf16", blk_cnt="full"),
               timing=dict(warmup=a.warmup, iters=a.iters, batches=a.batches, clock="torch.cuda.Event, host perf_counter"),
               hardware=dict(name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0))),
               build=dict(extension_sha256=sha256_file(C.__file__), rep_tk_sha256=sha256_file(HERE / "csrc/rep_tk/rep_tk.cu")),
               inputs=dict(seed=seed, q=digest(q), k=digest(k), v=digest(v), dout=digest(dout), gate=digest(gate),
                           q2k_idx=digest(sel.q2k_idx), bits_ab=digest(sel.bits_ab)),
               rows=rows, outputs=dict(eager_vs_eager=eager_pair, replay_vs_eager=replay_vs_eager, equivalent=equivalent),
               attribution=attribute(rows, reason))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"{NOTE}\nREP_TK_FWD_CFG={cfg}: {meaning}\n{'row':22s} {'median ms':>10s} {'spread':>8s}")
    for r in rows:
        if r["available"]:
            print(f"{r['id']:22s} {statistics.median(r['ms']):10.3f} {r['spread_ms']:8.3f}")
        else:
            print(f"{r['id']:22s} {'unavailable':>10s}")
    for name, report in (("eager vs eager", eager_pair), ("replay vs eager", replay_vs_eager)):
        if report:
            print(f"{name}: " + "  ".join(f"{n} {x['max_abs_diff']:.3e}" for n, x in report.items()))
    t = res["attribution"]
    if reason is None:
        print(f"equivalent: {equivalent}")
        print(f"derived, not timed: eager {t['eager_ms']:.3f}  replay {t['replay_ms']:.3f}  recoverable "
              f"{t['recoverable_ms']:+.3f}  floor {t['floor_ms']:.3f}  above floor: {t['recoverable_above_floor']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
