"""Measurement-only breakdown of one backward() call, one REP_TK_FWD_CFG per run.

Each step row comes from a real backward(), not a replay. For that call only, the operator module's handle on the
native extension is replaced by a proxy that records a CUDA event before and after every native call. The kernels,
their arguments and their order do not change. There is no synchronize inside a timed region: each sample finishes
its forward and synchronizes before its first event, and synchronizes again after its last one. Inputs are
bench_compare.py's deterministic 16K/H16/D128/RM4 draws; the recorded digests can be checked against its result.

  call.X        between the events around native call X, which includes its launch and any allocation inside it
  gap.X->Y      everything Python does between two native calls. torch.zeros cannot be split from its allocator,
                nor one zeros call from the next, so a zeroing gap is the closest honest grouping for table zeroing
  total         backward() with only two events, first and last; the spread of its batches is this run's floor
  isolated.*    the three tier kernels back to back with fixed intermediates, as in bench_compare's kernel rows
  context.*     forward and F+B loops without any synchronize, for the F+B - F estimate
  event_pair    two consecutive events on an idle stream

fine runs rep_fine_attention on the built selection and full runs the routed rep_attention. full has no row_dot,
because fuse_bwd produces delta, and it adds the coarse branch's fuse_bwd and attn_bwd.

These are GPU-timeline intervals. Host work appears only if the stream has drained while it runs, so host cost
behind a long kernel reads as zero. total has idle gaps from its synchronizes that the context loops do not have,
so part of the distance between total and isolated kernels may come from how each is measured, not from a step.

Stage times are not a performance claim and must not be added to bench_compare's kernel rows.

  REP_TK_FWD_CFG=1 <gpu-runner> python bench_bwd_stages.py --out bwd-stages.json
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

from bench_compare import HERE, active_fwd_cfg, sha256_file
from bench_stages import NOTE

ZEROING_GAPS = {
    "gap.row_dot->tk_native_bwd": "fine: torch.zeros of qg, kg, vg, with allocator",
    "gap.attn_bwd->tk_native_bwd": "full: torch.zeros of qg, kg, vg, with allocator",
    "gap.fuse_bwd->attn_bwd": "full: torch.zeros of the coarse dq, with allocator",
    "gap.tk_native_bwd->rep_tk_bwd_b": "torch.zeros of dk_mid, dv_mid, with allocator",
    "gap.rep_tk_bwd_b->rep_tk_bwd_c": "torch.zeros of kgt, vgt, with allocator",
}
OTHER_GAPS = {
    "gap.start->row_dot": "dout.contiguous and _row_mask",
    "gap.start->fuse_bwd": "dout.contiguous",
    "gap.rep_tk_bwd_c->finalize": "full: adds the coarse block gradients into kgt, vgt; fine: nothing",
    "gap.finalize->rep_unpool": "nothing between the calls",
    "gap.rep_unpool->end": "return to autograd and gradient accumulation",
}
TIER_KERNELS = ("call.tk_native_bwd", "call.rep_tk_bwd_b", "call.rep_tk_bwd_c")


def intervals(labels):
    """Row names of consecutive marks; a repeated name gets #2. `labels` are start, begin:x, end:x, ..., end."""
    bare = lambda m: m.split(":", 1)[-1]
    names = []
    for a, b in zip(labels, labels[1:]):
        name = f"call.{bare(a)}" if a == f"begin:{bare(b)}" and b.startswith("end:") else f"gap.{bare(a)}->{bare(b)}"
        names.append(name if name not in names else f"{name}#{names.count(name) + 1}")
    return names


def batches(samples, per_batch):
    """Median of each consecutive batch of single-call samples."""
    return [statistics.median(samples[i:i + per_batch]) for i in range(0, len(samples), per_batch)]


def attribute(rows, scope):
    """Derived figures for one scope, all unmeasured. A residual outside the floor is reported, never hidden."""
    own = {r["id"].split(".", 1)[1]: r for r in rows if r["id"].startswith(f"{scope}.")}
    med = {n: statistics.median(r["ms"]) for n, r in own.items()}
    totals = [own[n] for n in ("total", "total.repeat")]
    total = statistics.median(x for r in totals for x in r["ms"])
    per_batch = [x for r in totals for x in r["batch_medians_ms"]]
    steps = [n for n in med if n.startswith(("call.", "gap."))]
    sum_steps = sum(med[n] for n in steps)
    floor = max(per_batch) - min(per_batch)
    kernels = sum(med[n] for n in TIER_KERNELS)
    return dict(measured=False, note=NOTE, total_ms=total, steps=steps, sum_of_steps_ms=sum_steps,
                residual_ms=total - sum_steps, floor_ms=floor, sum_within_floor=abs(total - sum_steps) <= floor,
                tier_kernels_in_place_ms=kernels, outside_tier_kernels_ms=total - kernels,
                table_zeroing_and_allocator_ms=sum(med.get(n, 0.0) for n in ZEROING_GAPS),
                floor_method="max - min of the total batch medians of this run")


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

    import vsa2_rep.ops as ops
    from vsa2_rep.ops import PHASE_ALL, RepTiers, ext, rep_attention, rep_fine_attention, route_tiers, tk_tables

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

    class Recorder:
        """Stands in for the extension during one backward(); events are preallocated and reused."""

        def __init__(self, native):
            self.native, self.pool, self.labels = native, [torch.cuda.Event(True) for _ in range(64)], []

        def mark(self, label):
            self.pool[len(self.labels)].record()
            self.labels.append(label)

        def __getattr__(self, name):
            fn = getattr(self.native, name)

            def call(*args):
                self.mark(f"begin:{name}")
                res = fn(*args)
                self.mark(f"end:{name}")
                return res
            return call

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
    sel = route_tiers(q, k, cnt, tiers, R=R)
    tab = tk_tables(C, k, v, cnt, sel, rm)
    qq, kk, vv, gg = (t.detach().clone().requires_grad_(True) for t in (q, k, v, gate))
    rec = Recorder(C)

    def row(name, ms, per_batch=None, in_sum=False, note=None):
        medians = batches(ms, per_batch) if per_batch else ms
        return dict(id=name, ms=ms, batch_medians_ms=medians, spread_ms=max(medians) - min(medians), in_sum=in_sum,
                    label="measurement_only", measured=True, note=note or NOTE)

    def total(graph, leaves):
        """Same boundaries as bench_compare's b_direct."""
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        samples = []
        for i in range(a.warmup + a.iters * a.batches):
            o = graph()
            torch.cuda.synchronize()
            begin.record()
            o.backward(dout)
            end.record()
            torch.cuda.synchronize()
            if i >= a.warmup:
                samples.append(begin.elapsed_time(end))
            del o
            for t in leaves:
                t.grad = None
        return samples

    def steps(graph, leaves):
        """One real backward() per sample with an event around each native call."""
        names, samples = None, {}
        for i in range(a.warmup + a.iters * a.batches):
            o = graph()
            torch.cuda.synchronize()
            rec.labels = []
            rec.mark("start")
            ops._C = rec
            try:
                o.backward(dout)
            finally:
                ops._C = C
            rec.mark("end")
            torch.cuda.synchronize()
            now = intervals(rec.labels)
            if names is not None and now != names:
                raise RuntimeError(f"backward changed its native calls: {names} then {now}")
            names = now
            if i >= a.warmup:
                for n, (e0, e1) in zip(names, zip(rec.pool, rec.pool[1:])):
                    samples.setdefault(n, []).append(e0.elapsed_time(e1))
            del o
            for t in leaves:
                t.grad = None
        return samples

    def scope_rows(scope, graph, leaves):
        def fb():
            graph().backward(dout)
            for t in leaves:
                t.grad = None
        rows = [row(f"{scope}.total", total(graph, leaves), a.iters)]
        for n, ms in steps(graph, leaves).items():
            rows.append(row(f"{scope}.{n}", ms, a.iters, in_sum=True, note=ZEROING_GAPS.get(n) or OTHER_GAPS.get(n)))
        rows.append(row(f"{scope}.total.repeat", total(graph, leaves), a.iters, note="second measurement, for the floor"))
        rows.append(row(f"context.{scope}.f", loop(graph)))
        rows.append(row(f"context.{scope}.f_plus_b", loop(fb), note="measured F+B with no synchronize inside the loop"))
        return rows

    def event_pair():
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        samples = []
        for _ in range(a.iters * a.batches):
            torch.cuda.synchronize()
            begin.record()
            end.record()
            torch.cuda.synchronize()
            samples.append(begin.elapsed_time(end))
        return samples

    fine = lambda: rep_fine_attention(qq, kk, vv, cnt, sel, mid_reps=rm, backend="tk")[0]
    full = lambda: rep_attention(qq, kk, vv, gg, cnt, R=R, tiers=tiers, backend="tk")
    rows = scope_rows("fine", fine, (qq, kk, vv)) + scope_rows("full", full, (qq, kk, vv, gg))

    f32 = dict(device=dev, dtype=torch.float32)
    o_t, lse_t = C.rep_tk_fwd(q, k, v, tab, sel.q2k_ptr, sel.q2k_idx, sel.b_ptr, sel.b_idx, sel.bits_ab, cnt, rm, PHASE_ALL, 0)
    d_t = C.row_dot(dout, o_t, cnt)
    o_n, lse_n = C.tk_native_fwd(q, k, v, sel.q2k_ptr, sel.q2k_idx, cnt)
    d_n = C.row_dot(dout, o_n, cnt)
    qg, kg, vg = (torch.zeros(q.shape, **f32) for _ in range(3))
    dkm, dvm = (torch.zeros(B, H, nb * rm, D, **f32) for _ in range(2))
    kgt, vgt = (torch.zeros(tab[3].shape, **f32) for _ in range(2))
    rows += [
        row("isolated.tk_native_bwd", loop(lambda: C.tk_native_bwd(q, k, v, dout, lse_n, d_n, sel.k2q_ptr, sel.k2q_idx, cnt,
                                                                   qg, kg, vg))),
        row("isolated.rep_tk_bwd_b", loop(lambda: C.rep_tk_bwd_b(q, tab[0], tab[1], tab[6], dout, lse_t, d_t, sel.b_ptr, sel.b_idx,
                                                                 qg, dkm, dvm, rm))),
        row("isolated.rep_tk_bwd_c", loop(lambda: C.rep_tk_bwd_c(q, tab[3], tab[4], tab[5], dout, lse_t, d_t, sel.bits_ab, qg,
                                                                 kgt, vgt))),
        row("event_pair", event_pair(), a.iters),
    ]
    med = {r["id"]: statistics.median(r["ms"]) for r in rows}
    attribution = {s: attribute(rows, s) for s in ("fine", "full")}
    for s in attribution:
        attribution[s]["b_estimate_ms"] = med[f"context.{s}.f_plus_b"] - med[f"context.{s}.f"]
    res = dict(schema="vsa2-rep-bwd-stages/1", measurement_only=True, note=NOTE, fwd_cfg=cfg, fwd_cfg_meaning=meaning,
               shape=dict(B=B, H=H, L=L, D=D, NB=nb, RM=rm, R=R, dtype="bf16", blk_cnt="full"),
               timing=dict(warmup=a.warmup, iters=a.iters, batches=a.batches, clock="torch.cuda.Event"),
               hardware=dict(name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0))),
               build=dict(extension_sha256=sha256_file(C.__file__), rep_tk_sha256=sha256_file(HERE / "csrc/rep_tk/rep_tk.cu")),
               inputs=dict(seed=seed, q=digest(q), k=digest(k), v=digest(v), dout=digest(dout), gate=digest(gate)),
               rows=rows, attribution=attribution)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"{NOTE}\nREP_TK_FWD_CFG={cfg}: {meaning}\n{'row':46s} {'median ms':>10s} {'spread':>8s}")
    for r in rows:
        print(f"{r['id']:46s} {med[r['id']]:10.3f} {r['spread_ms']:8.3f}")
    for s, t in attribution.items():
        print(f"{s}, derived, not timed: total {t['total_ms']:.3f}  steps {t['sum_of_steps_ms']:.3f}  residual "
              f"{t['residual_ms']:+.3f}  floor {t['floor_ms']:.3f}  within floor: {t['sum_within_floor']}  outside tier kernels "
              f"{t['outside_tier_kernels_ms']:.3f}  F+B-F estimate {t['b_estimate_ms']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
