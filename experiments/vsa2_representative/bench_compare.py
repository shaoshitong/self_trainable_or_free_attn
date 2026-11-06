"""Same-build baseline vs candidate timing, with one machine-readable result per comparison.

REP_TK_FWD_CFG is cached in a function-local static on first use, so one process can only ever measure
one configuration. The driver therefore starts one child per arm in ABBA order, with those variables
removed from the inherited environment and only the requested ones set, and refuses to compare children
whose extension, sources, inputs, selection, shape or timing settings differ.

Profiles (see docs/vsa2_representative_handoff.md, section 7):

  representative      production semantics: REP_TK_FWD_CFG 0/1, diag=0, warmup 10, iters 30. Kernel,
                      fine-op and full-operator scope. Output cross-checked against the generic backend.
  controlled-softmax  REP_TK_FWD_CFG 4/5, warmup 10, iters 50, kernel forward only. TIMING ONLY: these
                      kernels run tier A whatever `phases` says and their output is not valid.

What each number is:

  kernel            one kernel with its tables prebuilt; never a complete call, and stages are never summed
  fine_op, full_op  a complete autograd call. full_op adds router, coarse branch and gate.
  f                 forward; the graph is built on the same leaves as f_plus_b and dropped
  f_plus_b          MEASURED forward + backward + grad reset, no synchronize inside the loop
  b_direct          MEASURED backward alone: fresh graph per sample (retain_graph=False, because backward
                    releases its tables and a graph cannot run twice), forward finished by a synchronize
                    before the interval, grad reset after it. The synchronizes add idle gaps the f_plus_b
                    loop does not have, so b_direct need not equal f_plus_b - f.
  b_estimate_ms     median(f_plus_b) - median(f): derived, an ESTIMATE, not a timed interval

Rows with work="tier_a_only" compute 10% of the pairs and are not equivalent to the 64/4/1 operator.
Control rows do not depend on the configuration; a control ratio away from 1 is run drift, not a result.
No hardware counters are collected (ERR_NVGPUCTRPERM): this file reports no MFU or stall figures.

  CUDA_VISIBLE_DEVICES=0 /opt/venv/bin/python bench_compare.py --profile controlled-softmax --out a.json
  CUDA_VISIBLE_DEVICES=0 /opt/venv/bin/python bench_compare.py --profile representative \
      --baseline REP_TK_FWD_CFG=1 --candidate REP_TK_FWD_CFG=1 --out aa.json
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCHEMA = "vsa2-rep-compare/1"
HERE = Path(__file__).resolve().parent
TUNABLES = ("REP_TK_FWD_CFG", "REP_TK_BWDC_SPLITS")
HANDOFF_BASELINE_SHA256 = "6224637ff9a96404e170aa7de568a070c449ae83707564c37cbfae3cffdc784e"
PROFILES = {
    "representative": dict(warmup=10, iters=30, fwd_cfgs=("0", "1"), baseline="REP_TK_FWD_CFG=1", candidate=None),
    "controlled-softmax": dict(warmup=10, iters=50, fwd_cfgs=("4", "5"), baseline="REP_TK_FWD_CFG=4",
                               candidate="REP_TK_FWD_CFG=5"),
}
CONTROLLED_LIMITS = [
    "timing only: CFG4/5 output has not been validated and the output check is not run",
    "CFG4/5 ignore `phases` and run tier A only; this is not the 64/4/1 operator",
    "CFG4/5 do not write the returned lse; query blocks with an empty tier-A list leave o and lse unwritten",
    "CFG5 normalizes only part of o (the .x lanes, by the l_r[0] row sum), so its epilogue does less work "
    "than CFG4's; the pair isolates the softmax only up to that difference",
    "cannot establish numerical or end-to-end acceleration",
]
# Checked at the run's shape, with the tolerances of tests/test_gpu_tk.py (test_native_and_generic,
# test_backward). A cross-check of this configuration, not a substitute for the three test suites.
TOL_O, TOL_LSE, TOL_GRAD = 1e-2, 1e-4, 2e-2


# ------------------------------------------------------------ driver: no torch, never touches the GPU
def parse_config(text, profile, new_cfgs=()):
    """'K=V,K=V' -> dict, restricted to the tunables and to values the profile's semantics allow.

    The kernel silently runs its default path for an unknown REP_TK_FWD_CFG, so a candidate value has to
    be declared with --new-cfg; otherwise a typo would be reported as a comparison of two equal kernels.
    """
    cfg = {}
    for item in filter(None, (text or "").split(",")):
        key, sep, val = item.partition("=")
        if not sep or key not in TUNABLES:
            raise ValueError(f"bad setting {item!r}: expected KEY=VALUE with KEY in {TUNABLES}")
        cfg[key] = val
    allowed = PROFILES[profile]["fwd_cfgs"] + (tuple(new_cfgs) if profile == "representative" else ())
    if cfg.get("REP_TK_FWD_CFG") not in allowed:
        raise ValueError(f"profile {profile} needs REP_TK_FWD_CFG in {allowed}, got {cfg.get('REP_TK_FWD_CFG')!r}"
                         " (CFG 2-5 do not run the representative operator; a new value needs --new-cfg)")
    splits = cfg.get("REP_TK_BWDC_SPLITS")
    if splits is not None and not (splits.isdigit() and int(splits) > 0):
        raise ValueError(f"REP_TK_BWDC_SPLITS must be positive, got {splits!r}")
    return cfg


def abba(rounds):
    """Run order; drift over the run cancels to first order instead of favouring one arm."""
    return [arm for i in range(rounds) for arm in (("baseline", "candidate") if i % 2 == 0 else
                                                   ("candidate", "baseline"))]


def identity_differences(children):
    """Paths in `identity` where the children disagree; any entry makes the comparison invalid."""
    def flat(x, pre=""):
        if isinstance(x, dict):
            return {p: w for k, y in x.items() for p, w in flat(y, f"{pre}{k}.").items()}
        return {pre[:-1]: x}
    tables = [flat(c["identity"]) for c in children]
    return sorted(k for k in set().union(*tables) if len({json.dumps(t.get(k)) for t in tables}) > 1)


def summarize(children):
    """Per-row medians, ratios and the labelled F+B - F estimate. Stages are never added together."""
    rows = {}
    for child in children:
        for row in child["rows"]:
            arms = rows.setdefault(row["id"], {"meta": row["meta"], "baseline": [], "candidate": []})
            arms[child["arm"]].extend(row["ms"])
    out = []
    median = {}
    for rid, r in rows.items():
        s = dict(id=rid, **r["meta"])
        for arm in ("baseline", "candidate"):
            median[rid, arm] = statistics.median(r[arm])
            s[arm] = dict(median_ms=median[rid, arm], min_ms=min(r[arm]), max_ms=max(r[arm]), n=len(r[arm]))
        s["candidate_over_baseline"] = median[rid, "candidate"] / median[rid, "baseline"]
        out.append(s)
    for s in list(out):
        if s["stage"] != "f_plus_b":
            continue
        base = s["id"].removesuffix(".f_plus_b")
        if (base + ".f", "baseline") not in median:
            continue
        out.append(dict(id=base + ".b_estimate", scope=s["scope"], stage="b_estimate", implementation=s["implementation"],
                        work=s["work"], control=s["control"], measured=False, complete_call=False,
                        method="median(f_plus_b) - median(f); an estimate, not a timed interval",
                        b_estimate_ms={arm: median[base + ".f_plus_b", arm] - median[base + ".f", arm]
                                       for arm in ("baseline", "candidate")}))
    return out


def claims(profile, children):
    """What this result may support; printed and stored so that a reader of the JSON cannot miss it."""
    if profile == "controlled-softmax":
        return dict(output_validated=False, may_claim_op_level_change=False, limits=CONTROLLED_LIMITS)
    ok = all(c.get("output_check") and all(x["ok"] for x in c["output_check"]) for c in children)
    return dict(output_validated=ok, may_claim_op_level_change=ok,
                limits=["needs the CPU, generic and TK test suites on this build as well",
                        "tier_a_only rows are not the 64/4/1 operator; their ratios are not efficiency claims"])


def assemble(args_dict, profile, configs, order, children):
    bad = identity_differences(children)
    claim = claims(profile, children)
    if bad:
        claim["may_claim_op_level_change"] = False
    return dict(schema=SCHEMA, profile=profile, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                settings=args_dict, configs=configs, run_order=order,
                status="invalid: arms differ in " + ", ".join(bad) if bad else "ok",
                identity_differences=bad, claims=claim,
                counters=dict(collected=False, reason="ERR_NVGPUCTRPERM; no permission workaround is authorized"),
                summary=summarize(children) if not bad else [], children=children)


def run_child(a, arm, config, round_idx):
    env = {k: v for k, v in os.environ.items() if k not in TUNABLES}
    env.update(config)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "child.json"
        spec = dict(profile=a.profile, len=a.len, heads=a.heads, dim=a.dim, mid_reps=a.mid_reps, seed=a.seed,
                    warmup=a.warmup, iters=a.iters, batches=a.batches, out=str(path))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(spec)], env=env, check=True)
        child = json.loads(path.read_text())
    child.update(arm=arm, config=config, round=round_idx)
    return child


def report(res):
    print(f"\n{res['profile']}: {res['status']}  run order {' '.join(res['run_order'])}")
    print(f"{'row':44s} {'work':12s} {'baseline':>9s} {'candidate':>9s}  ratio")
    for s in res["summary"]:
        if s["stage"] == "b_estimate":
            e = s["b_estimate_ms"]
            print(f"{s['id']:44s} {s['work']:12s} {e['baseline']:9.3f} {e['candidate']:9.3f}  ESTIMATE F+B-F")
        else:
            print(f"{s['id']:44s} {s['work']:12s} {s['baseline']['median_ms']:9.3f} {s['candidate']['median_ms']:9.3f}"
                  f"  {s['candidate_over_baseline']:.3f}{'  control' if s['control'] else ''}")
    print(f"output validated: {res['claims']['output_validated']}; "
          f"may claim op-level change: {res['claims']['may_claim_op_level_change']}")
    for lim in res["claims"]["limits"]:
        print(f"  limit: {lim}")


def main():
    ap = argparse.ArgumentParser(description="same-build baseline vs candidate timing")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--profile", choices=PROFILES, default="representative")
    ap.add_argument("--baseline")
    ap.add_argument("--candidate")
    ap.add_argument("--new-cfg", action="append", default=[], help="declare a REP_TK_FWD_CFG value the kernel implements")
    ap.add_argument("--len", type=int, default=16384)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--dim", type=int, default=128, choices=(64, 128))
    ap.add_argument("--mid-reps", type=int, default=4, choices=(1, 2, 4, 8, 16))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, help="default: the profile's")
    ap.add_argument("--iters", type=int, help="default: the profile's")
    ap.add_argument("--batches", type=int, default=3, help="timed loops per row per child; one synchronize after each")
    ap.add_argument("--rounds", type=int, default=2, help="ABBA rounds")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.child:
        return child_main(json.loads(a.child))
    p = PROFILES[a.profile]
    a.warmup = p["warmup"] if a.warmup is None else a.warmup
    a.iters = p["iters"] if a.iters is None else a.iters
    if a.len % 64 or a.len <= 0 or a.batches < 1 or a.rounds < 1 or a.iters < 1 or a.warmup < 0:
        ap.error("--len must be a positive multiple of 64, and the counts positive")
    if not a.out:
        ap.error("--out is required")
    out = Path(a.out)
    if out.exists():
        ap.error(f"{out} exists; results are not overwritten")
    candidate = a.candidate or p["candidate"]
    if candidate is None:
        ap.error("--candidate is required for this profile (equal to --baseline for an A/A noise run)")
    try:
        configs = dict(baseline=parse_config(a.baseline or p["baseline"], a.profile, a.new_cfg),
                       candidate=parse_config(candidate, a.profile, a.new_cfg))
    except ValueError as e:
        ap.error(str(e))
    order = abba(a.rounds)
    children = [run_child(a, arm, configs[arm], i) for i, arm in enumerate(order)]
    settings = {k: getattr(a, k) for k in ("len", "heads", "dim", "mid_reps", "seed", "warmup", "iters", "batches", "rounds")}
    settings["declared_new_cfgs"] = a.new_cfg
    res = assemble(settings, a.profile, configs, order, children)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(res, indent=1) + "\n")
    tmp.replace(out)
    report(res)
    print(f"wrote {out}")
    return 0 if res["status"] == "ok" else 1


# ------------------------------------------------------------ child: one configuration, one process
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(root, patterns):
    """{relative path: sha256}, plus one digest over them; None when the root is not available here."""
    root = Path(root)
    if not root.is_dir():
        return None
    files = sorted(p for pat in patterns for p in root.glob(pat) if p.is_file())
    table = {str(p.relative_to(root)): sha256_file(p) for p in files}
    return dict(digest=hashlib.sha256(json.dumps(table, sort_keys=True).encode()).hexdigest(), files=table)


def capture(*cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def telemetry():
    """Plain nvidia-smi state of the visible GPU (not performance counters), around each child."""
    fields = "pstate,clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw,utilization.gpu,memory.used"
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    target = ["-i", cvd] if cvd.isdigit() else []
    line = capture("nvidia-smi", *target, f"--query-gpu={fields}", "--format=csv,noheader")
    return dict(fields=fields, values=line)


def child_main(spec):
    telemetry_before = telemetry()
    import torch

    from vsa2_rep.ops import (PHASE_A, PHASE_ALL, PHASE_B, PHASE_C, RepTiers, ext, rep_attention, rep_fine_attention,
                              route_tiers, tk_tables)

    def digest(t):
        data = t.float() if t.dtype == torch.bfloat16 else t
        return hashlib.sha256(str(t.dtype).encode() + str(tuple(t.shape)).encode()
                              + data.detach().cpu().numpy().tobytes()).hexdigest()

    def loop(fn, warmup, iters, batches):
        """bench_tk.timeit repeated `batches` times: events around a loop, one synchronize at the end."""
        for _ in range(warmup):
            fn()
        ms = []
        for _ in range(batches):
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            for _ in range(iters):
                fn()
            b.record()
            torch.cuda.synchronize()
            ms.append(a.elapsed_time(b) / iters)
        return ms

    if not torch.cuda.is_available() or not ext().has_tk():
        sys.exit("needs a GPU and a VSA2_REP_TK=1 build")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd.isdigit() or torch.cuda.device_count() != 1:
        sys.exit(f"needs exactly one GPU selected by CUDA_VISIBLE_DEVICES, got {cvd!r}")

    dev, dt = "cuda", torch.bfloat16
    B, H, L, D, rm, nb = 1, spec["heads"], spec["len"], spec["dim"], spec["mid_reps"], spec["len"] // 64
    W, I, N = spec["warmup"], spec["iters"], spec["batches"]
    torch.manual_seed(spec["seed"])
    q, k, v = (torch.randn(B, H, L, D, device=dev, dtype=dt).contiguous() for _ in range(3))
    dout = torch.randn_like(q)
    gate = (0.1 * torch.randn(B, H, L, device=dev)).to(dt)
    cnt = torch.full((nb,), 64, dtype=torch.int32, device=dev)
    tiers = RepTiers(0.10, 0.20, rm)
    sel = route_tiers(q, k, cnt, tiers, R=16)
    qp, qi, kp, ki = sel.q2k_ptr, sel.q2k_idx, sel.k2q_ptr, sel.k2q_idx
    sel_fields = ("q2k_ptr", "q2k_idx", "k2q_ptr", "k2q_idx", "bits_a", "bits_ab", "b_ptr", "b_idx")
    f32 = dict(device=dev, dtype=torch.float32)
    C = ext()
    tab = tk_tables(C, k, v, cnt, sel, rm)
    rows = []

    def add(rid, scope, stage, impl, work, control, ms, **extra):
        rows.append(dict(id=rid, ms=ms, meta=dict(scope=scope, stage=stage, implementation=impl, work=work,
                                                  control=control, measured=True,
                                                  complete_call=stage in ("f", "f_plus_b"), **extra)))

    def kernel_fwd(ph):
        return lambda: C.rep_tk_fwd(q, k, v, tab, qp, qi, sel.b_ptr, sel.b_idx, sel.bits_ab, cnt, rm, ph, 0)

    add("kernel.native_tk.fwd.A", "kernel", "fwd", "native_tk", "tier_a_only", True,
        loop(lambda: C.tk_native_fwd(q, k, v, qp, qi, cnt), W, I, N))
    add("kernel.tk.fwd.A", "kernel", "fwd", "tk", "tier_a_only", False, loop(kernel_fwd(PHASE_A), W, I, N))
    output_check = None
    if spec["profile"] == "representative":
        for name, ph in (("A|B", PHASE_A | PHASE_B), ("A|B|C", PHASE_ALL), ("B|C", PHASE_B | PHASE_C)):
            add(f"kernel.tk.fwd.{name}", "kernel", "fwd", "tk", "64/4/1" if ph == PHASE_ALL else "tier_subset", False,
                loop(kernel_fwd(ph), W, I, N))
        gtab = C.rep_build(k, v, cnt, rm)
        add("kernel.generic.fwd.A|B|C", "kernel", "fwd", "generic", "64/4/1", True,
            loop(lambda: C.rep_attn_fwd(q, k, v, *gtab, qp, qi, sel.bits_a, sel.bits_ab, cnt, rm, PHASE_ALL, 0), W, I, N))
        add("kernel.tk.tables", "kernel", "tables", "tk", "64/4/1", False,
            loop(lambda: tk_tables(C, k, v, cnt, sel, rm), W, I, N))

        # Backward kernels, one stage each. Results accumulate into buffers that are never read and not
        # reset, as in bench_tk.py: zeroing them per call would be timed work no kernel row should carry.
        o_t, lse_t = kernel_fwd(PHASE_ALL)()
        d_t = C.row_dot(dout, o_t, cnt)
        o_n, lse_n = C.tk_native_fwd(q, k, v, qp, qi, cnt)
        d_n = C.row_dot(dout, o_n, cnt)
        qg, kg, vg = (torch.zeros(q.shape, **f32) for _ in range(3))
        dkm, dvm = torch.zeros(B, H, nb * rm, D, **f32), torch.zeros(B, H, nb * rm, D, **f32)
        kgt, vgt = torch.zeros(tab[3].shape, **f32), torch.zeros(tab[3].shape, **f32)
        for name, work, fn in (
            ("A", "tier_a_only", lambda: C.tk_native_bwd(q, k, v, dout, lse_n, d_n, kp, ki, cnt, qg, kg, vg)),
            ("B", "tier_subset", lambda: C.rep_tk_bwd_b(q, tab[0], tab[1], tab[6], dout, lse_t, d_t, sel.b_ptr, sel.b_idx,
                                                        qg, dkm, dvm, rm)),
            ("C", "tier_subset", lambda: C.rep_tk_bwd_c(q, tab[3], tab[4], tab[5], dout, lse_t, d_t, sel.bits_ab, qg, kgt,
                                                        vgt)),
        ):
            add(f"kernel.tk.bwd.{name}", "kernel", "bwd_stage", "tk", work, False, loop(fn, W, I, N),
                note="one stage; do not add stages and call the sum a complete call")
        del o_t, lse_t, d_t, o_n, lse_n, d_n, qg, kg, vg, dkm, dvm, kgt, vgt

        qq, kk, vv, gg = (t.detach().clone().requires_grad_(True) for t in (q, k, v, gate))

        def timed_op(prefix, scope, impl, work, control, graph, leaves):
            def fb():
                graph().backward(dout)
                for t in leaves:
                    t.grad = None

            def direct():
                samples = []
                for i in range(W + I * N):
                    o = graph()
                    torch.cuda.synchronize()  # the forward is done before the interval opens
                    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                    a.record()
                    o.backward(dout)
                    b.record()
                    torch.cuda.synchronize()
                    if i >= W:
                        samples.append(a.elapsed_time(b))
                    del o
                    for t in leaves:  # after the interval closes
                        t.grad = None
                return samples

            add(f"{prefix}.f", scope, "f", impl, work, control, loop(graph, W, I, N))
            add(f"{prefix}.f_plus_b", scope, "f_plus_b", impl, work, control, loop(fb, W, I, N),
                note="grad reset inside the timed loop; no synchronize inside it")
            add(f"{prefix}.b_direct", scope, "b_direct", impl, work, control, direct(),
                note="retain_graph=False, fresh graph per sample; forward completed and grads reset outside the interval")

        def fine(ph, be):
            return lambda: rep_fine_attention(qq, kk, vv, cnt, sel, mid_reps=rm, phases=ph, backend=be)[0]

        timed_op("fine_op.tk.A|B|C", "fine_op", "tk", "64/4/1", False, fine(PHASE_ALL, "tk"), (qq, kk, vv))
        timed_op("fine_op.tk.A", "fine_op", "tk", "tier_a_only", False, fine(PHASE_A, "tk"), (qq, kk, vv))
        timed_op("fine_op.generic.A|B|C", "fine_op", "generic", "64/4/1", True, fine(PHASE_ALL, "generic"), (qq, kk, vv))
        timed_op("full_op.tk.routed", "full_op", "tk", "64/4/1", False,
                 lambda: rep_attention(qq, kk, vv, gg, cnt, R=16, tiers=tiers, backend="tk"), (qq, kk, vv, gg))

        # Not timed. The routed rows only count as the same selection if the router reproduces it.
        _, routed = rep_attention(q, k, v, gate, cnt, R=16, tiers=tiers, backend="tk", return_selection=True)
        routed_same = all(digest(getattr(routed, n)) == digest(getattr(sel, n)) for n in sel_fields[:6])

        def compare(name, got, want, tol):
            got, want = got.float(), want.float()
            err, scale = (got - want).abs().max().item(), want.abs().max().item() or 1.0
            return dict(name=name, max_err=err, scale=scale, tol_rel=tol, ok=err <= tol * max(scale, 1.0))

        o_t, lse_t = kernel_fwd(PHASE_ALL)()
        o_g, lse_g = C.rep_attn_fwd(q, k, v, *gtab, qp, qi, sel.bits_a, sel.bits_ab, cnt, rm, PHASE_ALL, 0)
        output_check = [compare("o vs generic", o_t, o_g, TOL_O), compare("lse vs generic", lse_t, lse_g, TOL_LSE)]
        grads = {}
        for be in ("tk", "generic"):
            fine(PHASE_ALL, be)().backward(dout)
            grads[be] = [t.grad.clone() for t in (qq, kk, vv)]
            qq.grad = kk.grad = vv.grad = None
        output_check += [compare(f"{n} vs generic", grads["tk"][i], grads["generic"][i], TOL_GRAD)
                         for i, n in enumerate(("dQ", "dK", "dV"))]
        output_check.append(dict(name="router reproduces the frozen selection", ok=routed_same))

    props = torch.cuda.get_device_properties(0)
    source = os.environ.get("VSA2_REP_SOURCE")
    tk_root = os.environ.get("VSA2_REP_TK_ROOT")
    identity = dict(
        host=dict(hostname=platform.node(), cuda_visible_devices=cvd),
        hardware=dict(name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                      uuid=str(getattr(props, "uuid", "")), total_memory=props.total_memory,
                      driver=capture("nvidia-smi", "-i", cvd, "--query-gpu=driver_version", "--format=csv,noheader")),
        software=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                      nvcc=(capture("nvcc", "--version") or "").splitlines()[1:2]),
        # The extension's hash is the build identity. The source hashes say what was on disk at run time:
        # nothing here can prove that those bytes are what the extension was compiled from.
        build=dict(extension_sha256=sha256_file(C.__file__), extension=os.path.basename(C.__file__),
                   build_log_sha256=sha256_file(HERE / "build.log") if (HERE / "build.log").exists() else None),
        source=dict(package=tree_sha256(HERE, ("csrc/**/*", "vsa2_rep/*.py", "setup.py", "bench_compare.py")),
                    rep_tk_is_handoff_baseline=sha256_file(HERE / "csrc/rep_tk/rep_tk.cu") == HANDOFF_BASELINE_SHA256,
                    vsa2=tree_sha256(source, ("csrc/vsa/**/*",)) if source else None,
                    git_head=capture("git", "-C", str(HERE), "rev-parse", "HEAD"),
                    git_status=capture("git", "-C", str(HERE), "status", "--porcelain", "."),  # None: not a checkout
                    thunderkittens_head=capture("git", "-C", tk_root, "rev-parse", "HEAD") if tk_root else None),
        shape=dict(B=B, H=H, L=L, D=D, NB=nb, RM=rm, dtype="bf16", blk_cnt="full"),
        configuration=dict(frac_full=tiers.frac_full, frac_mid=tiers.frac_mid, mid_reps=rm, R=16, phases_all=PHASE_ALL,
                           diag=0, profile=spec["profile"]),
        timing=dict(warmup=W, iters=I, batches=N, clock="torch.cuda.Event"),
        inputs=dict(seed=spec["seed"], **{n: digest(t) for n, t in (("q", q), ("k", k), ("v", v), ("dout", dout),
                                                                     ("gate", gate), ("blk_cnt", cnt))}),
        selection={n: digest(getattr(sel, n)) for n in sel_fields},
    )
    Path(spec["out"]).write_text(json.dumps(dict(identity=identity, telemetry_before=telemetry_before, rows=rows,
                                                 output_check=output_check, telemetry_after=telemetry())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
