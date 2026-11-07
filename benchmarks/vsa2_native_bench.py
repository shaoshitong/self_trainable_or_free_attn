"""Native VSA2 operator (echosparse.vsa.ops): timing and equivalence of one env-toggle arm against the default.

Measurement only. This script reports medians, spreads, ratios and errors, and makes no performance claim.

One arm per run. For each shape the driver starts four processes in ABBA order: default, arm, arm, default. Every
ECHOSPARSE_VSA_* variable is removed from the inherited environment and only the arm's is set. `--arm baseline` makes
all four the default, which is the A/A run. Ratios only ever come from inside one run, because separate runs drift.

Rows, all through vsa2_attention with the router, a gate and backend="tk" (CUDA events, warmup 10, 30 iterations, 3 batches):
  native.fwd       the forward, in a loop, on the same grad-tracking leaves as f_plus_b
  native.bwd       one backward: fresh graph per sample, forward finished by a synchronize before the interval, gradients
                   reset after it. The synchronizes add idle gaps that f_plus_b does not have
  native.f_plus_b  a complete autograd call: forward, backward and gradient reset, no synchronize inside the loop
  b_estimate_ms    median(f_plus_b) - median(fwd): derived, not a timed interval

Equivalence, never folded into the timings, compares the four processes' tensors on the CPU:
  forward   o and the selection must be bit-identical: arm against default, and default against default
  backward  dq, dk, dv and dgate are cross-CTA fp32 reduce-adds, not run-to-run deterministic, so byte equality is the wrong
            bar. Each is reported as an error against the default, beside the same error between two identical runs
  padding   slots past blk_cnt must be exactly zero in o and all gradients

Verdict for one shape, worst first:
  FAIL            NaN, a nonzero padded slot, or o or the selection differing from the default
  NONDETERMINISTIC_FORWARD  the default's o differs from itself, so the forward check cannot separate arms
  OVER_TOL        a gradient's rel_l2 against the default exceeds --grad-tol
  ABOVE_AA        within tolerance, but max_abs or n_diff exceed AA_MARGIN times the larger A/A sample: noise does not explain it
  OK              the outputs match, and gradient differences are within what two identical runs show

The default is the reference. This says an arm matches the default, not that the default is right; the test suites do that.
An arm that silently does nothing also reads OK and neutral: the guards read source, not the compiled extension.

  <gpu-runner> python benchmarks/vsa2_native_bench.py --arm baseline --out aa.json
  <gpu-runner> python benchmarks/vsa2_native_bench.py --arm fused_cols --out fused.json
  python benchmarks/vsa2_native_bench.py --compare aa.json fused.json
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCHEMA = "vsa2-native-bench/1"
NOTE = "measurement_only: medians, spreads, ratios and errors; not a performance claim"
ROOT = Path(__file__).resolve().parents[1]
ENV_PREFIX = "ECHOSPARSE_VSA_"
ARMS = {"baseline": {}, "coarse_meta_cache": {"ECHOSPARSE_VSA_COARSE_META_CACHE": "1"},
        "fused_cols": {"ECHOSPARSE_VSA_FUSED_COLS": "1"}, "tk_bwd_store": {"ECHOSPARSE_VSA_TK_BWD_STORE": "1"}}
ROWS = ("native.fwd", "native.bwd", "native.f_plus_b")
GRADIENTS = ("dq", "dk", "dv", "dgate")
ORDER = ("default", "arm", "arm", "default")
LENS, HEADS, DIMS = (2048, 4096, 8192, 16384, 32768), (8, 16), (64, 128)
PARTIAL_SHAPE = dict(H=16, L=16384, D=128)
AA_MARGIN = 4.0  # identical runs were measured 300-1200 differing elements apart, so one A/A pair is only a rough floor


def shape_matrix(lens=LENS, heads=HEADS, dims=DIMS, partial=True):
    shapes = [dict(B=1, H=h, L=l, D=d, partial=False) for l in lens for h in heads for d in dims]
    return shapes + ([dict(B=1, partial=True, **PARTIAL_SHAPE)] if partial else [])


def shape_key(s):
    return f"L{s['L']}/H{s['H']}/D{s['D']}" + ("/partial" if s["partial"] else "")


def shape_error(s):
    """Why echosparse.vsa.ops cannot run this shape, by its layout contract, or None."""
    if s["L"] <= 0 or s["L"] % 64:
        return f"L must be a positive multiple of 64, got {s['L']}"
    if s["D"] not in (64, 128):
        return f"D must be 64 or 128, got {s['D']}"
    return None


def partial_blk_cnt(nb):
    """The native contract is 1..64 valid tokens per block, so no block is empty here."""
    def count(i):
        if i == nb - 1:
            return 9
        return 1 if i % 7 == 3 else 33 if i % 4 == 1 else 17 if i % 16 == 5 else 64
    return [count(i) for i in range(nb)]


def child_env(arm, environ=os.environ):
    env = {k: v for k, v in environ.items() if not k.startswith(ENV_PREFIX) or k == "ECHOSPARSE_VSA_BUILD"}
    env.update(ARMS[arm])
    return env


def source_files():
    return sorted(p for pat in ("echosparse/vsa/*.py", "csrc/vsa/**/*") for p in ROOT.glob(pat) if p.is_file())


def arm_error(arm, sources=None):
    """A toggle nothing reads would silently measure the default."""
    sources = [p.read_text(errors="replace") for p in source_files()] if sources is None else sources
    missing = [name for name in ARMS[arm] if not any(name in text for text in sources)]
    return f"--arm {arm}: nothing under echosparse/vsa or csrc/vsa reads {missing}" if missing else None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_digest():
    table = {str(p.relative_to(ROOT)): sha256_file(p) for p in source_files()}
    return hashlib.sha256(json.dumps(table, sort_keys=True).encode()).hexdigest()


def spread(values):
    return max(values) - min(values)


def time_rows(children):
    """Per-row timing for one shape; children are in ABBA order. Ratios come from inside this run only."""
    default = [c for c, who in zip(children, ORDER) if who == "default"]
    arm = [c for c, who in zip(children, ORDER) if who == "arm"]
    rows = {}
    for name in ROWS:
        base = [x for c in default for x in c["rows"][name]]
        cand = [x for c in arm for x in c["rows"][name]]
        first, last = (statistics.median(c["rows"][name]) for c in default)
        rows[name] = dict(default_median_ms=statistics.median(base), default_spread_ms=spread(base),
                          arm_median_ms=statistics.median(cand), arm_spread_ms=spread(cand),
                          ratio=statistics.median(cand) / statistics.median(base), aa_displacement_same_run=last / first,
                          samples=dict(default=base, arm=cand), measured=True)
    fb, f = rows["native.f_plus_b"], rows["native.fwd"]
    rows["b_estimate_ms"] = dict(measured=False, method="median(f_plus_b) - median(fwd); not a timed interval",
                                 default=fb["default_median_ms"] - f["default_median_ms"],
                                 arm=fb["arm_median_ms"] - f["arm_median_ms"])
    return rows


def worst(samples, key):
    """The larger of a metric over error samples; NaN wins, so it can never read as small."""
    values = [s[key] for s in samples]
    return next((v for v in values if v != v), max(values))


def gradient_verdict(vs_default, aa, grad_tol, margin=AA_MARGIN):
    """OK, ABOVE_AA, OVER_TOL or FAIL for one gradient, from error samples against the default and from A/A pairs."""
    err, n, rel = worst(vs_default, "max_abs"), worst(vs_default, "n_diff"), worst(vs_default, "rel_l2")
    if worst(vs_default, "n_nan") > 0 or err != err or rel != rel:
        return "FAIL"
    if rel > grad_tol:
        return "OVER_TOL"
    if err > margin * worst(aa, "max_abs") or n > margin * worst(aa, "n_diff"):
        return "ABOVE_AA"
    return "OK"


RANK = ("OK", "ABOVE_AA", "OVER_TOL", "NONDETERMINISTIC_FORWARD", "FAIL")


def shape_verdict(forward, backward, padded_nonzero):
    """The worst of the forward, padding and gradient results for one shape; see the module docstring."""
    found = ["OK"] + [g["verdict"] for g in backward.values()]
    if not forward["aa_o_identical"] or not forward["aa_selection_identical"]:
        found.append("NONDETERMINISTIC_FORWARD")
    if not forward["o_identical"] or not forward["selection_identical"] or any(padded_nonzero.values()):
        found.append("FAIL")
    return max(found, key=RANK.index)


def equivalence(children, errors, grad_tol):
    """errors(tensor name, i, j) is the error between children i and j, in ORDER positions."""
    o = [c["digests"]["o"] for c in children]
    sel = [c["digests"]["selection"] for c in children]
    forward = dict(o_sha256=o, o_identical=o[1] == o[0] and o[2] == o[3], aa_o_identical=o[0] == o[3],
                   selection_identical=sel[1] == sel[0] and sel[2] == sel[3], aa_selection_identical=sel[0] == sel[3])
    backward = {}
    for name in GRADIENTS:
        vs_default = [errors(name, 1, 0), errors(name, 2, 3)]
        aa = [errors(name, 3, 0), errors(name, 2, 1)]  # default against default, arm against arm
        backward[name] = dict(vs_default=vs_default, aa=aa, verdict=gradient_verdict(vs_default, aa, grad_tol))
    padded = {k: max(c["padded_nonzero"][k] for c in children) for k in children[0]["padded_nonzero"]}
    return dict(forward=forward, backward=backward, padded_nonzero=padded, grad_tol=grad_tol, aa_margin=AA_MARGIN,
                verdict=shape_verdict(forward, backward, padded))


def tensor_errors(a, b):
    """Error of a against b. n_diff counts numeric differences, n_bit_diff also counts a differing sign of zero."""
    import torch

    a, b = a.float(), b.float()
    diff = a - b
    norm = b.double().norm().item()
    return dict(max_abs=diff.abs().max().item(), scale=b.abs().max().item(), rel_l2=diff.double().norm().item() / (norm or 1.0),
                n_diff=int((a != b).sum().item()), n_bit_diff=int(((a != b) | (torch.signbit(a) != torch.signbit(b))).sum().item()),
                n_nan=int(torch.isnan(a).sum().item()), n_elements=a.numel())


def compare_shape(children, grad_tol):
    """Load the four processes' tensors on the CPU and compare them. The driver never touches the GPU."""
    import torch

    def errors(name, i, j):
        x = torch.load(children[i]["tensors"], map_location="cpu", mmap=True)[name]
        y = torch.load(children[j]["tensors"], map_location="cpu", mmap=True)[name]
        return tensor_errors(x, y)
    return equivalence(children, errors, grad_tol)


def run_shape(a, shape, scratch):
    """Four processes for one shape, then their comparison. A shape that cannot run says why."""
    work = tempfile.mkdtemp(prefix="vsa2_native_", dir=scratch)
    try:
        children = []
        for i, who in enumerate(ORDER):
            spec = dict(shape=shape, seed=a.seed, topk_frac=a.topk_frac, warmup=a.warmup, iters=a.iters, batches=a.batches,
                        out=os.path.join(work, f"{i}.json"), tensors=os.path.join(work, f"{i}.pt"))
            env = child_env(a.arm if who == "arm" else "baseline")
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(spec)], env=env,
                               capture_output=True, text=True)
            if not Path(spec["out"]).exists():
                tail = (r.stderr.strip().splitlines() or ["no output"])[-1]
                return dict(status="failed", reason=f"process {i} ({who}) exited {r.returncode}: {tail}")
            child = json.loads(Path(spec["out"]).read_text())
            if child.get("status") != "ok":
                return dict(status=child.get("status"), reason=f"process {i} ({who}): {child.get('reason')}")
            children.append(child)
        keys = {json.dumps(c["identity"], sort_keys=True) for c in children}
        if len(keys) > 1:
            return dict(status="invalid", reason="processes differ in build, hardware, inputs or shape")
        try:
            eq = compare_shape(children, a.grad_tol)
        except Exception as e:  # timings without their comparison must not read as a clean result
            return dict(status="failed", reason=f"comparing tensors: {type(e).__name__}: {e}")
        return dict(status="ok", topk=children[0]["topk"], blk_cnt=children[0]["blk_cnt"], run_order=list(ORDER),
                    timing=time_rows(children), equivalence=eq,
                    identity=children[0]["identity"],
                    stale_extension_sources=sorted({n for c in children for n in c["stale_extension_sources"]}))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def child_main(spec):
    out = Path(spec["out"])
    shape = spec["shape"]
    try:
        out.write_text(json.dumps(measure(spec, shape)))
    except Exception as e:  # a shape that does not run is a result, not a skipped row
        out.write_text(json.dumps(dict(status="failed", reason=f"{type(e).__name__}: {e}")))
    return 0


def measure(spec, shape):
    sys.path.insert(0, str(ROOT))
    import torch

    from echosparse.vsa import ops

    error = shape_error(shape)
    if error:
        return dict(status="unsupported", reason=error)
    if not torch.cuda.is_available():
        return dict(status="unsupported", reason="no GPU")
    if torch.cuda.device_count() != 1:
        return dict(status="unsupported", reason=f"needs exactly one visible GPU, got {torch.cuda.device_count()}")
    C = ops.ext()
    W, I, N = spec["warmup"], spec["iters"], spec["batches"]
    dev, dt, seed = "cuda", torch.bfloat16, spec["seed"]
    B, H, L, D, nb = shape["B"], shape["H"], shape["L"], shape["D"], shape["L"] // 64
    topk = max(1, round(spec["topk_frac"] * nb))

    def digest(t):
        data = t.float() if t.dtype == torch.bfloat16 else t
        return hashlib.sha256(str(t.dtype).encode() + str(tuple(t.shape)).encode()
                              + data.detach().cpu().numpy().tobytes()).hexdigest()

    def loop(fn):
        for _ in range(W):
            fn()
        ms = []
        for _ in range(N):
            torch.cuda.synchronize()
            t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
            t0.record()
            for _ in range(I):
                fn()
            t1.record()
            torch.cuda.synchronize()
            ms.append(t0.elapsed_time(t1) / I)
        return ms

    torch.manual_seed(seed)
    q, k, v = (torch.randn(B, H, L, D, device=dev, dtype=dt).contiguous() for _ in range(3))
    dout = torch.randn_like(q)
    gate = (0.1 * torch.randn(B, H, L, device=dev)).to(dt)
    counts = partial_blk_cnt(nb) if shape["partial"] else [64] * nb
    cnt = torch.tensor(counts, dtype=torch.int32, device=dev)
    valid = (torch.arange(64, device=dev)[None, :] < cnt.to(torch.int64)[:, None]).reshape(L)
    for t in (q, k, v, gate):  # the layout contract: padded slots hold finite values
        t[:, :, ~valid] = 0
    leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, gate)]
    qq, kk, vv, gg = leaves

    def forward(**kw):
        return ops.vsa2_attention(qq, kk, vv, gg, cnt, R=16, topk=topk, backend="tk", **kw)

    def reset():
        for t in leaves:
            t.grad = None

    def f_plus_b():
        forward().backward(dout)
        reset()

    def direct():
        samples = []
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        for i in range(W + I * N):
            o = forward()
            torch.cuda.synchronize()
            begin.record()
            o.backward(dout)
            end.record()
            torch.cuda.synchronize()
            if i >= W:
                samples.append(begin.elapsed_time(end))
            del o
            reset()
        return [statistics.median(samples[i:i + I]) for i in range(0, len(samples), I)]

    rows = {"native.fwd": loop(forward), "native.bwd": direct(), "native.f_plus_b": loop(f_plus_b)}

    o, sel = forward(return_selection=True)
    o.backward(dout)
    outputs = dict(o=o.detach(), dq=qq.grad, dk=kk.grad, dv=vv.grad, dgate=gg.grad)
    torch.save({n: t.detach().cpu() for n, t in outputs.items()}, spec["tensors"])
    padded = {n: int(((t.float()[:, :, ~valid] != 0)).sum().item()) for n, t in outputs.items()}
    stale = sorted(p.name for p in source_files() if p.suffix != ".py" and p.stat().st_mtime > os.path.getmtime(C.__file__))
    props = torch.cuda.get_device_properties(0)
    identity = dict(
        build=dict(extension_sha256=sha256_file(C.__file__), source_sha256=source_digest(), harness_sha256=sha256_file(__file__)),
        hardware=dict(name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                      multi_processor_count=getattr(props, "multi_processor_count", None),
                      uuid_sha256=hashlib.sha256(str(getattr(props, "uuid", "")).encode()).hexdigest()),
        software=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda),
        shape=dict(shape, NB=nb, R=16, topk=topk, dtype="bf16", backend="tk", gate="per token"),
        timing=dict(warmup=W, iters=I, batches=N, clock="torch.cuda.Event"),
        inputs=dict(seed=seed, **{n: digest(t) for n, t in (("q", q), ("k", k), ("v", v), ("dout", dout), ("gate", gate), ("blk_cnt", cnt))}))
    selection = hashlib.sha256("".join(digest(getattr(sel, n)) for n in ("q2k_ptr", "q2k_idx", "k2q_ptr", "k2q_idx")).encode()).hexdigest()
    return dict(status="ok", topk=topk, blk_cnt=dict(min=min(counts), max=max(counts), partial_blocks=sum(c < 64 for c in counts), blocks=nb),
                rows=rows, digests=dict(o=digest(o), selection=selection), padded_nonzero=padded, tensors=spec["tensors"],
                identity=identity, stale_extension_sources=stale)


# ------------------------------------------------------------ reading results
def compare_error(aa, arm):
    """Why these two results cannot be read together, or None."""
    if aa.get("schema") != SCHEMA or arm.get("schema") != SCHEMA:
        return f"both files must be {SCHEMA} results"
    if aa["arm"] != "baseline":
        return f"the first file must be the A/A run (--arm baseline), got --arm {aa['arm']}"
    if set(aa["shapes"]) != set(arm["shapes"]):
        return f"shapes differ: {sorted(set(aa['shapes']) ^ set(arm['shapes']))}"
    for key in ("warmup", "iters", "batches", "seed", "topk_frac", "grad_tol"):
        if aa["settings"][key] != arm["settings"][key]:
            return f"settings differ in {key}"
    for name in aa["shapes"]:
        x, y = aa["shapes"][name], arm["shapes"][name]
        if x["status"] == "ok" and y["status"] == "ok":
            for part in ("build", "hardware", "software", "shape", "timing", "inputs"):
                if x["identity"][part] != y["identity"][part]:
                    return f"{name}: runs differ in {part}"
    return None


def combined_verdict(shape, aa_shape):
    """The arm run's verdict, re-read against the larger floor of its own A/A pairs and the A/A run's."""
    eq = shape["equivalence"]
    extra = aa_shape["equivalence"]["backward"] if aa_shape and aa_shape["status"] == "ok" else {}
    backward = {n: dict(g, verdict=gradient_verdict(g["vs_default"], g["aa"] + extra.get(n, {}).get("aa", []), eq["grad_tol"]))
                for n, g in eq["backward"].items()}
    return shape_verdict(eq["forward"], backward, eq["padded_nonzero"])


def report(res, aa=None):
    print(NOTE)
    print(f"arm {res['arm']}, env {ARMS[res['arm']]}; ratio = arm / default inside this run, ABBA per shape")
    aa_note = "A/A same run = last default / first default in this run" + ("; A/A run = ratio of the --arm baseline run" if aa else "")
    print(aa_note)
    print(f"\n{'shape':24s} {'row':16s} {'default ms':>10s} {'spread':>7s} {'arm ms':>9s} {'spread':>7s} {'ratio':>7s} {'A/A same':>9s}"
          + (f" {'A/A run':>8s}" if aa else ""))
    for name, s in res["shapes"].items():
        if s["status"] != "ok":
            print(f"{name:24s} {s['status'].upper()}: {s['reason']}")
            continue
        for row in ROWS:
            t = s["timing"][row]
            line = (f"{name:24s} {row:16s} {t['default_median_ms']:10.3f} {t['default_spread_ms']:7.3f} {t['arm_median_ms']:9.3f} "
                    f"{t['arm_spread_ms']:7.3f} {t['ratio']:7.3f} {t['aa_displacement_same_run']:9.3f}")
            if aa:
                other = aa["shapes"][name]
                line += f" {other['timing'][row]['ratio']:8.3f}" if other["status"] == "ok" else f" {'n/a':>8s}"
            print(line)
    print(f"\n{'shape':24s} {'output':6s} {'max_abs':>10s} {'rel_l2':>10s} {'n_diff':>9s} {'of':>10s} {'A/A max_abs':>11s} {'A/A n_diff':>10s}  verdict")
    for name, s in res["shapes"].items():
        if s["status"] != "ok":
            continue
        eq = s["equivalence"]
        f = eq["forward"]
        print(f"{name:24s} {'o':6s} {'bit-identical' if f['o_identical'] else 'DIFFERS':>21s} selection "
              f"{'bit-identical' if f['selection_identical'] else 'DIFFERS'}; A/A {'bit-identical' if f['aa_o_identical'] else 'DIFFERS'}"
              f"; padded nonzero {sum(eq['padded_nonzero'].values())}")
        extra = aa["shapes"][name]["equivalence"]["backward"] if aa and aa["shapes"][name]["status"] == "ok" else {}
        for g, e in eq["backward"].items():
            floors = e["aa"] + extra.get(g, {}).get("aa", [])
            print(f"{name:24s} {g:6s} {worst(e['vs_default'], 'max_abs'):10.3e} {worst(e['vs_default'], 'rel_l2'):10.3e} "
                  f"{worst(e['vs_default'], 'n_diff'):9d} {e['vs_default'][0]['n_elements']:10d} {worst(floors, 'max_abs'):11.3e} "
                  f"{worst(floors, 'n_diff'):10d}  {gradient_verdict(e['vs_default'], floors, eq['grad_tol'])}")
        print(f"{name:24s} verdict {combined_verdict(s, aa['shapes'][name] if aa else None)}")
        if s["stale_extension_sources"]:
            print(f"{name:24s} WARNING extension older than {s['stale_extension_sources']}; the arm may not be in it")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("AA.json", "ARM.json"), help="the --arm baseline run, then the arm's run")
    ap.add_argument("--lens", type=int, nargs="+", default=list(LENS))
    ap.add_argument("--heads", type=int, nargs="+", default=list(HEADS))
    ap.add_argument("--dims", type=int, nargs="+", default=list(DIMS))
    ap.add_argument("--no-partial", action="store_true")
    ap.add_argument("--topk-frac", type=float, default=0.10, help="topk = max(1, round(frac * NB))")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--grad-tol", type=float, default=1e-3, help="largest gradient rel_l2 against the default")
    ap.add_argument("--scratch", help="directory for each shape's tensors; up to about 2.2 GB at the largest shape")
    a = ap.parse_args()
    if a.child:
        return child_main(json.loads(a.child))
    if a.compare:
        aa, arm = (json.loads(Path(p).read_text()) for p in a.compare)
        error = compare_error(aa, arm)
        if error:
            sys.exit(f"refusing to compare: {error}")
        report(arm, aa)
        return 0
    if not a.arm or not a.out:
        ap.error("--arm and --out are required, or use --compare")
    out = Path(a.out)
    if out.exists():
        ap.error(f"{out} exists; results are not overwritten")
    error = arm_error(a.arm)
    if error:
        ap.error(error)
    settings = {k: getattr(a, k) for k in ("warmup", "iters", "batches", "seed", "topk_frac", "grad_tol")}
    res = dict(schema=SCHEMA, note=NOTE, arm=a.arm, env=ARMS[a.arm], settings=settings,
               created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), shapes={})
    out.parent.mkdir(parents=True, exist_ok=True)
    for shape in shape_matrix(a.lens, a.heads, a.dims, not a.no_partial):
        res["shapes"][shape_key(shape)] = dict(run_shape(a, shape, a.scratch), shape=shape)
        tmp = out.with_suffix(out.suffix + ".tmp")  # keep what finished if a later shape stops the job
        tmp.write_text(json.dumps(res, indent=1) + "\n")
        tmp.replace(out)
    report(res)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
