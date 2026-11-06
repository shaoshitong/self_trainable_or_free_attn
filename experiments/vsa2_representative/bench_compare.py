"""Same-build baseline vs candidate timing, with one machine-readable result per comparison.

REP_TK_FWD_CFG is cached in a function-local static on first use, so one process can only ever measure
one configuration. The driver therefore starts one child per arm in ABBA order, with those variables
removed from the inherited environment and only the requested ones set, and refuses to compare children
whose extension, sources, inputs, selection, shape or timing settings differ.

Profiles (see docs/vsa2_representative_handoff.md, section 7):

  representative      all-tier fwd_ker, diag=0, warmup 10, iters 30. Baseline CFG1, candidate required; FWD_CFGS
                      says what each value runs. CFG21/22/23 change only the backward: compared with CFG1 or
                      its aliases CFG10/13, differing forward outputs make the result invalid. Kernel,
                      fine-op and full-operator scope. Output cross-checked against the generic backend.
  controlled-softmax  CFG 4/5/7, warmup 10, iters 50, kernel forward only. TIMING ONLY: these kernels run
                      tier A whatever `phases` says, write no lse and do not mask padded output.

What each number is:

  kernel            one kernel with its tables prebuilt; never a complete call, and stages are never summed.
                    Forward rows with work="tier_b_only", "tier_c_only" or "tier_subset" run part of the
                    tile stream: they locate cost, and their times do not add up to the A|B|C row.
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
No hardware counters are collected: this file reports no MFU or stall figures. Results record no node
name, address or raw device id; arms are matched on a hash of the GPU id.

  <gpu-runner> python bench_compare.py --candidate REP_TK_FWD_CFG=11 --out k-buffer.json
  <gpu-runner> python bench_compare.py --baseline REP_TK_FWD_CFG=1 --candidate REP_TK_FWD_CFG=1 --out aa.json
  <gpu-runner> python bench_compare.py --profile controlled-softmax --out cfg45.json \
      --baseline REP_TK_FWD_CFG=4 --candidate REP_TK_FWD_CFG=5
"""
import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCHEMA = "vsa2-rep-compare/1"
HERE = Path(__file__).resolve().parent
TUNABLES = ("REP_TK_FWD_CFG", "REP_TK_BWDC_SPLITS")
# What each REP_TK_FWD_CFG value runs. Only "operator" values compute the 64/4/1 operator.
FWD_CFGS = {
    "0": ("operator", "fwd_ker: TK-vector softmax, MINB 4, bulk tier-B gather"),
    "1": ("operator", "fwd_ker: TK-vector softmax, MINB 3, cp.async tier-B gather; tier-C backward split by "
                      "num_splits_heuristic; tier-A dK/dV stored by each kv block's single owner CTA, no kg/vg memset "
                      "(the kernel's default)"),
    "6": ("operator", "fwd_ker: hand-written softmax, MINB 3, bulk tier-B gather; a softmax pair with CFG20"),
    "8": ("operator", "fwd_ker: TK-vector softmax, MINB 2, bulk tier-B gather"),
    "9": ("operator", "fwd_ker: hand-written softmax, MINB 2, bulk tier-B gather"),
    "10": ("operator", "alias of CFG1, kept so old logs stay reproducible"),
    "11": ("operator", "measured negative arm, not a candidate: fwd_kdb_ker, CFG1 plus a double-buffered K tile; V unchanged"),
    "24": ("operator", "measured negative arm: fwd_pipe_ker, CFG11's K double buffer plus a two-stage softmax, the "
                       "next tile's QK issued before the current softmax finishes"),
    "32": ("operator", "measured negative arm: two CTAs per SM, K and V both double-buffered, 80 KB of tiles per CTA"),
    "33": ("operator", "measured negative arm: two CTAs per SM, K and V both triple-buffered, 112 KB of tiles per CTA, "
                       "224 KB per SM"),
    "34": ("operator", "CFG1 with the log2(e)*scale multiply and the tier-B/C bias add folded into one FFMA"),
    "35": ("operator", "CFG1 skipping the O rescale when no row's running maximum moved in this tile (__any_sync); "
                       "assumes alpha is exactly 1.0f then, unverified"),
    "13": ("operator", "alias of CFG1 since the num_splits_heuristic split became the default"),
    "20": ("operator", "the pre-round-4 forward: per-slot cp.async.bulk tier-B gather; CFG1's backward"),
    "21": ("operator", "CFG1 forward; tier-C backward split by the old wave policy"),
    "22": ("operator", "CFG1 forward; tier-C backward split forced to 4"),
    "23": ("operator", "CFG1 forward; tier-A backward __launch_bounds__(128, 3) occupancy probe, on the reduce-add path "
                       "of CFG37"),
    "36": ("operator", "alias of CFG1 since the single-writer dK/dV store became the default"),
    "37": ("operator", "CFG1 forward; the pre-round-11 tier-A backward: zeros(kg), zeros(vg) and the reduce-add kernel"),
    "38": ("operator", "CFG1 with the representative-table build overlapped with route_tiers on a second stream"),
    "39": ("operator", "CFG1 with the shape-only coarse CSR metadata cached across calls"),
    "40": ("operator", "pre-round-16 behaviour: the coarse CSR metadata is rebuilt on every call"),
    "2": ("tier_a_path", "tier-A-only specialisation at MINB 4; other phases run the default kernel"),
    "3": ("tier_a_path", "tier-A-only specialisation at MINB 3; other phases run the default kernel"),
    "4": ("timing_only", "copied native loop, native softmax"),
    "5": ("timing_only", "copied native loop, previous hand-written softmax"),
    "7": ("timing_only", "copied native loop, TK-vector softmax"),
}
PROFILES = {
    "representative": dict(warmup=10, iters=30, kind="operator", baseline="REP_TK_FWD_CFG=1", candidate=None),
    "controlled-softmax": dict(warmup=10, iters=50, kind="timing_only", baseline="REP_TK_FWD_CFG=4",
                               candidate="REP_TK_FWD_CFG=5"),
}
BACKWARD_ARMS = {"21": "C", "22": "C", "23": "A", "37": "A"}  # the only backward tier each one's setting reaches
ALIASES = ("1", "10", "13", "36", "38", "39", "40")  # no dispatch line of their own needed
OWN_FORWARD = ("20", "24", "32", "33")  # meaningless if they launch the default kernel
RUNTIME_ARMS = {"34", "35"}  # deliberately the default kernel; a host-set diag flag selects the behaviour
DISTINCT_FROM = {"24": ("11",), "32": ("11", "24"), "33": ("11", "24", "32"), "35": ("34",)}  # nor if they repeat an arm
TWO_CTA = ("32", "33")  # their tiles leave no room for a third CTA; see launch_bound
BYTE_EQUAL_PREMISE = ("35",)  # skipping a rescale by exactly 1.0f must not change a byte
FORWARD_EQUAL = {*ALIASES, *BACKWARD_ARMS}  # one forward kernel by definition; see forward_must_match
BACKWARD_READ_ROWS = ("kernel.tk.bwd.A", "kernel.tk.bwd.C", "fine_op.tk.A|B|C.b_direct", "full_op.tk.routed.b_direct",
                      "fine_op.tk.A|B|C.f_plus_b", "full_op.tk.routed.f_plus_b")
HISTORICAL_LIMIT = "historical H200 measurements are from other hardware and must not be compared with this run"
CONTROLLED_LIMITS = [
    "timing only: CFG4/5/7 are not a numerical reference and the output check is not run",
    "CFG4/5/7 ignore `phases` and run tier A only; this is not the 64/4/1 operator",
    "CFG4/5/7 write no lse and do not mask padded output; query blocks with an empty tier-A list leave o unwritten",
    "cannot establish numerical or end-to-end acceleration",
    "the older roughly 0.108 ms CFG4/CFG5 figure came from a CFG5 that normalized only part of o; "
    "it is confounded and superseded only by a rerun",
    HISTORICAL_LIMIT,
]
# Checked at the run's shape, with the tolerances of tests/test_gpu_tk.py (test_native_and_generic,
# test_backward). A cross-check of this configuration, not a substitute for the three test suites.
TOL_O, TOL_LSE, TOL_GRAD = 1e-2, 1e-4, 2e-2


# ------------------------------------------------------------ driver: no torch, never touches the GPU
def fwd_dispatch(source):
    """{cfg value: launched kernel, "default": the fallback} as read from fwd_impl; {} if unreadable."""
    body = re.search(r"static void fwd_impl\(.*?^}", source, re.S | re.M)
    if not body:
        return {}
    kernel = r"launch\((rep_tk::\w+<[^;]*?>)\)"
    table = dict(re.findall(rf"cfg == (\d+)[^\n]*?\)\s*{kernel}", body.group(0)))
    table.update(re.findall(r"cfg == (\d+)\)\s*\{[^}]*?(rep_tk::\w+<[^;()]*?>)", body.group(0)))  # a braced branch
    default = re.search(rf"else {kernel};", body.group(0))
    return dict(table, default=default.group(1)) if default else {}


def cfg_error(value, source):
    """Why this rep_tk.cu cannot run `value` as FWD_CFGS describes it, or None.

    An undispatched value silently runs the default kernel, and FWD_CFGS describes the kernel that has the
    CFG20 history arm: on an older source, CFG1 would be labelled with a gather it does not have.
    """
    if value not in FWD_CFGS:
        return f"REP_TK_FWD_CFG={value} is not in the config table"
    if not re.search(r"\bcfg == 20\b", source):
        return "this kernel source has no CFG20 history arm; the config table does not describe it"
    if value not in ALIASES and value not in BACKWARD_ARMS and not re.search(rf"\w*cfg\w* == {value}\b", source):
        return f"REP_TK_FWD_CFG={value} is not implemented by this kernel source"
    # A backward-only arm has no forward dispatch line: its evidence is the backward code that reads the
    # config. Checking that token keeps the guard honest instead of exempting the arm outright.
    if value in BACKWARD_ARMS and not re.search(rf"rep_cfg\(\) [=!]= {value}\b", source):
        return f"REP_TK_FWD_CFG={value} is not implemented by the backward path"
    # A runtime-flag arm launches the default kernel on purpose, so the dispatch text cannot distinguish it;
    # the evidence is the flag assignment next to the config number.
    if value in RUNTIME_ARMS:
        if not re.search(rf"cfg == {value}\b", source) or not re.search(r"g\.diag \|=", source):
            return f"REP_TK_FWD_CFG={value} is not implemented by a diag flag in this source"
        table = fwd_dispatch(source) or {}
        if table and table.get(value, table.get("default")) != table.get("default"):
            return f"REP_TK_FWD_CFG={value} must launch the default kernel and select by flag"
        return None
    if value in ("10", "13", "36", *OWN_FORWARD, *BACKWARD_ARMS):
        table = fwd_dispatch(source)
        if not table:
            return f"cannot read the forward dispatch to check REP_TK_FWD_CFG={value}"
        same = table.get(value, table["default"]) == table["default"]
        if same != (value not in OWN_FORWARD):
            return f"REP_TK_FWD_CFG={value} launches {table.get(value, table['default'])}, default {table['default']}"
    table = fwd_dispatch(source)
    for other in DISTINCT_FROM.get(value, ()):
        if table.get(value) == table.get(other):
            return f"REP_TK_FWD_CFG={value} launches the same kernel as CFG{other}"
    if value in TWO_CTA and launch_bound(table.get(value), source) != "2":
        return f"REP_TK_FWD_CFG={value} is a two-CTA arm; {table.get(value)} has launch bound " \
               f"{launch_bound(table.get(value), source)}"
    return None


def launch_bound(kernel, source):
    """The minimum blocks in `kernel`'s __launch_bounds__, as written in the source, or None."""
    name = re.match(r"rep_tk::(\w+)", kernel or "")
    found = name and re.search(rf"__launch_bounds__\(\s*\d+\s*,\s*([^)]*?)\s*\)\s*void\s+{name.group(1)}\b", source)
    if found and found.group(1) == "MINB":  # a template argument, as in fwd_ker<D, 3, ...>
        arg = re.search(r"<\s*D\s*,\s*(\d+)", kernel)
        return arg.group(1) if arg else None
    return found.group(1) if found else None


def fwd_kernel_name(value, source):
    """The kernel a profiler should select for this value's forward, without its template arguments."""
    table = fwd_dispatch(source)
    return re.match(r"rep_tk::(\w+)", table.get(value, table["default"])).group(1)


def forward_must_match(configs):
    """Two different arms that are one forward kernel by definition, so their outputs must be byte-equal."""
    arms = {c.get("REP_TK_FWD_CFG") for c in configs.values()}
    return len(arms) == 2 and arms <= FORWARD_EQUAL


def arms_error(configs):
    """One variable per comparison, for the arms whose meaning depends on their partner."""
    arms = {c.get("REP_TK_FWD_CFG") for c in configs.values()}
    backward = arms & BACKWARD_ARMS.keys()
    if backward and not arms <= FORWARD_EQUAL:
        return f"CFG{min(backward)} changes only the backward; compare it with CFG1, not {sorted(arms)}"
    if "23" in arms and arms != {"23"} and arms != {"23", "37"}:
        return "CFG23 runs on CFG37's reduce-add path, so only CFG37 vs CFG23 changes one thing"
    if len({BACKWARD_ARMS[a] for a in backward}) > 1:
        return f"{sorted(backward)} change different things in the backward; compare each with CFG1"
    if any(c.get("REP_TK_FWD_CFG") in BACKWARD_ARMS and "REP_TK_BWDC_SPLITS" in c for c in configs.values()):
        return "REP_TK_BWDC_SPLITS would override or add to what a backward arm measures"
    return None


def backward_row_is_control(tier, env):
    """Backward kernels read neither forward setting; a backward arm or a forced split reaches only its tier."""
    reached = BACKWARD_ARMS.get(env.get("REP_TK_FWD_CFG"))
    return tier != reached and not (tier == "C" and "REP_TK_BWDC_SPLITS" in env)


def active_fwd_cfg(env=os.environ, source=None):
    """(value, kind, meaning) of this process's REP_TK_FWD_CFG, for the standalone scripts."""
    value = env.get("REP_TK_FWD_CFG", "1")
    source = (HERE / "csrc/rep_tk/rep_tk.cu").read_text() if source is None else source
    error = cfg_error(value, source)
    if error:
        raise SystemExit(error)
    return (value,) + FWD_CFGS[value]


def parse_config(text, profile):
    """'K=V,K=V' -> dict, restricted to the tunables and to the values the profile's semantics allow."""
    cfg = {}
    for item in filter(None, (text or "").split(",")):
        key, sep, val = item.partition("=")
        if not sep or key not in TUNABLES:
            raise ValueError(f"bad setting {item!r}: expected KEY=VALUE with KEY in {TUNABLES}")
        cfg[key] = val
    allowed = [v for v, (kind, _) in FWD_CFGS.items() if kind == PROFILES[profile]["kind"]]
    if cfg.get("REP_TK_FWD_CFG") not in allowed:
        raise ValueError(f"profile {profile} needs REP_TK_FWD_CFG in {allowed}, got {cfg.get('REP_TK_FWD_CFG')!r}")
    splits = cfg.get("REP_TK_BWDC_SPLITS")
    if splits is not None and splits != "wave" and not (splits.isdigit() and int(splits) > 0):
        raise ValueError(f"REP_TK_BWDC_SPLITS must be positive or wave, got {splits!r}")
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
            arms = rows.setdefault(row["id"], {"meta": dict(row["meta"]), "baseline": [], "candidate": []})
            arms["meta"]["control"] = arms["meta"]["control"] and row["meta"]["control"]  # in every arm
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


def mark_shared_forward(summary, configs):
    """Between arms sharing a forward kernel, forward-only rows are controls; byte-equal output is checked apart."""
    if forward_must_match(configs):
        for s in summary:
            if s["stage"] in ("fwd", "f"):
                s["control"] = True
    return summary


def claims(profile, children):
    """What this result may support; printed and stored so that a reader of the JSON cannot miss it."""
    if profile == "controlled-softmax":
        return dict(output_validated=False, may_claim_op_level_change=False, limits=CONTROLLED_LIMITS)
    ok = all(c.get("output_check") and all(x["ok"] for x in c["output_check"]) for c in children)
    return dict(output_validated=ok, may_claim_op_level_change=ok,
                limits=["needs the CPU, generic and TK test suites on this build as well",
                        "tier_a_only rows are not the 64/4/1 operator; their ratios are not efficiency claims",
                HISTORICAL_LIMIT])


def assemble(args_dict, profile, configs, order, children):
    bad = identity_differences(children)
    claim = claims(profile, children)
    digests = [c.get("forward_output") for c in children]
    identical = None if None in digests else len({json.dumps(d, sort_keys=True) for d in digests}) == 1
    if forward_must_match(configs) and identical is not True:
        bad = bad + ["forward outputs of arms defined to share one forward kernel"]
    if {c.get("REP_TK_FWD_CFG") for c in configs.values()} & BACKWARD_ARMS.keys():
        ids = {row["id"] for c in children for row in c["rows"]}
        bad = bad + [f"missing row {r}" for r in BACKWARD_READ_ROWS if r not in ids]
    if bad:
        claim["may_claim_op_level_change"] = False
    arms = {c.get("REP_TK_FWD_CFG") for c in configs.values()}
    if arms & set(BYTE_EQUAL_PREMISE) and arms & set(ALIASES):
        if identical is not True:
            claim["may_claim_op_level_change"] = False
        claim["limits"] = claim["limits"] + [
            f"alpha premise: forward outputs byte-identical to CFG1: {identical}. This resolves bf16 o and fp32 lse only, "
            "so an alpha within bf16 rounding of 1.0f passes; it does not show alpha is exactly 1.0f"]
    return dict(schema=SCHEMA, profile=profile, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                settings=args_dict, configs=configs, run_order=order,
                status="invalid: arms differ in " + ", ".join(bad) if bad else "ok",
                identity_differences=bad, claims=claim, forward_outputs_byte_identical=identical,
                counters=dict(collected=False, reason="not collected by this harness; no counter workaround is authorized"),
                summary=[] if bad else mark_shared_forward(summarize(children), configs), children=children)


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
    print("f_plus_b: measured F+B | b_direct: measured backward alone | ESTIMATE: F+B - F, not a timed interval")
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
    try:
        configs = dict(baseline=parse_config(a.baseline or p["baseline"], a.profile),
                       candidate=parse_config(a.candidate or p["candidate"], a.profile))
    except ValueError as e:
        ap.error(str(e))
    source = (HERE / "csrc/rep_tk/rep_tk.cu").read_text()
    for arm, config in configs.items():
        error = cfg_error(config["REP_TK_FWD_CFG"], source)
        if error:
            ap.error(f"{arm}: {error}")
    error = arms_error(configs)
    if error:
        ap.error(error)
    order = abba(a.rounds)
    children = [run_child(a, arm, configs[arm], i) for i, arm in enumerate(order)]
    settings = {k: getattr(a, k) for k in ("len", "heads", "dim", "mid_reps", "seed", "warmup", "iters", "batches", "rounds")}
    settings["fwd_cfg_meanings"] = {arm: FWD_CFGS[c["REP_TK_FWD_CFG"]][1] for arm, c in configs.items()}
    dispatch = fwd_dispatch(source)
    settings["fwd_kernels_in_source"] = {arm: dispatch.get(c["REP_TK_FWD_CFG"], dispatch.get("default"))
                                         for arm, c in configs.items()}
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


def smi_target():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return ["-i", cvd] if cvd and "," not in cvd else []


def telemetry():
    """Plain nvidia-smi state of the visible GPU (not performance counters), around each child."""
    fields = "pstate,clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw,utilization.gpu,memory.used"
    line = capture("nvidia-smi", *smi_target(), f"--query-gpu={fields}", "--format=csv,noheader")
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
    if torch.cuda.device_count() != 1:
        sys.exit(f"needs exactly one visible GPU, got {torch.cuda.device_count()}")

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
    output_check = forward_output = None
    if spec["profile"] == "representative":
        phases = {"A": PHASE_A}
        for name, ph, work in (("A|B", PHASE_A | PHASE_B, "tier_subset"), ("A|B|C", PHASE_ALL, "64/4/1"),
                               ("B|C", PHASE_B | PHASE_C, "tier_subset"), ("B", PHASE_B, "tier_b_only"),
                               ("C", PHASE_C, "tier_c_only")):
            phases[name] = ph
            add(f"kernel.tk.fwd.{name}", "kernel", "fwd", "tk", work, False, loop(kernel_fwd(ph), W, I, N))
        gtab = C.rep_build(k, v, cnt, rm)
        add("kernel.generic.fwd.A|B|C", "kernel", "fwd", "generic", "64/4/1", True,
            loop(lambda: C.rep_attn_fwd(q, k, v, *gtab, qp, qi, sel.bits_a, sel.bits_ab, cnt, rm, PHASE_ALL, 0), W, I, N))
        add("kernel.tk.tables", "kernel", "tables", "tk", "64/4/1", True,
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
            control = backward_row_is_control(name, os.environ)
            add(f"kernel.tk.bwd.{name}", "kernel", "bwd_stage", "tk", work, control, loop(fn, W, I, N),
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
        forward_output = {}
        for name, ph in phases.items():
            o_p, lse_p = kernel_fwd(ph)()
            forward_output[name] = dict(o=digest(o_p), lse=digest(lse_p))
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
        hardware=dict(name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                      uuid_sha256=hashlib.sha256(str(getattr(props, "uuid", "")).encode()).hexdigest(), total_memory=props.total_memory,
                      multi_processor_count=getattr(props, "multi_processor_count", None),
                      driver=capture("nvidia-smi", *smi_target(), "--query-gpu=driver_version", "--format=csv,noheader")),
        software=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                      nvcc=(capture("nvcc", "--version") or "").splitlines()[1:2]),
        # The extension's hash is the build identity. The source hashes say what was on disk at run time:
        # nothing here can prove that those bytes are what the extension was compiled from.
        build=dict(extension_sha256=sha256_file(C.__file__), extension=os.path.basename(C.__file__),
                   build_log_sha256=sha256_file(HERE / "build.log") if (HERE / "build.log").exists() else None),
        source=dict(package=tree_sha256(HERE, ("csrc/**/*", "vsa2_rep/*.py", "setup.py", "bench_compare.py")),
                    rep_tk_sha256=sha256_file(HERE / "csrc/rep_tk/rep_tk.cu"),
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
                                                 output_check=output_check, forward_output=forward_output, telemetry_after=telemetry())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
