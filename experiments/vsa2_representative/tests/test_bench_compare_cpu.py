"""CPU tests of what bench_compare.py lets a result claim. No torch, GPU or build.

Run: python3 tests/test_bench_compare_cpu.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench_stages import attribute, compare_stages
from bench_compare import (BACKWARD_READ_ROWS, launch_bound, FWD_CFGS, PROFILES, abba, active_fwd_cfg, arms_error, assemble,
                           backward_row_is_control, cfg_error, claims, forward_must_match, fwd_dispatch, fwd_kernel_name,
                           parse_config, summarize)


def child(arm, scale, check=True):
    meta = dict(scope="fine_op", implementation="tk", work="64/4/1", control=False, measured=True)
    return dict(arm=arm, identity=dict(build=dict(extension_sha256="x"), inputs=dict(q="y")),
                output_check=[dict(name="o", ok=check)],
                rows=[dict(id="op.f", ms=[1.0 * scale] * 3, meta=dict(meta, stage="f", complete_call=True)),
                      dict(id="op.f_plus_b", ms=[3.0 * scale] * 3, meta=dict(meta, stage="f_plus_b", complete_call=True)),
                      dict(id="k.a", ms=[0.5] * 3, meta=dict(meta, scope="kernel", stage="bwd_stage", complete_call=False)),
                      dict(id="k.b", ms=[0.5] * 3, meta=dict(meta, scope="kernel", stage="bwd_stage", complete_call=False))])


def rejected(text, profile):
    try:
        parse_config(text, profile)
    except ValueError:
        return True
    return False


def test_configs_keep_their_semantics():
    assert parse_config("REP_TK_FWD_CFG=1,REP_TK_BWDC_SPLITS=8", "representative") == \
        {"REP_TK_FWD_CFG": "1", "REP_TK_BWDC_SPLITS": "8"}
    for cfg in "23457":
        assert rejected(f"REP_TK_FWD_CFG={cfg}", "representative"), "a diagnostic passed as production"
    for cfg in ("0", "1", "2", "6", "8", "9", "10", "11", "13", "20", "21", "22", "23", "24", "32", "33", "34", "35"):
        assert rejected(f"REP_TK_FWD_CFG={cfg}", "controlled-softmax"), "only CFG 4/5/7 are this diagnostic"
    for cfg in "457":
        assert parse_config(f"REP_TK_FWD_CFG={cfg}", "controlled-softmax")
    for cfg in ("8", "9", "10", "11", "13", "20", "21", "22", "23", "24", "32", "33", "34", "35"):
        assert parse_config(f"REP_TK_FWD_CFG={cfg}", "representative")
    assert rejected("REP_TK_FWD_CFG=25", "representative"), "the kernel would silently run its default path"
    assert parse_config("REP_TK_FWD_CFG=1,REP_TK_BWDC_SPLITS=wave", "representative"), "the kernel accepts wave"
    assert rejected("REP_TK_FWD_CFG=1,REP_TK_BWDC_SPLITS=other", "representative")
    assert PROFILES["representative"]["baseline"].endswith("=1") and PROFILES["representative"]["candidate"] is None
    assert rejected("", "representative") and rejected("OTHER=1,REP_TK_FWD_CFG=1", "representative")
    assert rejected("REP_TK_FWD_CFG=1,REP_TK_BWDC_SPLITS=0", "representative")


def kernel(*lines):
    return "static void fwd_impl(int x) {\n  if (cfg == 4) launch(rep_tk::nat_copy_ker<D, 0>);\n" + \
        "".join(f"  else if {line}\n" for line in lines) + "  else launch(rep_tk::fwd_ker<D, 3, false, false, true>);\n}\n"


BULK = "(cfg == 20) launch(rep_tk::fwd_ker<D, 3>);"
DOUBLE = "(cfg == 11) launch(rep_tk::fwd_ker<D, 3, false, false, true, true>);"
ROUND5 = kernel(BULK, DOUBLE)
ROUND6 = ROUND5 + ("int bwd(int bwd_cfg) { return bwd_cfg == 21 ? 1 : bwd_cfg == 22 ? 4 : bwd_cfg == 23 ? 3 : 0; }\n"
                   "const char* split() { return rep_cfg() == 21 ? \"wave\" : rep_cfg() == 22 ? \"4\" : 0; }\n"
                   "int minb() { return rep_cfg() == 23 ? 8 : 0; }\n")


def test_standalone_scripts_see_what_runs():
    assert active_fwd_cfg({}, ROUND5)[:2] == ("1", "operator")
    assert active_fwd_cfg({"REP_TK_FWD_CFG": "7"}, ROUND5 + "cfg == 7")[1] == "timing_only"
    assert fwd_dispatch(ROUND5)["20"] == "rep_tk::fwd_ker<D, 3>"
    for value, source in (("1", kernel()), ("11", kernel(BULK)), ("6", ROUND5), ("21", ROUND5), ("24", ROUND6), ("25", ROUND6)):
        try:
            active_fwd_cfg({"REP_TK_FWD_CFG": value}, source)
        except SystemExit:
            continue
        raise AssertionError(f"CFG{value} accepted")


def test_arm_meanings_hold_in_the_source():
    assert cfg_error("10", ROUND5) is None, "an alias needs no dispatch line of its own"
    assert cfg_error("10", kernel(BULK, "(cfg == 10) launch(rep_tk::fwd_ker<D, 3, false, false, true>);")) is None
    assert cfg_error("10", kernel(BULK, "(cfg == 10) launch(rep_tk::fwd_ker<D, 3>);")), "an alias that differs"
    assert cfg_error("13", ROUND5) is None, "the heuristic is the default now"
    braced = kernel(BULK, "(cfg == 11) {\n    set(rep_tk::fwd_kdb_ker<D>, mem);\n    rep_tk::fwd_kdb_ker<D><<<g>>>(x);\n  }")
    assert fwd_dispatch(braced)["11"] == "rep_tk::fwd_kdb_ker<D>", "a braced branch must not read as the default"
    for value in ("13", "21", "22", "23"):
        change = kernel(BULK, f"(cfg == {value}) launch(rep_tk::fwd_ker<D, 4>);")
        assert cfg_error(value, change), f"forward change under backward arm CFG{value}"
        assert cfg_error(value, kernel(BULK, f"(cfg == {value}) {{\n    rep_tk::fwd_kdb_ker<D><<<g>>>(x);\n  }}"))
    for value in ("21", "22", "23"):
        assert cfg_error(value, ROUND6) is None
        assert cfg_error(value, ROUND5), "must not pass before the kernel has it"
    assert cfg_error("20", ROUND5) is None
    same = "(cfg == 20) launch(rep_tk::fwd_ker<D, 3, false, false, true>);"
    assert cfg_error("20", kernel(same)), "a history arm that is the default"
    assert cfg_error("13", ROUND5.replace("static void fwd_impl", "void renamed")), "must fail closed"


def test_two_stage_arm_is_its_own_forward():
    kdb = "(cfg == 11) {\n    set(rep_tk::fwd_kdb_ker<D>, m);\n    rep_tk::fwd_kdb_ker<D><<<g>>>(x);\n  }"
    two = "(cfg == 24) {\n    constexpr int m = 2;\n    set(rep_tk::fwd_kdb_ker<D, true>, m);\n    rep_tk::fwd_kdb_ker<D, true><<<g>>>(x);\n  }"
    assert cfg_error("24", kernel(BULK, kdb, two)) is None
    assert fwd_kernel_name("24", kernel(BULK, kdb, two)) == "fwd_kdb_ker", "a profiler selecting fwd_ker would miss it"
    assert fwd_kernel_name("1", kernel(BULK, kdb, two)) == "fwd_ker"
    assert cfg_error("24", kernel(BULK, kdb, "(cfg == 24) {\n    rep_tk::fwd_kdb_ker<D><<<g>>>(x);\n  }")), "CFG11 again"
    assert cfg_error("24", kernel(BULK, "(cfg == 24) launch(rep_tk::fwd_ker<D, 3, false, false, true>);")), "the default"
    pair = {"baseline": {"REP_TK_FWD_CFG": "1"}, "candidate": {"REP_TK_FWD_CFG": "24"}}
    assert not forward_must_match(pair)
    a, b = child("baseline", 1), child("candidate", 1)
    a["forward_output"], b["forward_output"] = {"B": dict(o="x")}, {"B": dict(o="y")}
    res = assemble({}, "representative", pair, [], [a, b])
    assert res["status"] == "ok" and res["forward_outputs_byte_identical"] is False, "its forward may differ"
    assert {s["id"]: s["control"] for s in res["summary"]}["op.f"] is False
    assert res["claims"]["may_claim_op_level_change"] is True
    b["output_check"] = [dict(name="o", ok=False)]
    assert assemble({}, "representative", pair, [], [a, b])["claims"]["output_validated"] is False, "the generic check still gates it"
    assert arms_error({"baseline": {"REP_TK_FWD_CFG": "24"}, "candidate": {"REP_TK_FWD_CFG": "21"}}), "two variables"
    assert arms_error({"baseline": {"REP_TK_FWD_CFG": "11"}, "candidate": {"REP_TK_FWD_CFG": "24"}}) is None


def deep(*branches, bound="2"):
    return kernel(BULK, *branches) + f"__global__ __launch_bounds__(128, {bound}) void fwd_deep_ker(const G g) {{}}\n"


def test_deep_pipeline_arms_are_two_cta_forwards():
    two = "(cfg == 32) {\n    rep_tk::fwd_deep_ker<D, 2><<<g>>>(x);\n  }"
    three = "(cfg == 33) {\n    rep_tk::fwd_deep_ker<D, 3><<<g>>>(x);\n  }"
    assert cfg_error("32", deep(two, three)) is None and cfg_error("33", deep(two, three)) is None
    assert fwd_kernel_name("33", deep(two, three)) == "fwd_deep_ker"
    assert cfg_error("33", deep(two)), "must not pass before the kernel has it"
    assert cfg_error("33", deep(two, "(cfg == 33) {\n    rep_tk::fwd_deep_ker<D, 2><<<g>>>(x);\n  }")), "CFG32 again"
    assert cfg_error("32", deep("(cfg == 32) launch(rep_tk::fwd_ker<D, 3, false, false, true>);")), "the default"
    assert "launch bound 3" in cfg_error("32", deep(two, three, bound="3")), "a three-CTA kernel under a two-CTA label"
    assert cfg_error("32", kernel(BULK, two)), "a bound that cannot be read must not pass"
    for value in ("32", "33"):
        pair = {"baseline": {"REP_TK_FWD_CFG": "1"}, "candidate": {"REP_TK_FWD_CFG": value}}
        assert not forward_must_match(pair) and arms_error(pair) is None
        a, b = child("baseline", 1), child("candidate", 1)
        a["forward_output"], b["forward_output"] = {"B": dict(o="x")}, {"B": dict(o="y")}
        assert assemble({}, "representative", pair, [], [a, b])["status"] == "ok", "their forward may differ"
        b["output_check"] = [dict(name="o", ok=False)]
        assert assemble({}, "representative", pair, [], [a, b])["claims"]["output_validated"] is False
    assert arms_error({"baseline": {"REP_TK_FWD_CFG": "33"}, "candidate": {"REP_TK_FWD_CFG": "21"}}), "two variables"


def test_softmax_reductions_stay_on_the_default_path():
    bound = "template <int D, int MINB> __global__ __launch_bounds__(128, MINB) void fwd_ker(const G g) {}\n"
    fold = "(cfg == 34) g.diag |= rep_tk::kDiagFma;"
    skip = "(cfg == 35) g.diag |= rep_tk::kDiagSkipRescale;"
    src = kernel(BULK, fold, skip) + bound
    assert cfg_error("34", src) is None and cfg_error("35", src) is None
    assert cfg_error("34", kernel(BULK, skip) + bound), "must not pass before the kernel has it"
    assert cfg_error("34", kernel(BULK, "(cfg == 34) launch(rep_tk::fwd_kdb_ker<D, 9>);") + bound), \
        "a separate kernel is not the default path"
    assert cfg_error("34", kernel(BULK, "(cfg == 34) ;") + bound), "a branch with no flag assignment is no evidence"

def test_backward_arm_is_single_variable():
    cfg = lambda v, **kw: dict(REP_TK_FWD_CFG=v, **kw)
    pair = lambda a, b: {"baseline": a, "candidate": b}
    for value in ("21", "22", "23"):
        assert arms_error(pair(cfg("1"), cfg(value))) is None
        assert arms_error(pair(cfg("11"), cfg(value))), "a forward change alongside a backward arm"
        assert arms_error(pair(cfg("1"), cfg(value, REP_TK_BWDC_SPLITS="4"))), "forced split on top of a backward arm"
    assert arms_error(pair(cfg("21"), cfg("22"))) is None, "two tier-C split policies are one variable"
    assert arms_error(pair(cfg("21"), cfg("23"))), "tier C and tier A at once"
    assert arms_error(pair(cfg("1"), cfg("1", REP_TK_BWDC_SPLITS="wave"))) is None


def test_backward_arm_reaches_only_its_tier():
    env = lambda v, **kw: dict(REP_TK_FWD_CFG=v, **kw)
    for value in ("21", "22"):
        assert not backward_row_is_control("C", env(value)) and backward_row_is_control("A", env(value))
    assert not backward_row_is_control("A", env("23")) and backward_row_is_control("C", env("23"))
    assert backward_row_is_control("B", env("23")) and backward_row_is_control("C", env("13")), "13 is an alias"
    assert not backward_row_is_control("C", env("1", REP_TK_BWDC_SPLITS="4"))


def test_forward_change_under_backward_arm_is_invalid():
    configs = {"baseline": {"REP_TK_FWD_CFG": "1"}, "candidate": {"REP_TK_FWD_CFG": "21"}}
    a, b = child("baseline", 1), child("candidate", 1)
    meta = dict(a["rows"][0]["meta"], stage="bwd_stage")
    for c in (a, b):
        c["rows"] += [dict(id=r, ms=[1.0], meta=meta) for r in BACKWARD_READ_ROWS]
    res = assemble({}, "representative", configs, [], [a, b])
    assert res["status"].startswith("invalid"), "unrecorded forward output must not pass"
    a["forward_output"], b["forward_output"] = {"B": dict(o="x")}, {"B": dict(o="x")}
    res = assemble({}, "representative", configs, [], [a, b])
    assert res["status"] == "ok" and res["forward_outputs_byte_identical"] is True
    assert {s["id"]: s["control"] for s in res["summary"]}["op.f"] is True
    assert {s["id"]: s["control"] for s in res["summary"]}["op.f_plus_b"] is False
    b["forward_output"] = {"B": dict(o="y")}
    res = assemble({}, "representative", configs, [], [a, b])
    assert res["status"].startswith("invalid") and res["claims"]["may_claim_op_level_change"] is False
    b["forward_output"] = {"B": dict(o="x")}
    for c in (a, b):
        c["rows"] = [r for r in c["rows"] if r["id"] != "kernel.tk.bwd.C"]
    assert "missing row kernel.tk.bwd.C" in assemble({}, "representative", configs, [], [a, b])["status"]
    assert assemble({}, "representative", {**configs, "candidate": {"REP_TK_FWD_CFG": "13"}}, [], [a, b])["status"] == "ok"
    other = {"baseline": {"REP_TK_FWD_CFG": "1"}, "candidate": {"REP_TK_FWD_CFG": "11"}}
    assert assemble({}, "representative", other, [], [a, b])["status"] == "ok", "CFG11 may change bytes; only reported"


def test_stage_runs_compare_only_within_one_build():
    run = lambda cfg, ext, ms: dict(fwd_cfg=cfg, build=dict(extension_sha256=ext), inputs={}, shape={}, timing={}, hardware={},
                                    rows=[dict(id="fine", ms=[ms], reads_fwd_cfg=True)])
    assert compare_stages(run("1", "x", 1.0), run("11", "x", 0.5))["rows"][0]["b_over_a"] == 0.5
    try:
        compare_stages(run("1", "x", 1.0), run("11", "y", 0.5))
    except ValueError:
        return
    raise AssertionError("runs from two builds compared")


def test_abba_is_balanced():
    assert abba(2) == ["baseline", "candidate", "candidate", "baseline"]


def test_estimate_is_labelled_and_stages_are_not_summed():
    rows = {s["id"]: s for s in summarize([child("baseline", 1), child("candidate", 2)])}
    est = rows["op.b_estimate"]
    assert est["measured"] is False and est["complete_call"] is False
    assert est["b_estimate_ms"] == {"baseline": 2.0, "candidate": 4.0}
    assert rows["op.f_plus_b"]["measured"] is True and rows["op.f_plus_b"]["baseline"]["median_ms"] == 3.0
    assert rows["k.a"]["complete_call"] is False
    assert set(rows) == {"op.f", "op.f_plus_b", "op.b_estimate", "k.a", "k.b"}, "a derived row appeared"


def test_row_is_a_control_only_if_it_is_in_every_arm():
    a, b = child("baseline", 1), child("candidate", 1)
    a["rows"][2]["meta"]["control"] = True  # a tier-C row, with REP_TK_BWDC_SPLITS set only in the candidate
    b["rows"][3]["meta"]["control"] = True
    rows = {s["id"]: s for s in summarize([a, b])}
    assert rows["k.a"]["control"] is False and rows["k.b"]["control"] is False
    a["rows"][3]["meta"]["control"] = True
    assert {s["id"]: s for s in summarize([a, b])}["k.b"]["control"] is True


def test_differing_arms_are_invalid():
    a, b = child("baseline", 1), child("candidate", 1)
    other = copy.deepcopy(b)
    other["identity"]["build"]["extension_sha256"] = "z"
    res = assemble({}, "representative", {}, [], [a, other])
    assert res["status"].startswith("invalid") and res["summary"] == []
    assert res["identity_differences"] == ["build.extension_sha256"]
    assert res["claims"]["may_claim_op_level_change"] is False
    assert assemble({}, "representative", {}, [], [a, b])["status"] == "ok"


def test_byte_identity_is_reported_only_when_recorded():
    a, b = child("baseline", 1), child("candidate", 1)
    assert assemble({}, "representative", {}, [], [a, b])["forward_outputs_byte_identical"] is None
    a["forward_output"], b["forward_output"] = dict(o="x", lse="y"), dict(o="x", lse="y")
    assert assemble({}, "representative", {}, [], [a, b])["forward_outputs_byte_identical"] is True
    b["forward_output"] = dict(o="z", lse="y")
    assert assemble({}, "representative", {}, [], [a, b])["forward_outputs_byte_identical"] is False


def stage_rows(residual):
    row = lambda n, ms, in_sum: dict(id=n, ms=[ms, ms, ms], in_sum=in_sum)
    return [dict(id="total", ms=[10.0, 10.1, 10.0], in_sum=False), row("pool", 1.0, True),
            row("route", 3.0, True), row("route.one_budget", 2.0, False), row("fine", 6.0 - residual, True),
            row("fine.tables", 0.5, False), dict(id="total.repeat", ms=[10.0, 9.9, 10.0], in_sum=False)]


def test_stage_attribution_counts_each_part_once():
    t = attribute(stage_rows(0.0))
    assert t["measured"] is False and "measurement_only" in t["note"]
    assert t["parts"] == ["pool", "route", "fine"], "a sub-row or the total was summed"
    assert t["sum_of_parts_ms"] == 10.0 and t["sum_within_floor"] is True
    assert t["outside_fine_ms"] == 4.0
    t = attribute(stage_rows(1.0))
    assert t["residual_ms"] == 1.0 and t["sum_within_floor"] is False, "an unexplained residual must show"


def test_claims():
    good = [child("baseline", 1), child("candidate", 1)]
    assert claims("representative", good)["may_claim_op_level_change"] is True
    assert claims("representative", [good[0], child("candidate", 1, check=False)])["output_validated"] is False
    # timing-only, even if an output check were present and passing
    c = claims("controlled-softmax", good)
    assert c["output_validated"] is False and c["may_claim_op_level_change"] is False
    text = " ".join(c["limits"])
    assert "0.108" in text and "rerun" in text and "H200" in text


if __name__ == "__main__":
    test_configs_keep_their_semantics()
    test_standalone_scripts_see_what_runs()
    test_arm_meanings_hold_in_the_source()
    test_two_stage_arm_is_its_own_forward()
    test_deep_pipeline_arms_are_two_cta_forwards()
    test_softmax_reductions_stay_on_the_default_path()
    test_backward_arm_is_single_variable()
    test_backward_arm_reaches_only_its_tier()
    test_forward_change_under_backward_arm_is_invalid()
    test_stage_runs_compare_only_within_one_build()
    test_abba_is_balanced()
    test_estimate_is_labelled_and_stages_are_not_summed()
    test_row_is_a_control_only_if_it_is_in_every_arm()
    test_differing_arms_are_invalid()
    test_byte_identity_is_reported_only_when_recorded()
    test_stage_attribution_counts_each_part_once()
    test_claims()
    print("all bench_compare CPU tests passed")
