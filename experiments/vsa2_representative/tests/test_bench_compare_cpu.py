"""CPU tests of what bench_compare.py lets a result claim. No torch, GPU or build.

Run: python3 tests/test_bench_compare_cpu.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench_stages import attribute, compare_stages
from bench_compare import (FWD_CFGS, PROFILES, abba, active_fwd_cfg, arms_error, assemble, cfg_error, claims,
                           fwd_dispatch, parse_config, summarize)


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
    for cfg in ("0", "1", "2", "6", "8", "9", "10", "11", "13", "20"):
        assert rejected(f"REP_TK_FWD_CFG={cfg}", "controlled-softmax"), "only CFG 4/5/7 are this diagnostic"
    for cfg in "457":
        assert parse_config(f"REP_TK_FWD_CFG={cfg}", "controlled-softmax")
    for cfg in ("8", "9", "10", "11", "13", "20"):
        assert parse_config(f"REP_TK_FWD_CFG={cfg}", "representative")
    assert rejected("REP_TK_FWD_CFG=21", "representative"), "the kernel would silently run its default path"
    assert PROFILES["representative"]["baseline"].endswith("=1") and PROFILES["representative"]["candidate"] is None
    assert rejected("", "representative") and rejected("OTHER=1,REP_TK_FWD_CFG=1", "representative")
    assert rejected("REP_TK_FWD_CFG=1,REP_TK_BWDC_SPLITS=0", "representative")


def kernel(*lines):
    return "static void fwd_impl(int x) {\n  if (cfg == 4) launch(rep_tk::nat_copy_ker<D, 0>);\n" + \
        "".join(f"  else if {line}\n" for line in lines) + "  else launch(rep_tk::fwd_ker<D, 3, false, false, true>);\n}\n"


BULK = "(cfg == 20) launch(rep_tk::fwd_ker<D, 3>);"
DOUBLE = "(cfg == 11) launch(rep_tk::fwd_ker<D, 3, false, false, true, true>);"
ROUND5 = kernel(BULK, DOUBLE) + "int bwd(int fwd_cfg) { return fwd_cfg == 13 ? 2 : 1; }\n"


def test_standalone_scripts_see_what_runs():
    assert active_fwd_cfg({}, ROUND5)[:2] == ("1", "operator")
    assert active_fwd_cfg({"REP_TK_FWD_CFG": "7"}, ROUND5 + "cfg == 7")[1] == "timing_only"
    assert fwd_dispatch(ROUND5)["20"] == "rep_tk::fwd_ker<D, 3>"
    for value, source in (("1", kernel()), ("11", kernel(BULK)), ("6", ROUND5), ("21", ROUND5)):
        try:
            active_fwd_cfg({"REP_TK_FWD_CFG": value}, source)
        except SystemExit:
            continue
        raise AssertionError(f"CFG{value} accepted")


def test_arm_meanings_hold_in_the_source():
    assert cfg_error("10", ROUND5) is None, "an alias needs no dispatch line of its own"
    assert cfg_error("10", kernel(BULK, "(cfg == 10) launch(rep_tk::fwd_ker<D, 3, false, false, true>);")) is None
    assert cfg_error("10", kernel(BULK, "(cfg == 10) launch(rep_tk::fwd_ker<D, 3>);")), "an alias that differs"
    assert cfg_error("13", ROUND5) is None
    assert cfg_error("13", kernel(BULK, "(cfg == 13) launch(rep_tk::fwd_ker<D, 4>);")), "forward change under CFG13"
    assert cfg_error("20", ROUND5) is None
    same = "(cfg == 20) launch(rep_tk::fwd_ker<D, 3, false, false, true>);"
    assert cfg_error("20", kernel(same)), "a history arm that is the default"
    assert cfg_error("13", ROUND5.replace("static void fwd_impl", "void renamed")), "must fail closed"


def test_backward_arm_is_single_variable():
    one, thirteen = {"REP_TK_FWD_CFG": "1"}, {"REP_TK_FWD_CFG": "13"}
    assert arms_error({"baseline": one, "candidate": thirteen}) is None
    assert arms_error({"baseline": {"REP_TK_FWD_CFG": "11"}, "candidate": thirteen})
    assert arms_error({"baseline": one, "candidate": dict(thirteen, REP_TK_BWDC_SPLITS="4")})


def test_forward_change_under_backward_arm_is_invalid():
    configs = {"baseline": {"REP_TK_FWD_CFG": "1"}, "candidate": {"REP_TK_FWD_CFG": "13"}}
    a, b = child("baseline", 1), child("candidate", 1)
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
    test_backward_arm_is_single_variable()
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
