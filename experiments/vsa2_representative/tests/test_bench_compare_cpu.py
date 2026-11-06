"""CPU tests of what bench_compare.py lets a result claim. No torch, GPU or build.

Run: python3 tests/test_bench_compare_cpu.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench_compare import FWD_CFGS, PROFILES, abba, active_fwd_cfg, assemble, claims, implemented, parse_config, summarize


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
    for cfg in "012689":
        assert rejected(f"REP_TK_FWD_CFG={cfg}", "controlled-softmax"), "only CFG 4/5/7 are this diagnostic"
    for cfg in "457":
        assert parse_config(f"REP_TK_FWD_CFG={cfg}", "controlled-softmax")
    for cfg in "89":
        assert parse_config(f"REP_TK_FWD_CFG={cfg}", "representative")
    assert rejected("REP_TK_FWD_CFG=10", "representative"), "the kernel would silently run its default path"
    assert PROFILES["representative"]["baseline"].endswith("=6") and PROFILES["representative"]["candidate"].endswith("=1")
    assert rejected("", "representative") and rejected("OTHER=1,REP_TK_FWD_CFG=1", "representative")
    assert rejected("REP_TK_FWD_CFG=1,REP_TK_BWDC_SPLITS=0", "representative")


def test_standalone_scripts_see_what_runs():
    new = "if (cfg == 4) a; else if (cfg == 5) b; else if (cfg == 7) c; else if (cfg == 0) d; " \
          "else if (cfg == 2 && phases == 1) e; else if (cfg == 3 && phases == 1) f; else if (cfg == 6) g; " \
          "else if (cfg == 8) h; else if (cfg == 9) i;"
    assert all(implemented(v, new) for v in FWD_CFGS)
    old = "if (cfg == 4) a; else if (cfg == 5) b; else if (cfg == 0) d;"
    assert active_fwd_cfg({}, old)[:2] == ("1", "operator")
    assert active_fwd_cfg({"REP_TK_FWD_CFG": "6"}, new)[1] == "operator"
    assert active_fwd_cfg({"REP_TK_FWD_CFG": "7"}, new)[1] == "timing_only", "CFG7 must never print operator rows"
    for value, source in (("6", old), ("7", old), ("8", old), ("9", old), ("10", new)):  # would run the default kernel
        assert not (value in FWD_CFGS and implemented(value, source))
        try:
            active_fwd_cfg({"REP_TK_FWD_CFG": value}, source)
        except SystemExit:
            continue
        raise AssertionError(f"CFG{value} accepted")


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
    test_abba_is_balanced()
    test_estimate_is_labelled_and_stages_are_not_summed()
    test_row_is_a_control_only_if_it_is_in_every_arm()
    test_differing_arms_are_invalid()
    test_claims()
    print("all bench_compare CPU tests passed")
