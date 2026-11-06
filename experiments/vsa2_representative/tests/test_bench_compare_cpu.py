"""CPU tests of what bench_compare.py lets a result claim. No torch, GPU or build.

Run: python3 tests/test_bench_compare_cpu.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench_compare import abba, assemble, claims, parse_config, summarize


def child(arm, scale, check=True):
    meta = dict(scope="fine_op", implementation="tk", work="64/4/1", control=False, measured=True)
    return dict(arm=arm, identity=dict(build=dict(extension_sha256="x"), inputs=dict(q="y")),
                output_check=[dict(name="o", ok=check)],
                rows=[dict(id="op.f", ms=[1.0 * scale] * 3, meta=dict(meta, stage="f", complete_call=True)),
                      dict(id="op.f_plus_b", ms=[3.0 * scale] * 3, meta=dict(meta, stage="f_plus_b", complete_call=True)),
                      dict(id="k.a", ms=[0.5] * 3, meta=dict(meta, scope="kernel", stage="bwd_stage", complete_call=False)),
                      dict(id="k.b", ms=[0.5] * 3, meta=dict(meta, scope="kernel", stage="bwd_stage", complete_call=False))])


def rejected(text, profile, new=()):
    try:
        parse_config(text, profile, new)
    except ValueError:
        return True
    return False


def test_configs_keep_their_semantics():
    assert parse_config("REP_TK_FWD_CFG=1,REP_TK_BWDC_SPLITS=8", "representative") == \
        {"REP_TK_FWD_CFG": "1", "REP_TK_BWDC_SPLITS": "8"}
    for cfg in "2345":
        assert rejected(f"REP_TK_FWD_CFG={cfg}", "representative"), "a diagnostic passed as production"
    assert rejected("REP_TK_FWD_CFG=1", "controlled-softmax")
    assert rejected("REP_TK_FWD_CFG=6", "representative"), "the kernel would silently run its default path"
    assert parse_config("REP_TK_FWD_CFG=6", "representative", ["6"])["REP_TK_FWD_CFG"] == "6"
    assert rejected("", "representative") and rejected("OTHER=1,REP_TK_FWD_CFG=1", "representative")
    assert rejected("REP_TK_FWD_CFG=1,REP_TK_BWDC_SPLITS=0", "representative")


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
    assert c["output_validated"] is False and c["may_claim_op_level_change"] is False and c["limits"]


if __name__ == "__main__":
    test_configs_keep_their_semantics()
    test_abba_is_balanced()
    test_estimate_is_labelled_and_stages_are_not_summed()
    test_differing_arms_are_invalid()
    test_claims()
    print("all bench_compare CPU tests passed")
