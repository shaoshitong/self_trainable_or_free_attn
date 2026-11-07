"""CPU tests of what benchmarks/vsa2_native_bench.py lets a result say. No torch, GPU or build.

Run: python3 tests/test_vsa2_native_bench_cpu.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))

import vsa2_native_bench as b


def err(max_abs=0.0, n_diff=0, rel_l2=0.0, n_nan=0):
    return dict(max_abs=max_abs, scale=1.0, rel_l2=rel_l2, n_diff=n_diff, n_bit_diff=n_diff, n_nan=n_nan, n_elements=100)


def child(o="o", sel="s", pad=0, fwd=1.0, identity="i"):
    return dict(digests=dict(o=o, selection=sel), padded_nonzero=dict(o=pad, dq=0, dk=0, dv=0, dgate=0), identity=identity,
                rows={r: [fwd, fwd, fwd] for r in b.ROWS})


def test_shape_matrix_and_partial():
    shapes = b.shape_matrix()
    assert len(shapes) == 5 * 2 * 2 + 1 and sum(s["partial"] for s in shapes) == 1
    assert b.shape_key(shapes[-1]) == "L16384/H16/D128/partial"
    counts = b.partial_blk_cnt(256)
    assert min(counts) >= 1 and max(counts) == 64 and sum(c < 64 for c in counts) > 10, "1..64, by the native contract"
    assert b.shape_error(dict(L=100, D=64)) and b.shape_error(dict(L=64, D=32)) and b.shape_error(dict(L=2048, D=64)) is None


def test_each_process_sees_only_its_arm():
    env = dict(PATH="x", ECHOSPARSE_VSA_FUSED_COLS="1", ECHOSPARSE_VSA_TK_BWD_OPTS="3", ECHOSPARSE_VSA_BUILD="1")
    assert b.child_env("baseline", env) == dict(PATH="x", ECHOSPARSE_VSA_BUILD="1"), "a stray toggle must not reach the default"
    assert b.child_env("tk_bwd_store", env)["ECHOSPARSE_VSA_TK_BWD_STORE"] == "1" and "ECHOSPARSE_VSA_FUSED_COLS" not in b.child_env("tk_bwd_store", env)
    assert b.ORDER == ("default", "arm", "arm", "default")


def test_an_arm_nothing_reads_is_refused():
    assert b.arm_error("baseline", ["nothing"]) is None
    assert b.arm_error("fused_cols", ["no toggle here"])
    assert b.arm_error("fused_cols", ['getenv("ECHOSPARSE_VSA_FUSED_COLS")']) is None


def test_ratio_and_aa_come_from_one_run():
    rows = b.time_rows([child(fwd=1.0), child(fwd=2.0), child(fwd=2.0), child(fwd=1.1)])
    t = rows["native.fwd"]
    assert abs(t["ratio"] - 2.0 / 1.05) < 1e-9 and abs(t["aa_displacement_same_run"] - 1.1) < 1e-9
    assert rows["b_estimate_ms"]["measured"] is False


def test_gradient_verdicts():
    aa = [err(0.01, 1000, 1e-5)]
    assert b.gradient_verdict([err(0.01, 1200, 1e-5)], aa, 1e-3) == "OK", "the noise floor is not a difference"
    assert b.gradient_verdict([err(0.2, 1200, 1e-5)], aa, 1e-3) == "ABOVE_AA"
    assert b.gradient_verdict([err(0.01, 9000, 1e-5)], aa, 1e-3) == "ABOVE_AA"
    assert b.gradient_verdict([err(0.2, 9000, 5e-3)], aa, 1e-3) == "OVER_TOL"
    assert b.gradient_verdict([err(float("nan"), 1, 1e-5)], aa, 1e-3) == "FAIL"
    assert b.gradient_verdict([err(0.0, 0, 0.0, n_nan=1)], aa, 1e-3) == "FAIL"
    assert b.gradient_verdict([err(0.0, 1, 1e-9)], [err()], 1e-3) == "ABOVE_AA", "any difference exceeds an exact A/A"


def test_gradients_are_never_held_to_bytes_and_forward_always_is():
    def run(children, e=err(0.01, 1000, 1e-5)):
        return b.equivalence(children, lambda n, i, j: e, 1e-3)
    same = [child(), child(), child(), child()]
    assert run(same)["verdict"] == "OK", "differing gradients alone must not fail"
    assert run([child(), child(o="x"), child(o="x"), child()])["verdict"] == "FAIL"
    assert run([child(), child(sel="x"), child(sel="x"), child()])["verdict"] == "FAIL"
    assert run([child(), child(pad=1), child(), child()])["verdict"] == "FAIL", "padding is where this breaks"
    assert run([child(), child(), child(o="x"), child()])["verdict"] == "FAIL"
    assert run([child(o="a"), child(o="a"), child(o="b"), child(o="b")])["verdict"] == "NONDETERMINISTIC_FORWARD"


def result(arm="baseline", build="x", shape="L2048/H8/D64"):
    identity = dict(build=build, hardware=1, software=1, shape=1, timing=1, inputs=1)
    settings = dict(warmup=10, iters=30, batches=3, seed=0, topk_frac=0.1, grad_tol=1e-3)
    return dict(schema=b.SCHEMA, arm=arm, settings=settings, shapes={shape: dict(status="ok", identity=identity)})


def test_compare_refuses_what_cannot_be_read_together():
    assert b.compare_error(result(), result("fused_cols")) is None
    assert "A/A" in b.compare_error(result("fused_cols"), result())
    assert "shapes differ" in b.compare_error(result(), result("fused_cols", shape="L4096/H8/D64"))
    assert "build" in b.compare_error(result(), result("fused_cols", build="y"))
    other = result("fused_cols")
    other["settings"]["seed"] = 1
    assert "seed" in b.compare_error(result(), other)
    assert "must be" in b.compare_error(dict(schema="other"), result())


def test_results_are_json_and_a_stored_verdict_is_reread_against_both_floors():
    eq = b.equivalence([child()] * 4, lambda n, i, j: err(0.01, 1000, 1e-5) if {i, j} != {0, 3} else err(), 1e-3)
    shape = dict(status="ok", equivalence=json.loads(json.dumps(eq)))
    aa = dict(status="ok", equivalence=json.loads(json.dumps(eq)))
    assert b.combined_verdict(shape, aa) == "OK"
    assert b.combined_verdict(shape, None) == "OK"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all vsa2_native_bench CPU tests passed")
