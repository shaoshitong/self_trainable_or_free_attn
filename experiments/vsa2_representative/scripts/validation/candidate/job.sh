#!/usr/bin/env bash
# Portable worker. Resource policy, paths and compiler locations are supplied privately by the controller.
set -euo pipefail
: "${REP_TASK_ROOT:?}" "${REP_SOURCE_SHA:?}" "${REP_SOURCE_COMMIT:?}" "${REP_PYTHON:?}"
: "${REP_TK_ROOT:?}" "${CUDA_HOME:?}" "${CC:?}" "${CXX:?}" "${REP_COMPILER_LIBS:?}"
SOURCE="$REP_TASK_ROOT/source/$REP_SOURCE_SHA"
OUT="$REP_TASK_ROOT/out/$REP_SOURCE_SHA/job-$SLURM_JOB_ID"
mkdir -p "$OUT/work" "$OUT/tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATH="$(dirname "$CXX"):$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$REP_COMPILER_LIBS:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TMPDIR="$OUT/tmp" MAX_JOBS=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST=9.0a VSA2_REP_BUILD=1 VSA2_REP_TK=1 VSA2_REP_PTXAS_VERBOSE=1
export VSA2_REP_TK_ROOT="$REP_TK_ROOT" REP_TK_FWD_CFG=1
unset REP_TK_BWDC_SPLITS
STAGE=prepare
finish() {
 rc=$?; trap - EXIT
 printf '%s\n' "$rc" > "$OUT/exitcode.txt"
 python3 - "$OUT" "$STAGE" "$rc" <<'PY'
from pathlib import Path
import datetime,json,os,sys
p=Path(sys.argv[1]);p.joinpath('final.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=os.environ['SLURM_JOB_ID'],exit_code=int(sys.argv[3]),stage=sys.argv[2],source_commit=os.environ['REP_SOURCE_COMMIT'],source_sha256=os.environ['REP_SOURCE_SHA']),indent=2)+'\n')
PY
 printf 'END rc=%s stage=%s utc=%s\n' "$rc" "$STAGE" "$(date -u +%FT%TZ)"
 exit "$rc"
}
trap finish EXIT
stage() {
 STAGE="$1";shift
 printf 'BEGIN %s %s\n' "$STAGE" "$(date -u +%FT%TZ)"
 set +e; "$@" > "$OUT/$STAGE.log" 2>&1; rc=$?; set -e
 printf 'END %s rc=%s %s\n' "$STAGE" "$rc" "$(date -u +%FT%TZ)"
 return "$rc"
}
cp -R "$SOURCE/." "$OUT/work/"
chmod -R u+w "$OUT/work"
export VSA2_REP_SOURCE="$OUT/work"
PKG="$OUT/work/experiments/vsa2_representative"
export PYTHONPATH="$PKG"
cd "$PKG"
stage prepare "$REP_PYTHON" "$OUT/work/_control/probe.py" prepare "$OUT"
ALLOCATED_UUID="$(python3 -S - "$OUT/hardware.json" <<'PYCODE'
from pathlib import Path
import json,sys
u=json.loads(Path(sys.argv[1]).read_text())['uuid'];print(u if u.startswith('GPU-') else 'GPU-'+u)
PYCODE
)"
export CUDA_VISIBLE_DEVICES="$ALLOCATED_UUID"
stage cpu env CUDA_VISIBLE_DEVICES="" "$REP_PYTHON" tests/test_reference_cpu.py
stage cpu_check python3 -S "$OUT/work/_control/check_suites.py" cpu "$OUT"
stage helper_cpu env CUDA_VISIBLE_DEVICES="" "$REP_PYTHON" tests/test_bench_compare_cpu.py
stage helper_cpu_check python3 -S "$OUT/work/_control/check_suites.py" helper_cpu "$OUT"
stage build "$REP_PYTHON" setup.py build_ext --inplace
stage import "$REP_PYTHON" "$OUT/work/_control/probe.py" import "$OUT"
stage generic "$REP_PYTHON" tests/test_gpu.py
stage generic_check python3 -S "$OUT/work/_control/check_suites.py" generic "$OUT"
stage tk "$REP_PYTHON" tests/test_gpu_tk.py
stage tk_check python3 -S "$OUT/work/_control/check_suites.py" tk "$OUT"
stage tk_hand env REP_TK_FWD_CFG=6 "$REP_PYTHON" tests/test_gpu_tk.py
stage tk_hand_check python3 -S "$OUT/work/_control/check_suites.py" tk_hand "$OUT"
BENCH=bench_compare.py
COMMON=(--len 16384 --heads 16 --dim 128 --mid-reps 4 --seed 0 --rounds 2 --batches 3)
stage aa "$REP_PYTHON" "$BENCH" "${COMMON[@]}" --baseline REP_TK_FWD_CFG=1 --candidate REP_TK_FWD_CFG=1 --out "$OUT/aa.json"
stage aa_check python3 -S "$OUT/work/_control/check_compare.py" representative "$OUT/aa.json" "$OUT/aa_check.json"
stage main "$REP_PYTHON" "$BENCH" "${COMMON[@]}" --baseline REP_TK_FWD_CFG=6 --candidate REP_TK_FWD_CFG=1 --out "$OUT/main.json"
stage main_check python3 -S "$OUT/work/_control/check_compare.py" representative "$OUT/main.json" "$OUT/main_check.json"
stage second_decision python3 -S "$OUT/work/_control/check_compare.py" decide "$OUT/aa.json" "$OUT/main.json"
if python3 -S -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))["run_second_length"] else 1)' "$OUT/second_length_decision.json"; then
 stage main32 "$REP_PYTHON" "$BENCH" --len 32768 --heads 16 --dim 128 --mid-reps 4 --seed 0 --rounds 2 --batches 3 --baseline REP_TK_FWD_CFG=6 --candidate REP_TK_FWD_CFG=1 --out "$OUT/main32.json"
 stage main32_check python3 -S "$OUT/work/_control/check_compare.py" representative "$OUT/main32.json" "$OUT/main32_check.json"
fi
stage diag45 "$REP_PYTHON" "$BENCH" "${COMMON[@]}" --profile controlled-softmax --baseline REP_TK_FWD_CFG=4 --candidate REP_TK_FWD_CFG=5 --out "$OUT/diag45.json"
stage diag45_check python3 -S "$OUT/work/_control/check_compare.py" controlled-softmax "$OUT/diag45.json" "$OUT/diag45_check.json"
stage diag47 "$REP_PYTHON" "$BENCH" "${COMMON[@]}" --profile controlled-softmax --baseline REP_TK_FWD_CFG=4 --candidate REP_TK_FWD_CFG=7 --out "$OUT/diag47.json"
stage diag47_check python3 -S "$OUT/work/_control/check_compare.py" controlled-softmax "$OUT/diag47.json" "$OUT/diag47_check.json"
STAGE=finished
