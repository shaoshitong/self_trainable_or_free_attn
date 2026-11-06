#!/bin/bash
# Sync this experiment package to a configured Linux CUDA host and build in place.
# This wrapper contains no site-specific host, path, Python, or CUDA defaults.
#
# Required:
#   VSA2_REP_HOST     SSH host/alias
#   VSA2_REP_REMOTE   remote work root; package becomes $VSA2_REP_REMOTE/pkg
#   VSA2_REP_SOURCE   remote VSA2 source tree containing csrc/vsa
#   VSA2_REP_TK_ROOT  remote ThunderKittens checkout
#
# Optional:
#   VSA2_REP_PYTHON   default /opt/venv/bin/python
#   VSA2_REP_CUDA_HOME default /usr/local/cuda
#   VSA2_REP_ARCH     default 9.0a
#   VSA2_REP_JOBS     default 24
#
# Usage: ./scripts/sync_build.sh [sync|build|all]
set -euo pipefail

host=${VSA2_REP_HOST:?set VSA2_REP_HOST}
remote=${VSA2_REP_REMOTE:?set VSA2_REP_REMOTE}
source_tree=${VSA2_REP_SOURCE:?set VSA2_REP_SOURCE}
tk_root=${VSA2_REP_TK_ROOT:?set VSA2_REP_TK_ROOT}
python=${VSA2_REP_PYTHON:-/opt/venv/bin/python}
cuda_home=${VSA2_REP_CUDA_HOME:-/usr/local/cuda}
arch=${VSA2_REP_ARCH:-9.0a}
jobs=${VSA2_REP_JOBS:-24}
mode=${1:-all}
ssh_cmd=(ssh -o BatchMode=yes -o ConnectTimeout=20)

case "$mode" in
  sync|all)
    rsync -rlpz --checksum --no-times \
      --exclude '__pycache__' --exclude '*.pyc' --exclude 'build' --exclude '*.so' --exclude '*.egg-info' \
      -e "${ssh_cmd[*]}" ./ "$host:$remote/pkg/"
    ;;
esac

case "$mode" in
  build|all)
    "${ssh_cmd[@]}" "$host" \
      "mkdir -p '$remote/pkg' && cd '$remote/pkg' && rm -f build.log build.done && setsid nohup bash -c 'export PATH=$cuda_home/bin:\$PATH CUDA_HOME=$cuda_home VSA2_REP_BUILD=1 VSA2_REP_TK=1 VSA2_REP_TK_ROOT=$tk_root VSA2_REP_SOURCE=$source_tree TORCH_CUDA_ARCH_LIST=$arch MAX_JOBS=$jobs; $python setup.py build_ext --inplace > build.log 2>&1; echo \$? > build.done' >/dev/null 2>&1 </dev/null &"
    ;;
  sync|build|all) ;;
  *)
    echo "usage: $0 [sync|build|all]" >&2
    exit 2
    ;;
esac
