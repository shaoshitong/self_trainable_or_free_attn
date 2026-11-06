#!/bin/bash
# Wait for scripts/sync_build.sh's detached build, then run a command remotely.
# Usage: ./scripts/wait_run.sh "<command run inside the remote package directory>"
set -euo pipefail

host=${VSA2_REP_HOST:?set VSA2_REP_HOST}
remote=${VSA2_REP_REMOTE:?set VSA2_REP_REMOTE}
remote_pkg="$remote/pkg"
ssh_cmd=(ssh -o BatchMode=yes -o ConnectTimeout=20)

for _ in $(seq 1 60); do
  state=$("${ssh_cmd[@]}" "$host" "cat '$remote_pkg/build.done' 2>/dev/null || echo RUNNING" 2>/dev/null | tail -1)
  [[ "$state" != RUNNING ]] && break
  sleep 15
done

if [[ "$state" != 0 ]]; then
  echo "build failed (exit=$state)" >&2
  "${ssh_cmd[@]}" "$host" "grep -E 'error' '$remote_pkg/build.log' | head -30" >&2 || true
  exit 1
fi

echo "build ok"
"${ssh_cmd[@]}" "$host" "cd '$remote_pkg' && $1" 2>&1 | grep -v '^Warning' || true
