#!/usr/bin/env bash
set -u

PYTHON=/home/u25s065008/.conda/envs/pointworld-env/bin/python
ROOT=/home/u25s065008/datasets/libero
OUT=/home/u25s065008/datasets/libero_pointworld_all_suites
COMMON=(
  --output_layout pool
  --frame_step 2
  --window_stride 5
  --num_workers 2
  --camera_layout oblique_triplet
  --camera_names frontview sideview birdview
  --skip_existing
)

run_suite() {
  local suite="$1"
  local bounds=("${@:2}")
  echo "[$(date -Is)] START ${suite}"
  "$PYTHON" -m tools.libero.bulk_export \
    --libero_root "$ROOT/$suite" \
    --output_root "$OUT/$suite" \
    "${COMMON[@]}" \
    --workspace_bounds "${bounds[@]}" \
    2>&1 | tee -a "run_logs/libero_full_export_${suite}.log"
  local rc=${PIPESTATUS[0]}
  echo "[$(date -Is)] END ${suite} rc=${rc}"
  return "$rc"
}

run_suite libero_goal -0.8 -0.8 0.65 0.8 0.8 1.5 || exit $?
run_suite libero_object -0.8 -0.8 -0.05 0.8 0.8 0.5 || exit $?
run_suite libero_10 -0.8 -0.8 0.65 0.8 0.8 1.5 || exit $?

echo "[$(date -Is)] ALL SUITES COMPLETE"
