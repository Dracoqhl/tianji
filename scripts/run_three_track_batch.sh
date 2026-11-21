#!/usr/bin/env bash

set -eo pipefail
set +u
if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != '--b-only' ]; }; then
  echo '用法: run_three_track_batch.sh [--b-only]' >&2
  exit 2
fi
cd /workspace/qhl
python=/workspace/ctt/model_augmentation/final_check/.venv/bin/python
out=/workspace/qhl/runs/three_track_gpu_batch_v1
hik=/workspace/qhl/runs/t1_champion_st_tree_gpu_v1
test -x "$python"
mkdir -p "$out/logs"
exec 8>"$out/.batch.lock"
if ! flock -n 8; then
  echo '同一目录已有一批实验在运行' >&2
  exit 1
fi
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

run_stage() {
  local name=${1:?missing stage}
  shift
  local rc=0
  printf '%s\t%s\tstart\n' "$(date -u +%FT%TZ)" "$name" >> "$out/batch_status.tsv"
  "$python" -u "$@" 2>&1 | tee -a "$out/logs/$name.log" || rc=$?
  printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$name" "$rc" >> "$out/batch_status.tsv"
  return "$rc"
}

failed=0
if [ "${1:-}" = '--b-only' ]; then
  echo '[B] 复用已完成的 A，自动准入、预测和计分'
else
  echo '[1/5] T2 日电量四个窗口'
  if [ ! -s "$out/T2/results.json" ]; then
    run_stage T2 "$script_dir/batch_energy_short.py" T2 --out "$out/T2" --hik-run "$hik" || failed=1
  fi
  echo '[2/5] T3 短历史三种训练方式'
  if [ ! -s "$out/T3/results.json" ]; then
    run_stage T3 "$script_dir/batch_energy_short.py" T3 --out "$out/T3" || failed=1
  fi
  echo '[3–4/5] T1 逐站筛选与 A 集复核'
  if [ ! -s "$out/T1/a_station_routes.json" ]; then
    t1=("$script_dir/screen_companion_t1_stations.py")
    settings=(--candidate-set spatial --baseline-run "$hik" --out "$out/T1")
    if run_stage T1_screen "${t1[@]}" screen "${settings[@]}"; then
      run_stage T1_A "${t1[@]}" confirm-a "${settings[@]}" || failed=1
    else
      failed=1
      printf '%s\tT1_A\tskipped_after_T1_screen\n' "$(date -u +%FT%TZ)" >> "$out/batch_status.tsv"
    fi
  fi
  echo '[5/5] B 自动准入、预测和计分'
fi
run_stage B_auto "$script_dir/advance_batch_b.py" --batch "$out" || failed=1

echo "[三赛道结果] $out"
"$python" -c '
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
metric = "readme_unclipped_score"
for track, window in (("T2", "A"), ("T3", "fixed_50_10")):
  result = root / track / "results.json"
  if result.is_file():
      scores = json.loads(result.read_text())["windows"][window]["scores"]
      print(track, window, {name: round(row[metric], 6) for name, row in scores.items()})
t1 = root / "T1/a_station_routes.json"
if t1.is_file():
  report = json.loads(t1.read_text())
  print("T1 A", {arm: round(row[metric], 6) for arm, row in report["scores"].items()},
        "routes=", report["routes"])
' "$out"
echo "阶段状态 $out/batch_status.tsv"
exit "$failed"
