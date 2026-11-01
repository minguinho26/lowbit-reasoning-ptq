#!/usr/bin/env bash
# 본 실행. 모델 하나를 받아 calibration, 양자화, 평가, 속도 측정을 단위별로 하나씩 돌린다.
#   단위: 양자화는 실행 단위 1개, 평가는 (실행 단위, 벤치, seed) 1개. 단계 하나가 실패해도 다음 단계로 넘어간다.
#   이미 끝난 단위(quant_meta.json, .done.json이 있는 것)는 건너뛰므로, 끊기면 같은 명령으로 다시 돌리면 이어서 한다.
# 순서: 원본 평가(공개 수치 재현 확인)를 맨 앞에 두고, 그다음 실행 단위마다 양자화 -> 평가. 속도 측정은 마지막에 한 번에.
# 로그와 status.tsv, summary.txt는 /workspace/logs/main_<모델>_<시각>/ 에 남는다.
# AUTO_TERMINATE=1 이면 모든 단계가 끝난 뒤 pod을 terminate한다.
#
# 예: tmux new -s q
#     AUTO_TERMINATE=1 bash queue_main.sh qwen3-8b
set -uo pipefail

MODEL=${1:?모델 이름이 필요함. 예: bash queue_main.sh qwen3-8b}
export WS=${WS:-/workspace}
export HF_HOME=${HF_HOME:-$WS/hf_cache}
cd "$(dirname "$0")"
PY=$WS/venv_serve/bin/python
QPY=$WS/venv_quant/bin/python
LOG=$WS/logs/main_${MODEL}_$(date +%Y%m%d_%H%M)
mkdir -p "$LOG"
STATUS=$LOG/status.tsv
STEP_TIMEOUT=${STEP_TIMEOUT:-6h}
[ -n "${HF_TOKEN:-}" ] || echo "경고: HF_TOKEN 없음. GPQA 단계가 실패한다" | tee -a "$LOG/warn.txt"

n=0
step() {  # step <이름> <명령...>
  n=$((n + 1))
  local name=$(printf "%03d_%s" "$n" "$1"); shift
  local t0=$(date +%s)
  echo "== [$(date '+%F %T')] $name: $*"
  timeout "$STEP_TIMEOUT" "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  local mins=$(( ($(date +%s) - t0) / 60 ))
  printf "%s\t%s\t%dmin\n" "$name" "$([ $rc -eq 0 ] && echo ok || echo "FAIL($rc)")" "$mins" | tee -a "$STATUS"
}

# 벤치와 seed 목록(설정 파일 기준), 이 모델의 실행 단위 목록
BENCH_SEEDS=$($PY -c "
import sys; sys.path.insert(0, 'src')
from common import load_config
cfg = load_config()
for b, bc in cfg['benchmarks'].items():
    for s in bc['seeds']:
        print(f'{b}:{s}')")
# 원본 다음 순서: 공개 수치와 비교할 검증용 W4A16, NVFP4 W4A4 RTN을 먼저 두고 나머지는 설정 파일 순서
RUNS=$($PY -c "
import sys; sys.path.insert(0, 'src')
from common import load_config, enumerate_runs
runs = [r['name'] for r in enumerate_runs(load_config()) if r['model'] == '$MODEL' and r['fmt'] != 'bf16']
first = [r for r in runs if '__w4a16__' in r] + [r for r in runs if '__nvfp4__rtn__' in r]
print('\n'.join(first + [r for r in runs if r not in first]))")
[ -n "$RUNS" ] || { echo "설정에 없는 모델: $MODEL"; exit 1; }

evaluate_all() {  # evaluate_all <실행 단위>
  local run=$1 bs b s
  for bs in $BENCH_SEEDS; do
    b=${bs%%:*}; s=${bs##*:}
    [ -f "$WS/results/eval/$run/${b}__s${s}.done.json" ] && continue
    step "eval_${run#${MODEL}__}_${b}_s${s}" $PY src/evaluate.py --run "$run" --bench "$b" --seed "$s"
  done
}

# 1) 원본과 calibration
[ -f "$WS/models/${MODEL}__bf16/config.json" ] || step fetch $PY src/fetch_models.py --model "$MODEL"
step calib $PY src/run_matrix.py calib --model "$MODEL"
evaluate_all "${MODEL}__bf16"

# 2) 실행 단위마다 양자화 -> 평가
for run in $RUNS; do
  [ -f "$WS/models/$run/quant_meta.json" ] || step "quant_${run#${MODEL}__}" $QPY src/quantize.py --run "$run"
  [ -f "$WS/models/$run/quant_meta.json" ] && evaluate_all "$run"
done

# 3) 속도: 원본과 양자화 설정을 같은 pod에서 연달아
step speed $PY src/run_matrix.py speed --model "$MODEL"

# 요약
{
  echo; echo "== status"; cat "$STATUS"
  echo; echo "== eval summaries"
  for f in "$LOG"/*_eval_*.log; do
    [ -f "$f" ] || continue
    grep -A9 '"run"' "$f" | tr -d '\n ' | sed 's/^{//' | grep -oE '"run":"[^"]*"|"bench":"[^"]*"|"seed":[0-9]+|"n":[0-9]+|"acc":[0-9.]+|"mean_tokens":[0-9.]+|"truncation_rate":[0-9.]+|"loop_rate":[0-9.]+' | paste -sd' '
  done
  echo; echo "== kernels / KV (첫 평가 로그 기준, 실행 단위별)"
  for f in "$LOG"/*_eval_*_s0.log; do
    [ -f "$f" ] && echo "-- $(basename "$f")" && grep -h -i "Using .*Kernel\|KV cache size" "$f" | sed 's/^.*INFO //' | sort -u
  done
  echo; echo "== quant"
  for f in "$LOG"/*_quant_*.log; do
    [ -f "$f" ] && echo "-- $(basename "$f")" && grep -h '"quantize_seconds"\|"quantized_param_fraction"\|Error\|OutOfMemory' "$f" | head -3
  done
} | tee "$LOG/summary.txt"

if [ "${AUTO_TERMINATE:-0}" = "1" ]; then
  echo "== terminate pod ${RUNPOD_POD_ID:-?}" | tee -a "$LOG/summary.txt"
  sync
  runpodctl pod delete "$RUNPOD_POD_ID" || runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"
fi
