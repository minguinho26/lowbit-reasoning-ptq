#!/usr/bin/env bash
# 2주차 시작분: 새 구성(Qwen3 1.7B/4B/8B, NVFP4 W4A16 추가)의 확인 작업 후 본 실행 일부를 연달아 돌린다.
# 한 단계가 실패해도 다음 단계로 넘어간다. 로그와 status.tsv, summary.txt는 /workspace/logs/queue_<시각>/ 에 남는다.
# AUTO_TERMINATE=1 이면 모든 단계가 끝난 뒤 pod을 terminate한다.
#
# 예: tmux new -s q
#     AUTO_TERMINATE=1 bash queue_week2.sh
set -uo pipefail

export WS=${WS:-/workspace}
export HF_HOME=${HF_HOME:-$WS/hf_cache}
cd "$(dirname "$0")"
PY=$WS/venv_serve/bin/python
QPY=$WS/venv_quant/bin/python
LOG=$WS/logs/queue_$(date +%Y%m%d_%H%M)
mkdir -p "$LOG"
STATUS=$LOG/status.tsv
STEP_TIMEOUT=${STEP_TIMEOUT:-6h}
[ -n "${HF_TOKEN:-}" ] || echo "경고: HF_TOKEN 없음. GPQA 단계가 실패한다" | tee -a "$LOG/warn.txt"

n=0
step() {  # step <이름> <명령...>
  n=$((n + 1))
  local name=$(printf "%02d_%s" "$n" "$1"); shift
  local t0=$(date +%s)
  echo "== [$(date '+%F %T')] $name: $*"
  timeout "$STEP_TIMEOUT" "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  local mins=$(( ($(date +%s) - t0) / 60 ))
  printf "%s\t%s\t%dmin\n" "$name" "$([ $rc -eq 0 ] && echo ok || echo "FAIL($rc)")" "$mins" | tee -a "$STATUS"
}

# 1) Qwen3-4B 원본: 32K 안에 들어오는지, 처리량
step fetch_q4b          $PY src/fetch_models.py --model qwen3-4b
step eval_q4b_bf16      $PY src/evaluate.py --run qwen3-4b__bf16 --bench aime120 --seed 0 --limit 30

# 2) NVFP4 W4A16 커널과 양자화 경로 (데이터 없는 RTN이라 바로 됨)
step quant_q1.7b_a16    $QPY src/quantize.py --run qwen3-1.7b__nvfp4a16__rtn
step eval_q1.7b_a16     $PY src/evaluate.py --run qwen3-1.7b__nvfp4a16__rtn --bench aime120 --seed 0 --limit 30

# 3) Qwen3-8B INT2 AutoRound: 8B에서 AutoRound가 메모리 안에서 끝나는지, 실제 출력 길이(비용 추정의 최대 불확실성)
step calib_q8b          $PY src/run_matrix.py calib --model qwen3-8b
step quant_q8b_w2_ar    $QPY src/quantize.py --run qwen3-8b__w2a16__autoround__trace
step eval_q8b_w2_ar     $PY src/evaluate.py --run qwen3-8b__w2a16__autoround__trace --bench aime120 --seed 0 --limit 30

# 4) Qwen3-4B calibration (trace 생성)
step calib_q4b          $PY src/run_matrix.py calib --model qwen3-4b

# 5) Qwen3-1.7B 본 실행: calibration, 나머지 양자화 전부, 원본 전체 평가(AIME, GPQA 각 3 seed)
step calib_q1.7b        $PY src/run_matrix.py calib --model qwen3-1.7b
for run in $($PY src/run_matrix.py list --model qwen3-1.7b | awk '$1 ~ /__/ && $1 !~ /__bf16$/ && $2 == "quant=-" {print $1}'); do
  step "quant_${run#qwen3-1.7b__}" $QPY src/quantize.py --run "$run"
done
step eval_q1.7b_bf16    $PY src/run_matrix.py evaluate --model qwen3-1.7b --fmt bf16

# 요약
{
  echo; echo "== status"; cat "$STATUS"
  echo; echo "== eval summaries"
  for f in "$LOG"/*eval*.log; do
    [ -f "$f" ] && echo "-- $(basename "$f")" && grep -A9 '"run"' "$f" | grep -E '"(run|bench|seed|acc|mean_tokens|truncation_rate|loop_rate|throughput_tok_s_this_run)"'
  done
  echo; echo "== kernels / KV"
  for f in "$LOG"/*eval*.log; do
    [ -f "$f" ] && echo "-- $(basename "$f")" && grep -h -i "Using .*Kernel\|KV cache size" "$f" | sed 's/^.*INFO //' | sort -u
  done
  echo; echo "== quant times"
  for f in "$LOG"/*quant*.log; do
    [ -f "$f" ] && echo "-- $(basename "$f")" && grep -h '"quantize_seconds"\|"quantized_param_fraction"\|Error\|OutOfMemory' "$f" | head -3
  done
} | tee "$LOG/summary.txt"

if [ "${AUTO_TERMINATE:-0}" = "1" ]; then
  echo "== terminate pod ${RUNPOD_POD_ID:-?}" | tee -a "$LOG/summary.txt"
  sync
  runpodctl pod delete "$RUNPOD_POD_ID" || runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"
fi
