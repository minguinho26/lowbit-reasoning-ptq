#!/usr/bin/env bash
# 본 실행. pod의 GPU 전부에 워커를 하나씩 두고 run_matrix.py queue로 작업 단위를 나눠 돌린 뒤,
# 모델 하나의 평가가 끝나면 GPU 하나로 INT2를 검증하고 속도를 잰 뒤 다음 모델로 간다.
#   작업 단위: 원본 다운로드, calibration, 양자화 1개, 평가 1개(실행 단위 x 벤치 x seed).
#   이미 끝난 단위는 건너뛰므로, 끊기면 같은 명령으로 다시 돌리면 이어서 한다.
#   첫 모델의 원본 평가(공개 수치 재현 확인)가 맨 앞이다.
# WAIT_FOR_IDLE=1 이면 기존 GPU 작업을 기다려 GPU가 비는 순서대로 시작한다(큐 교체용).
# 로그: /workspace/logs/main_<시각>/ (단위별 로그, status.tsv, summary.txt)
# AUTO_TERMINATE=1 이면 모든 단계가 끝난 뒤 pod을 terminate한다.
#
# 예: tmux new -s q
#     AUTO_TERMINATE=1 bash queue_main.sh                       # 기본 모델 순서 qwen3-8b,qwen3-1.7b
#     GPUS=0 bash queue_main.sh qwen3-8b                         # GPU와 모델을 지정
set -uo pipefail

MODELS=${1:-qwen3-8b,qwen3-1.7b}
export WS=${WS:-/workspace}
export HF_HOME=${HF_HOME:-$WS/hf_cache}
cd "$(dirname "$0")"
PY=$WS/venv_serve/bin/python
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}
TIMEOUT=${TIMEOUT:-43200}   # INT2 반복 출력도 완료할 수 있도록 단위 하나 최대 12시간
LOG=$WS/logs/main_$(date +%Y%m%d_%H%M)
mkdir -p "$LOG"
[ -n "${HF_TOKEN:-}" ] || echo "경고: HF_TOKEN 없음. GPQA 평가가 실패한다" | tee -a "$LOG/warn.txt"
echo "models=$MODELS gpus=$GPUS log=$LOG"

# 큐는 한 번에 하나만. 두 개가 돌면 같은 GPU에 엔진이 겹쳐 OOM이 난다
exec 9>"${QUEUE_LOCK:-/tmp/lowbit_queue.lock}"   # 테스트는 별도 lock을 사용한다. 실제 큐는 하나만 실행.
flock -n 9 || { echo "다른 queue_main.sh가 이미 돌고 있음. ps aux | grep run_matrix 로 확인할 것"; exit 1; }

# 이전 실행에서 남은 프로세스(특히 vLLM EngineCore)가 GPU 메모리를 잡고 있으면 새 엔진이 OOM으로 못 뜬다
LEFT=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
if [ -n "$LEFT" ] && [ "${WAIT_FOR_IDLE:-0}" != "1" ]; then
  echo "GPU에 이미 프로세스가 있음. 정리하고 다시 실행할 것:"; echo "$LEFT"
  echo '  pkill -9 -f "evaluate.py|quantize.py|make_calib.py|bench_speed.py|run_matrix.py|EngineCore"; sleep 5; nvidia-smi'
  exit 1
fi

# 모델별 평가 -> INT2 진단 -> 속도를 끝낸 뒤 다음 모델로 진행한다.
WAIT_ARGS=()
[ "${WAIT_FOR_IDLE:-0}" = "1" ] && WAIT_ARGS+=(--wait-for-idle)
QRC=0
for m in ${MODELS//,/ }; do
  $PY src/run_matrix.py queue "${WAIT_ARGS[@]}" --gpus "$GPUS" --models "$m" --log-dir "$LOG" --timeout "$TIMEOUT" 2>&1 | tee -a "$LOG/queue.out"
  QRC=${PIPESTATUS[0]}
  # 다음 단계는 현재 모델의 모든 GPU 작업이 종료된 후에만 실행한다.
  [ "$QRC" -eq 0 ] || break
  CUDA_VISIBLE_DEVICES=${GPUS%%,*} timeout "$TIMEOUT" $PY src/run_matrix.py diagnose --model "$m" > "$LOG/diagnose__$m.log" 2>&1
  printf "diagnose:%s\t%s\n" "$m" "$([ $? -eq 0 ] && echo ok || echo FAIL)" >> "$LOG/status.tsv"
  CUDA_VISIBLE_DEVICES=${GPUS%%,*} timeout "$TIMEOUT" $PY src/run_matrix.py speed --force --model "$m" > "$LOG/speed__$m.log" 2>&1
  printf "speed:%s\t%s\n" "$m" "$([ $? -eq 0 ] && echo ok || echo FAIL)" >> "$LOG/status.tsv"
  $PY src/analyze.py --update-readme > "$LOG/analysis__$m.log" 2>&1
  printf "analysis:%s\t%s\n" "$m" "$([ $? -eq 0 ] && echo ok || echo FAIL)" >> "$LOG/status.tsv"
done
# 큐가 정상 종료하지 않았으면(사람이 끊었거나 죽었으면) 속도 측정과 자동 종료로 넘어가지 않는다
# rc=2: 워커가 연속 실패로 멈춤. 남은 작업을 못 하는 상태라 속도 측정은 건너뛰고, 비용이 계속 나가지 않게 자동 종료는 한다
if [ "$QRC" -ne 0 ] && [ "$QRC" -ne 2 ]; then
  echo "run_matrix queue가 비정상 종료(rc=$QRC). 속도 측정과 자동 종료를 건너뜀" | tee -a "$LOG/warn.txt"
  exit "$QRC"
fi
[ "$QRC" -eq 2 ] && echo "워커가 연속 실패로 멈춤. 속도 측정을 건너뜀. 원인은 status.tsv의 FAIL 단위 로그 참고" | tee -a "$LOG/warn.txt"

# 요약
{
  echo "== status: $(grep -c $'\tok' "$LOG/status.tsv") ok, $(grep -c FAIL "$LOG/status.tsv") fail"
  grep FAIL "$LOG/status.tsv"
  echo; tail -n +$(grep -n "queue finished" "$LOG/queue.out" | tail -n 1 | cut -d: -f1) "$LOG/queue.out" 2>/dev/null
  echo; echo "== eval summaries"
  for f in "$LOG"/eval__*.log; do
    [ -f "$f" ] || continue
    grep -A9 '"run"' "$f" | tr -d '\n ' | grep -oE '"run":"[^"]*"|"bench":"[^"]*"|"seed":[0-9]+|"n":[0-9]+|"acc":[0-9.]+|"mean_tokens":[0-9.]+|"truncation_rate":[0-9.]+|"loop_rate":[0-9.]+' | paste -sd' '
  done
  echo; echo "== kernels / KV (seed 0 AIME 로그 기준)"
  for f in "$LOG"/eval__*__aime120__s0.log; do
    [ -f "$f" ] && echo "-- $(basename "$f")" && grep -h -i "Using .*Kernel\|KV cache size" "$f" | sed 's/^.*INFO //' | sort -u
  done
  echo; echo "== quant"
  for f in "$LOG"/quant__*.log; do
    [ -f "$f" ] && echo "-- $(basename "$f")" && grep -h '"quantize_seconds"\|"quantized_param_fraction"\|Error\|OutOfMemory' "$f" | head -3
  done
} > "$LOG/summary.txt"
head -40 "$LOG/summary.txt"

if [ "${AUTO_TERMINATE:-0}" = "1" ]; then
  echo "== terminate pod ${RUNPOD_POD_ID:-?}" | tee -a "$LOG/summary.txt"
  sync
  runpodctl pod delete "$RUNPOD_POD_ID" || runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"
fi
