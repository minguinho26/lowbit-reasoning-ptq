#!/usr/bin/env bash
# 1주차 확인 작업과 calibration trace 생성을 연달아 돌린다. 한 단계가 실패해도 다음 단계로 넘어간다.
# 각 단계 로그는 /workspace/logs/queue_<시각>/ 에, 단계별 성공/실패와 소요 시간은 그 폴더의 status.tsv 에 남는다.
# AUTO_TERMINATE=1 이면 모든 단계가 끝난 뒤 pod을 terminate한다. 볼륨(/workspace) 밖의 데이터는 사라진다.
#
# 예: tmux new -s q
#     AUTO_TERMINATE=1 bash queue_week1.sh
set -uo pipefail

export WS=${WS:-/workspace}
export HF_HOME=${HF_HOME:-$WS/hf_cache}
cd "$(dirname "$0")"
PY=$WS/venv_serve/bin/python
QPY=$WS/venv_quant/bin/python
LOG=$WS/logs/queue_$(date +%Y%m%d_%H%M)
mkdir -p "$LOG"
STATUS=$LOG/status.tsv
STEP_TIMEOUT=${STEP_TIMEOUT:-6h}   # 한 단계가 멈춰서 GPU 비용만 나가는 것을 막기 위한 상한
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

# 1) R1-1.5B로 양자화 경로 전체 확인: calibration 2종, NVFP4/INT2 x RTN/GPTQ/AutoRound 11개
step calib_r1-1.5b      $PY src/run_matrix.py calib --model r1-qwen-1.5b
# 설정마다 따로 돌려서 하나가 실패해도 나머지는 진행되게 함
for run in $($PY src/run_matrix.py list --model r1-qwen-1.5b | awk '$1 ~ /__/ && $1 !~ /__bf16$/ {print $1}'); do
  step "quant_${run#r1-qwen-1.5b__}" $QPY src/quantize.py --run "$run"
done
step eval_r1-1.5b_nvfp4 $PY src/evaluate.py --run r1-qwen-1.5b__nvfp4__rtn__wikitext2 --bench aime120 --seed 0 --limit 30
step eval_r1-1.5b_w2    $PY src/evaluate.py --run r1-qwen-1.5b__w2a16__rtn --bench aime120 --seed 0 --limit 30

# 2) GPQA 출력 길이 (본 실행 결과로 그대로 쓰임)
step gpqa_r1-1.5b_bf16  $PY src/evaluate.py --run r1-qwen-1.5b__bf16 --bench gpqa_diamond --seed 0

# 3) Qwen3-8B NVFP4 처리량과 커널 (WikiText2 윈도우만 먼저 만듦)
step win_q8b_wiki       $QPY src/make_calib.py windows --model qwen3-8b --source wikitext2
step quant_q8b_nvfp4    $QPY src/quantize.py --run qwen3-8b__nvfp4__rtn__wikitext2
step eval_q8b_nvfp4     $PY src/evaluate.py --run qwen3-8b__nvfp4__rtn__wikitext2 --bench aime120 --seed 0 --limit 30

# 4) 나머지 모델 calibration (trace 생성이 대부분의 시간)
step calib_q1.7b        $PY src/run_matrix.py calib --model qwen3-1.7b
step calib_r1-7b        $PY src/run_matrix.py calib --model r1-qwen-7b
step calib_q8b          $PY src/run_matrix.py calib --model qwen3-8b

# 요약
{
  echo; echo "== status"; cat "$STATUS"
  echo; echo "== eval summaries"
  for f in "$LOG"/*eval*.log "$LOG"/*gpqa*.log; do
    [ -f "$f" ] && echo "-- $(basename "$f")" && grep -A9 '"run"' "$f" | grep -E '"(run|bench|acc|mean_tokens|truncation_rate|loop_rate|throughput_tok_s_this_run)"'
  done
  echo; echo "== kernels / KV"
  grep -h -i "NVFP4 GEMM\|for W2\|humming\|KV cache size" "$LOG"/*eval*.log 2>/dev/null | sed 's/^.*INFO //' | sort -u
} | tee "$LOG/summary.txt"

if [ "${AUTO_TERMINATE:-0}" = "1" ]; then
  echo "== terminate pod ${RUNPOD_POD_ID:-?}" | tee -a "$LOG/summary.txt"
  sync
  runpodctl pod delete "$RUNPOD_POD_ID" || runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"
fi
