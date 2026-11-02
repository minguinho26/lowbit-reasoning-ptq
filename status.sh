#!/usr/bin/env bash
# 진행 상황 요약. 끝난 평가의 seed 평균(정확도, 평균 출력 토큰, truncation), 실패 수, 지금 도는 단위.
# 예: bash status.sh
WS=${WS:-/workspace}
"$WS/venv_serve/bin/python" - "$WS" <<'EOF'
import collections, glob, json, sys
ws = sys.argv[1]
d = collections.defaultdict(list)
for f in sorted(glob.glob(f"{ws}/results/eval/qwen3-*/*.done.json")):
    j = json.load(open(f))
    d[(j["run"], j["bench"])].append((j["acc"], j["mean_tokens"], j["truncation_rate"]))
for (r, b), v in d.items():
    n = len(v)
    print(f"{r:45s} {b[:4]} n={n} acc={100*sum(x[0] for x in v)/n:5.1f} tok={sum(x[1] for x in v)/n:6.0f} trunc={100*sum(x[2] for x in v)/n:4.1f}")
EOF
LOG=$(ls -td "$WS"/logs/main_* 2>/dev/null | head -1)
[ -n "$LOG" ] || exit 0
echo
echo "fail: $(grep -c FAIL "$LOG/status.tsv" 2>/dev/null || echo 0)  ok: $(grep -c $'\tok' "$LOG/status.tsv" 2>/dev/null || echo 0)"
grep FAIL "$LOG/status.tsv" 2>/dev/null
tail -2 "$LOG/queue.out"
