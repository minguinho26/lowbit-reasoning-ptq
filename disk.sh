#!/usr/bin/env bash
# 볼륨 사용량과, 지금 설정에서 안 쓰는 파일 목록.
#   지울 후보: 설정 파일에 없는 모델의 원본/체크포인트(models/), calibration(calib/),
#             저장 도중 실패해서 quant_meta.json이 없는 체크포인트 폴더
#   평가 결과(results/)는 지우지 않는다.
# 큐가 돌고 있을 때는 clean 하지 말 것(저장 중인 체크포인트도 미완성으로 보고 지운다).
# 예: bash disk.sh          # 사용량과 지울 후보만 보여줌
#     bash disk.sh clean    # 위 후보를 실제로 지움
WS=${WS:-/workspace}
cd "$(dirname "$0")"

echo "== 사용량"
du -sh "$WS"/* 2>/dev/null | sort -h

KEEP=$("$WS/venv_serve/bin/python" -c "
import sys; sys.path.insert(0, 'src')
from common import load_config
cfg = load_config()
print(' '.join(dict.fromkeys(list(cfg['models']) + list(cfg.get('sanity', {})))))")
echo; echo "== 설정의 모델: $KEEP"

CAND=()
for d in "$WS"/models/*/; do
  d=${d%/}; name=$(basename "$d"); model=${name%%__*}
  if [[ " $KEEP " != *" $model "* ]]; then
    CAND+=("$d")
  elif [[ "$name" != *__bf16 && ! -f "$d/quant_meta.json" ]]; then
    CAND+=("$d")   # 저장 실패로 남은 미완성 체크포인트
  fi
done
for d in "$WS"/calib/*/; do
  d=${d%/}; model=$(basename "$d")
  [[ " $KEEP " != *" $model "* ]] && CAND+=("$d")
done

echo; echo "== 지울 후보"
[ ${#CAND[@]} -eq 0 ] && { echo "(없음)"; exit 0; }
du -sh "${CAND[@]}" | sort -h
du -sch "${CAND[@]}" | tail -1

if [ "${1:-}" = "clean" ]; then
  rm -rf "${CAND[@]}"
  echo; echo "== 지운 뒤"; du -sh "$WS"/models "$WS"/calib
fi
