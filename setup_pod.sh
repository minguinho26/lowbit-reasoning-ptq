#!/usr/bin/env bash
# RunPod pod 초기 세팅. Network Volume이 /workspace에 마운트돼 있다고 가정.
# venv와 HF cache를 볼륨에 두므로 pod을 새로 띄워도 다시 설치하거나 다운로드하지 않는다.
set -euo pipefail

WS=${WS:-/workspace}
HERE=$(cd "$(dirname "$0")" && pwd)
export HF_HOME=$WS/hf_cache
export UV_CACHE_DIR=$WS/uv_cache
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$WS/models" "$WS/calib" "$WS/results"

echo "== GPU"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$DRIVER_MAJOR" -lt 580 ]; then
  echo "경고: 드라이버 $DRIVER_MAJOR. vllm 0.32.0(CUDA 13 빌드)은 580 이상이 필요할 수 있음. 다른 호스트로 pod을 다시 띄울 것"
fi

command -v uv >/dev/null || pip install -q uv

[ -x "$WS/venv_quant/bin/python" ] || uv venv "$WS/venv_quant" --python 3.12
[ -x "$WS/venv_serve/bin/python" ] || uv venv "$WS/venv_serve" --python 3.12
uv pip install --python "$WS/venv_quant/bin/python" -r "$HERE/requirements-quant.txt"
uv pip install --python "$WS/venv_serve/bin/python" -r "$HERE/requirements-serve.txt"

echo "== versions"
"$WS/venv_quant/bin/python" -c "import torch, transformers, llmcompressor; print('quant', torch.__version__, torch.version.cuda, transformers.__version__, llmcompressor.__version__, torch.cuda.get_device_capability(0))"
"$WS/venv_serve/bin/python" -c "import torch, transformers, vllm; print('serve', torch.__version__, torch.version.cuda, transformers.__version__, vllm.__version__, torch.cuda.get_device_capability(0))"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN이 없음. GPQA-Diamond(Idavidrein/gpqa)는 gated라서 토큰과 이용 동의가 필요함."
  echo "  export HF_TOKEN=... 또는 $WS/venv_serve/bin/huggingface-cli login"
fi

cat <<EOF

다음 단계 (README.md 참고)
  export WS=$WS HF_HOME=$HF_HOME
  $WS/venv_serve/bin/python $HERE/src/fetch_models.py
  $WS/venv_serve/bin/python $HERE/src/run_matrix.py list
EOF
