# lowbit-reasoning-ptq

리즈닝 모델 저비트 PTQ 실험 코드. 계획과 설계 근거는 PLAN.md, 실험 설정은 configs/experiment.yaml.

## 파일

| 파일 | 역할 | 환경 |
|---|---|---|
| setup_pod.sh | pod 초기 세팅(venv 2개, HF cache를 /workspace에) | - |
| requirements-quant.txt / requirements-serve.txt | 버전 고정 | - |
| configs/experiment.yaml | 모델, 포맷, 알고리즘, calibration, 벤치, 디코딩, 속도 측정 설정 | - |
| src/common.py | 설정 로딩, 실행 단위 이름 규칙, 경로 | 공통 |
| src/fetch_models.py | 원본 다운로드(HF 형식 파일만) | serve |
| src/benchmarks.py | AIME-120, GPQA-Diamond 로딩, 프롬프트, 채점 | serve |
| src/engine.py | vLLM 엔진, 샘플링 설정 | serve |
| src/make_calib.py | calibration trace 생성(traces), 토큰 윈도우 추출(windows) | traces: serve, windows: quant |
| src/quantize.py | llm-compressor PTQ(RTN, GPTQ, AutoRound) | quant |
| src/evaluate.py | 벤치 평가, 문제 단위로 이어서 돌리기 가능 | serve |
| src/bench_speed.py | 고정 길이 decode 처리량 | serve |
| src/analyze.py | 저하 폭, paired bootstrap CI, 길이 비율, 요인 효과, effective speedup | serve |
| src/run_matrix.py | 실행 단위를 펼쳐서 안 끝난 것만 순서대로 실행 | serve |

실행 단위 이름: `<model>__<format>[__<algorithm>__<calibration>]`. 예: `qwen35-9b__nvfp4__autoround__trace`, `ministral3-3b__w2a16__rtn`, `qwen35-4b__bf16`.

## RunPod 준비

1. Network Volume 생성(200GB 정도). RTX 5090 재고가 있는 데이터센터로 고른다.
2. Pod: RTX 5090 × 1, PyTorch 템플릿, 볼륨을 /workspace에 연결, container disk 30GB.
3. pod에서 이 repo를 clone.

```bash
cd /workspace && git clone https://github.com/minguinho26/lowbit-reasoning-ptq.git && cd lowbit-reasoning-ptq
export HF_TOKEN=...            # GPQA-Diamond 접근용
bash setup_pod.sh
export WS=/workspace HF_HOME=/workspace/hf_cache
PY=/workspace/venv_serve/bin/python
$PY src/fetch_models.py
$PY src/run_matrix.py list     # 실행 단위 50개 확인
```

장시간 작업은 tmux 안에서 돌리고, 끝나면 pod은 terminate한다. 결과는 볼륨에 남는다.

## 1주차: 확인 작업

순서대로 하고, 각 결과를 기록해둔다.

```bash
# (1) 원본 평가가 도는지, 실측 처리량과 출력 길이 (모델당 20문제)
$PY src/evaluate.py --run qwen35-4b__bf16 --bench aime120 --seed 0 --limit 20
$PY src/evaluate.py --run ministral3-8b__bf16 --bench aime120 --seed 0 --limit 20

# (2) 파이프라인 검증: Qwen3-8B BF16 전체 AIME-120, GPQA-D를 2504.04823 Table 5와 비교
$PY src/run_matrix.py all --model qwen3-8b

# (3) 커널 경로: 양자화 후 vLLM 로그에서 확인
$PY src/run_matrix.py calib --model qwen35-4b
/workspace/venv_quant/bin/python src/quantize.py --run qwen35-4b__nvfp4__rtn__wikitext2
$PY src/evaluate.py --run qwen35-4b__nvfp4__rtn__wikitext2 --bench aime120 --seed 0 --limit 20 2>&1 | grep -i "NVFP4 GEMM\|falling back"
/workspace/venv_quant/bin/python src/quantize.py --run qwen35-4b__w2a16__rtn
$PY src/evaluate.py --run qwen35-4b__w2a16__rtn --bench aime120 --seed 0 --limit 20 2>&1 | grep -i "Using .* for\|Humming"
```

확인할 것:
- NVFP4 GEMM 커널 이름. Marlin, W4A16, Humming이 들어가면 weight-only로 떨어진 것이라 W4A4 수치가 아니다. Emulation이면 속도 측정이 의미 없다.
- W2A16이 로드되고 출력이 나오는지.
- Qwen3.5(하이브리드), Ministral 3에서 GPTQ와 AutoRound가 끝까지 도는지. 메모리가 모자라면 configs의 autoround batch_size를 줄인다.
- (1)과 INT2 결과의 throughput_tok_s_this_run, mean_tokens로 PLAN.md 9절 비용을 다시 계산한다.

## 본 실행

```bash
for m in qwen35-4b qwen35-9b ministral3-3b ministral3-8b; do
  $PY src/run_matrix.py calib --model $m
  $PY src/run_matrix.py quantize --model $m
done
for m in qwen35-4b qwen35-9b ministral3-3b ministral3-8b; do
  $PY src/run_matrix.py evaluate --model $m
done
# 속도: 모델 하나의 원본과 양자화 설정을 같은 pod에서 연달아
for m in qwen35-4b qwen35-9b ministral3-3b ministral3-8b; do
  $PY src/run_matrix.py speed --model $m
done
$PY src/analyze.py             # results/analysis/*.csv
```

`--dry`를 붙이면 실행할 명령만 출력한다.
