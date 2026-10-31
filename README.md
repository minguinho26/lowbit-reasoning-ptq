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

실행 단위 이름: `<model>__<format>[__<algorithm>__<calibration>]`. 예: `qwen3-8b__nvfp4__autoround__trace`, `r1-qwen-1.5b__w2a16__rtn`, `qwen3-1.7b__bf16`.

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
$PY src/run_matrix.py list     # 실행 단위 49개 확인
```

장시간 작업은 tmux 안에서 돌리고, 끝나면 pod은 terminate한다. 결과는 볼륨에 남는다.

## 1주차: 확인 작업

순서대로 하고, 각 결과를 기록해둔다.

```bash
# (1) 원본이 32K 안에 들어오는지, 처리량과 출력 길이 (모델당 AIME 30문제)
for m in qwen3-1.7b r1-qwen-1.5b r1-qwen-7b qwen3-8b; do
  $PY src/evaluate.py --run ${m}__bf16 --bench aime120 --seed 0 --limit 30 2>&1 | tee /workspace/logs/bf16_${m}.log
done

# (2) 공개 수치 재현: Qwen3-8B BF16과 GPTQ W4A16 전체 (2504.04823 Table 5, ReSET과 비교)
$PY src/run_matrix.py calib --model qwen3-8b
$PY src/run_matrix.py quantize --model qwen3-8b --fmt w4a16
$PY src/run_matrix.py evaluate --model qwen3-8b --fmt bf16
$PY src/run_matrix.py evaluate --model qwen3-8b --fmt w4a16

# (3) 커널 경로와 양자화 경로
QPY=/workspace/venv_quant/bin/python
$QPY src/quantize.py --run qwen3-8b__nvfp4__rtn__wikitext2
$PY src/evaluate.py --run qwen3-8b__nvfp4__rtn__wikitext2 --bench aime120 --seed 0 --limit 30 2>&1 | tee /workspace/logs/e_nvfp4.log
grep -i "NVFP4 GEMM\|falling back" /workspace/logs/e_nvfp4.log
$QPY src/quantize.py --run qwen3-8b__w2a16__rtn
$PY src/evaluate.py --run qwen3-8b__w2a16__rtn --bench aime120 --seed 0 --limit 10 2>&1 | tee /workspace/logs/e_w2.log
$QPY src/quantize.py --run qwen3-8b__nvfp4__gptq__trace
$QPY src/quantize.py --run qwen3-8b__nvfp4__autoround__trace
```

확인할 것:
- (1) truncation_rate. 원본부터 많이 잘리는 모델은 32K 예산이 부족한 것이다.
- (3) NVFP4 GEMM 커널 이름. Marlin, W4A16, Humming이 들어가면 weight-only로 떨어진 것이라 W4A4 수치가 아니다. Emulation이면 속도 측정이 의미 없다.
- W2A16이 로드되고 출력이 나오는지. GPTQ와 AutoRound가 끝까지 도는지. 메모리가 모자라면 configs의 autoround batch_size를 줄인다.
- throughput_tok_s_this_run, mean_tokens로 PLAN.md 9절 비용을 다시 계산한다.

## 본 실행

```bash
for m in qwen3-1.7b qwen3-8b r1-qwen-1.5b r1-qwen-7b; do
  $PY src/run_matrix.py calib --model $m
  $PY src/run_matrix.py quantize --model $m
done
for m in qwen3-1.7b qwen3-8b r1-qwen-1.5b r1-qwen-7b; do
  $PY src/run_matrix.py evaluate --model $m
done
# 속도: 모델 하나의 원본과 양자화 설정을 같은 pod에서 연달아
for m in qwen3-1.7b qwen3-8b r1-qwen-1.5b r1-qwen-7b; do
  $PY src/run_matrix.py speed --model $m
done
$PY src/analyze.py             # results/analysis/*.csv
```

`--dry`를 붙이면 실행할 명령만 출력한다.
