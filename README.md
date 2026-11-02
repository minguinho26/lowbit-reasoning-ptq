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

실행 단위 이름: `<model>__<format>[__<algorithm>__<calibration>]`. 예: `qwen3-8b__nvfp4__autoround__trace`, `qwen3-4b__w2a16__rtn`, `qwen3-1.7b__bf16`.

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
$PY src/run_matrix.py list     # 실행 단위 52개 확인
```

장시간 작업은 tmux 안에서 돌리고, 끝나면 pod은 terminate한다. 결과는 볼륨에 남는다.

## 1주차: 확인 작업

순서대로 하고, 각 결과를 기록해둔다.

```bash
# (1) 원본이 32K 안에 들어오는지, 처리량과 출력 길이 (모델당 AIME 30문제)
for m in qwen3-1.7b qwen3-4b qwen3-8b; do
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

자리를 오래 비울 때는 위 확인 작업과 calibration trace 생성을 묶은 queue_week1.sh를 tmux에서 돌린다. 단계가 실패해도 다음 단계로 넘어가고, `AUTO_TERMINATE=1`이면 다 끝난 뒤 pod을 terminate한다. 결과 요약은 /workspace/logs/queue_<시각>/summary.txt.

확인할 것:
- (1) truncation_rate. 원본부터 많이 잘리는 모델은 32K 예산이 부족한 것이다.
- (3) NVFP4 GEMM 커널 이름. Marlin, W4A16, Humming이 들어가면 weight-only로 떨어진 것이라 W4A4 수치가 아니다. Emulation이면 속도 측정이 의미 없다.
- W2A16이 로드되고 출력이 나오는지. GPTQ와 AutoRound가 끝까지 도는지. 메모리가 모자라면 configs의 autoround batch_size를 줄인다.
- throughput_tok_s_this_run, mean_tokens로 PLAN.md 9절 비용을 다시 계산한다.

## 본 실행

```bash
tmux new -s q
AUTO_TERMINATE=1 bash queue_main.sh        # pod의 GPU 전부 사용, 모델 순서 qwen3-8b,qwen3-4b,qwen3-1.7b
```

- GPU마다 워커가 하나씩 붙어서 작업 단위(다운로드, calibration, 양자화 1개, 평가 1개 = 실행 단위 x 벤치 x seed)를 순서대로 나눠 가져간다. 앞 단계가 끝난 단위만 가져가고, 실패한 단위에 걸린 단위는 건너뛴다.
- 첫 모델(Qwen3-8B)의 원본 평가가 맨 앞이다. 2504.04823 Table 5, ReSET 수치와 먼저 비교한다.
- 평가가 다 끝나면 GPU 하나로 속도를 잰다. 그다음 /workspace/logs/main_<시각>/summary.txt를 쓰고 pod을 terminate한다.
- 끊기면 같은 명령으로 다시 돌린다. 끝난 단위는 건너뛴다.
- 진행 상황: `bash status.sh` (끝난 평가의 seed 평균, 실패 단위와 에러 줄, 디스크).
- 볼륨 정리: `bash disk.sh`로 사용량과 지울 후보(설정에 없는 모델, 저장 실패 체크포인트)를 보고, 큐가 멈춘 상태에서 `bash disk.sh clean`.
- 작업 단위 목록 확인: `$PY src/run_matrix.py queue --models qwen3-8b,qwen3-4b,qwen3-1.7b --dry`
- 분석: `$PY src/analyze.py` (results/analysis/*.csv)

`--dry`를 붙이면 실행할 명령만 출력한다.
