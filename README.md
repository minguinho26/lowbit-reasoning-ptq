# lowbit-reasoning-ptq

LLM의 weight와 activation 데이터 타입에 따른 reasoning 태스크의 정확도 차이를 비교했다. 비교 항목은 다음과 같다.

1. Weight type 변경: Activation을 BF16(16bit)으로 고정하고, weight를 NVFP4와 INT2로 양자화해 정확도를 비교했다.
2. Activation 데이터 타입 변경: Weight type을 NVFP4로 고정하고, activation을 BF16(16bit)으로 유지하는 경우와 FP4(4bit)로 양자화하는 경우의 정확도를 비교했다.

## 실험에 사용한 모델과 벤치마크

### 모델

[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)와 [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)를 RTN, [GPTQ](https://arxiv.org/pdf/2210.17323), [AWQ](https://arxiv.org/pdf/2306.00978)로 양자화했다. 양자화 모델과 비교하기 위해 각 모델의 BF16 정확도도 측정했다. Weight에는 BF16, NVFP4, INT2를 사용했고, activation에는 BF16(16bit, 표에서 A16으로 표기)과 FP4(4bit, 표에서 A4로 표기)를 사용했다. Calibration data로는 [WikiText2](https://huggingface.co/datasets/Salesforce/wikitext)의 train split과 각 BF16 모델이 [NuminaMath-1.5](https://huggingface.co/datasets/project-numina/NuminaMath-1.5)의 문제를 풀며 생성한 reasoning trace를 사용했다.

> Reasoning trace로 calibration 샘플을 만들 때는 문제 프롬프트와 BF16 모델이 생성한 응답을 함께 사용했다.

제작 과정은 다음과 같다.

1. 각 calibration data의 텍스트를 이어 붙인 뒤 토큰화한다.
2. 무작위 시작점에서 연속된 2,048토큰을 잘라내는 과정을 반복해 데이터별로 샘플 128개를 만든다.
   1. 샘플링 seed는 0으로 고정했다.

### 벤치마크와 측정 지표

AIME-120([AIME90](https://huggingface.co/datasets/xiaoyuanliu/AIME90) + [AIME 2025](https://huggingface.co/datasets/yentinglin/aime_2025))과 [GPQA-Diamond](https://huggingface.co/datasets/Idavidrein/gpqa)의 전체 데이터를 평가에 사용했다. 두 벤치마크는 각각 120문항과 198문항으로 구성된다. seed를 0, 1, 2로 바꿔 각각 응답을 생성하고 평가했으며, 평가에 사용한 지표는 아래와 같다.

- 정확도: 전체 문항 중 정답을 맞힌 비율이다. 세 seed의 정확도를 평균 ± 표본표준편차(std, ddof=1)로 표시했다.
- 출력 길이: 풀이 과정과 최종 답변을 포함해 생성한 응답의 토큰 수다.
- 최대 출력 길이에 도달한 비율: 전체 응답 중 `max_tokens=32768`에 도달해 생성이 중단된 응답의 비율이다. 이 응답은 오답으로 처리했다.
- 처리량: 초당 생성한 출력 토큰 수(output tok/s)다. 여러 요청을 함께 처리하면 모든 요청의 출력 토큰 수를 합산한다.

#### 처리량 측정 설정

BF16과 양자화 모델을 같은 GPU에서 동일한 설정으로 측정했다. 입력 길이는 512토큰, 출력 길이는 2,048토큰으로 고정하고 batch size 1, 16, 64를 사용했다. 각 조건에서 세 번 측정한 시간의 중앙값으로 처리량을 계산했다. 측정 시간에는 prefill도 포함했다. 이 설정은 처리량 측정에만 적용했으며, 벤치마크 평가에는 각 문항의 프롬프트를 사용했다.

### 실험 환경 및 설정

| 작업 | Qwen3-1.7B | Qwen3-8B |
|---|---|---|
| 양자화 및 벤치마크 평가 | RTX 5090 | RTX 5090, RTX PRO 4500 Blackwell |
| 처리량 측정 | RTX 5090 | RTX 5090 |

> 벤치마크 평가: 정확도, 출력 길이, 최대 출력 길이에 도달한 비율을 측정, 처리량 측정: 각 모델의 BF16과 양자화 모델을 측정.

- 평가 패키지: vLLM 0.32.0, PyTorch 2.13.0. 모델별 버전 기록은 `reports/`의 `environment.json`에 있다.
- CUDA: 12.8

#### 인퍼런스 세팅

- Thinking 모드: 활성화
- Sampling: temperature 0.6, top-p 0.95, top-k 비활성화
- KV cache dtype: BF16

#### 양자화 설정

양자화 대상에서 임베딩, norm, `lm_head`를 제외했다. Weight-only RTN은 calibration이 필요하지 않아 데이터별로 나누지 않았다. 표에서는 해당 RTN 행의 Calibration data를 `-`로 표시했다. AWQ의 출력 cache 호환성 패치 적용 여부는 체크포인트 메타데이터에 기록했다.

> NVFP4 (A4)의 RTN은 activation을 FP4로 양자화하는 데 필요한 global scale을 calibration data로 정한다. 데이터에 따라 관측되는 activation의 범위와 global scale이 달라질 수 있으므로, WikiText2와 reasoning trace로 각각 calibration한 체크포인트를 만들고 평가했다. 출처: [LLM Compressor 문서](https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/compression_schemes/#nvfp4)

실험에 사용한 프롬프트 및 채점 코드는 `src/benchmarks.py`에 있고, 양자화 설정은 `configs/experiment.yaml`에 있다.

## 실험 결과

실험에 사용한 각 양자화 체크포인트는 문서 하단의 [체크포인트](#체크포인트) 섹션에 있는 Hugging Face 링크에서 다운로드할 수 있다.

### Qwen3-8B 결과

![Qwen3-8B 결과: weight type별 굵은 구분선](assets/qwen3-8b-results.svg)

<details>
<summary>텍스트 표 보기</summary>

| Weight type (Activation type) | Quantization | Calibration data | AIME-120 (%) | GPQA-Diamond (%) | Average (%) |
|---|---|---|---:|---:|---:|
| BF16 | - | - | 71.39 ± 0.96 | 61.11 ± 2.31 | 66.25 ± 1.62 |
| NVFP4 (A16) | RTN | - | 66.39 ± 4.59 | 56.40 ± 0.29 | 61.39 ± 2.36 |
| NVFP4 (A16) | GPTQ | WikiText2 | 68.61 ± 2.55 | 58.42 ± 2.78 | 63.51 ± 2.13 |
| NVFP4 (A16) | GPTQ | Trace | 67.22 ± 2.68 | 57.91 ± 0.58 | 62.57 ± 1.05 |
| NVFP4 (A16) | AWQ | WikiText2 | 68.33 ± 1.67 | 59.09 ± 3.54 | 63.71 ± 0.93 |
| NVFP4 (A16) | AWQ | Trace | 71.11 ± 3.76 | 58.42 ± 1.05 | 64.76 ± 1.79 |
| NVFP4 (A4) | RTN | WikiText2 | 60.56 ± 3.37 | 57.24 ± 1.05 | 58.90 ± 1.71 |
| NVFP4 (A4) | RTN | Trace | 62.78 ± 2.93 | 54.55 ± 1.01 | 58.66 ± 1.80 |
| NVFP4 (A4) | GPTQ | WikiText2 | 63.61 ± 3.94 | 56.90 ± 1.54 | 60.26 ± 2.38 |
| NVFP4 (A4) | GPTQ | Trace | 67.22 ± 1.73 | 57.24 ± 3.25 | 62.23 ± 0.76 |
| NVFP4 (A4) | AWQ | WikiText2 | 63.33 ± 2.20 | 55.56 ± 1.34 | 59.44 ± 1.74 |
| NVFP4 (A4) | AWQ | Trace | 63.89 ± 1.73 | 57.74 ± 0.77 | 60.82 ± 0.72 |
| INT2 (A16) | RTN | - | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| INT2 (A16) | GPTQ | WikiText2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| INT2 (A16) | GPTQ | Trace | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| INT2 (A16) | AWQ | WikiText2 | 0.28 ± 0.48 | 0.00 ± 0.00 | 0.14 ± 0.24 |
| INT2 (A16) | AWQ | Trace | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

</details>

### Qwen3-1.7B 결과

![Qwen3-1.7B 결과: weight type별 굵은 구분선](assets/qwen3-1.7b-results.svg)

<details>
<summary>텍스트 표 보기</summary>

| Weight type (Activation type) | Quantization | Calibration data | AIME-120 (%) | GPQA-Diamond (%) | Average (%) |
|---|---|---|---:|---:|---:|
| BF16 | - | - | 40.00 ± 5.07 | 40.40 ± 3.54 | 40.20 ± 3.49 |
| NVFP4 (A16) | RTN | - | 30.56 ± 3.15 | 36.70 ± 2.04 | 33.63 ± 0.58 |
| NVFP4 (A16) | GPTQ | WikiText2 | 29.72 ± 3.76 | 35.19 ± 2.28 | 32.45 ± 2.54 |
| NVFP4 (A16) | GPTQ | Trace | 36.94 ± 1.73 | 36.36 ± 2.67 | 36.65 ± 1.66 |
| NVFP4 (A16) | AWQ | WikiText2 | 29.44 ± 3.15 | 36.03 ± 1.62 | 32.74 ± 1.93 |
| NVFP4 (A16) | AWQ | Trace | 30.83 ± 0.83 | 36.70 ± 1.54 | 33.77 ± 0.99 |
| NVFP4 (A4) | RTN | WikiText2 | 22.78 ± 2.10 | 33.33 ± 3.54 | 28.06 ± 2.74 |
| NVFP4 (A4) | RTN | Trace | 23.33 ± 3.00 | 32.15 ± 1.46 | 27.74 ± 2.09 |
| NVFP4 (A4) | GPTQ | WikiText2 | 23.89 ± 1.73 | 32.49 ± 1.77 | 28.19 ± 1.65 |
| NVFP4 (A4) | GPTQ | Trace | 29.44 ± 2.10 | 33.33 ± 0.00 | 31.39 ± 1.05 |
| NVFP4 (A4) | AWQ | WikiText2 | 28.89 ± 3.15 | 33.50 ± 1.05 | 31.20 ± 1.11 |
| NVFP4 (A4) | AWQ | Trace | 28.89 ± 2.93 | 38.55 ± 2.96 | 33.72 ± 2.33 |
| INT2 (A16) | RTN | - | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| INT2 (A16) | GPTQ | WikiText2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| INT2 (A16) | GPTQ | Trace | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| INT2 (A16) | AWQ | WikiText2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| INT2 (A16) | AWQ | Trace | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

</details>

## 실험 결과 분석

### Weight type 변경에 따른 분석

#### Qwen3-8B

NVFP4 (A16)의 평균 정확도는 AIME-120에서 66.39~71.11%, GPQA-Diamond에서 56.40~59.09%였으며, BF16 대비 각각 0.28~5.00 pp, 2.02~4.71 pp 낮았다. AIME-120에서는 AWQ + Trace가 71.11%로 BF16보다 0.28 pp 낮았지만, 같은 설정의 GPQA-Diamond 정확도는 58.42%로 BF16보다 2.69 pp 낮았다. INT2 (A16)은 AIME-120의 AWQ + WikiText2에서만 0.28%였고, 나머지 설정과 GPQA-Diamond 전체에서는 0%였다. [8B 평가 결과](reports/qwen3-8b/main_table.csv)

#### Qwen3-1.7B

NVFP4 (A16)의 평균 정확도는 AIME-120에서 29.44~36.94%, GPQA-Diamond에서 35.19~36.70%였으며, BF16 대비 각각 3.06~10.56 pp, 3.70~5.22 pp 낮았다. NVFP4 (A16) 중 AIME-120 정확도가 가장 높은 설정은 GPTQ + Trace의 36.94%였다. INT2 (A16)은 두 태스크의 모든 설정에서 0%였다. INT2 정확도 저하의 원인은 저장된 짧은 reference 및 kernel 진단만으로 특정하지 못했다. [1.7B 평가 결과](reports/qwen3-1.7b/main_table.csv)

### Activation data type 변경에 따른 분석

#### Qwen3-8B

AIME-120에서는 GPTQ + Trace가 A16과 A4 모두 67.22%였고, 나머지 다섯 비교는 A4에서 평균 정확도가 3.61~7.22 pp 낮았다. AWQ + Trace는 71.11%에서 63.89%로 7.22 pp 낮아졌다. GPQA-Diamond에서는 RTN + WikiText2가 56.40%에서 57.24%로 0.84 pp 높아졌고, 나머지 다섯 비교는 0.67~3.54 pp 낮아졌다. [8B 설정별 비교](reports/qwen3-8b/factor_effects.csv)

#### Qwen3-1.7B

AIME-120에서 A16 대비 A4의 평균 정확도는 RTN과 GPTQ에서 5.83~7.78 pp, AWQ에서 0.56~1.94 pp 낮았다. GPTQ + Trace는 A16의 36.94%에서 A4의 29.44%로 7.50 pp 낮아져, 같은 비교에서 평균이 유지된 8B와 다른 결과를 보였다. 반대로 AWQ + Trace의 감소폭은 1.94 pp로 8B의 7.22 pp보다 작았다. GPQA-Diamond에서는 AWQ + Trace가 36.70%에서 38.55%로 1.85 pp 높아졌고, 나머지 다섯 비교는 2.53~4.55 pp 낮아졌다. [1.7B 설정별 비교](reports/qwen3-1.7b/factor_effects.csv)

## 한계점

### GPU

8B의 벤치마크 평가에는 RTX 5090과 RTX PRO 4500 Blackwell에서 생성한 결과가 포함돼 있다. GPU 변경의 영향을 별도로 분리해 측정하지 않았으므로, 작은 점수 차이를 양자화 설정만의 영향으로 단정하기 어렵다.

### 실험에 사용한 모델

Qwen3의 두 모델 크기만 평가했으므로 다른 모델 계열이나 크기에서도 같은 경향이 나타나는지는 확인하지 않았다.

### 양자화 설정

NVFP4 (A16)과 INT2 (A16)은 데이터 타입, 비트 수, group 크기, scale 방식이 함께 달라지므로, 어느 한 요소만의 영향을 분리한 비교는 아니다.

## 저장된 결과

모델별 평가 결과, 처리량 원본 측정값, INT2 진단 결과와 실험 설정은 `reports/`에 있다. 각 파일의 내용과 열의 의미, 정확도 차이의 뺄셈 방향은 [평가 결과 파일 안내](reports/README.md)에 정리했다.

## 서버 환경 및 실행 방법

### 서버 환경

CUDA를 사용할 수 있는 Linux GPU 서버에서 실행한다. NVFP4 (A4)의 activation을 FP4로 처리하기 위해 Blackwell GPU 사용을 권장한다. Python 3.12를 사용했으며, 양자화용 `venv_quant`와 평가 및 처리량 측정용 `venv_serve`를 별도로 구성해 실험에 사용했다. [지원 GPU 안내](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-scheme/)

### 실행 방법

저장소를 내려받은 뒤 저장소 루트로 이동한다.

```bash
git clone https://github.com/minguinho26/lowbit-reasoning-ptq.git
cd lowbit-reasoning-ptq
```

아래 명령어로 가상환경을 만들고 필요한 패키지를 설치한다. `WS`는 모델과 실험 결과를 저장할 디렉터리 경로다.

> GPQA-Diamond를 사용하려면 Hugging Face에서 이용 동의를 완료해야 한다. 아래 HF 인증 단계에서는 동의한 계정으로 로그인해야 서버에서 데이터셋을 다운로드할 수 있다. [HF 인증 안내](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

```bash
export WS="$HOME/lowbit-reasoning-ptq-workspace"
export HF_HOME="$WS/hf_cache"
mkdir -p "$WS"
python3.12 -m venv "$WS/venv_quant"
python3.12 -m venv "$WS/venv_serve"
"$WS/venv_quant/bin/python" -m pip install -r requirements-quant.txt
"$WS/venv_serve/bin/python" -m pip install -r requirements-serve.txt
PY="$WS/venv_serve/bin/python"

# 위에서 지정한 HF_HOME 경로에 인증 정보를 저장
"$WS/venv_serve/bin/hf" auth login

# BF16 모델 다운로드
$PY src/fetch_models.py

# 실험 목록과 실행 계획 확인
$PY src/run_matrix.py list
$PY src/run_matrix.py queue --models qwen3-8b,qwen3-1.7b --dry # 실행할 작업만 표시하며 실제 실험은 수행하지 않음
```

공개한 양자화 체크포인트를 `$WS/models/<실험 이름>/`에 내려받으면, 다시 양자화하지 않고 성능을 측정할 수 있다.

> 실험 이름은 `<model>__<format>[__<algorithm>__<calibration>]` 형식이다. 예를 들어 `qwen3-8b__nvfp4a16__awq__trace`는 Qwen3-8B에 NVFP4 weight, BF16 activation, AWQ, Trace calibration을 사용한 실험이다.

실험을 돌리는 명령어 종류는 아래와 같다.

```bash
# 실험 하나를 seed 하나로 전체 문항 평가
$PY src/evaluate.py --run qwen3-8b__nvfp4a16__awq__trace --bench aime120 --seed 0

# 일부 문항 확인: 전체 평가와 별도 파일에 저장
$PY src/evaluate.py --run qwen3-8b__bf16 --bench aime120 --seed 0 --limit 30

# 저장된 원본 평가 기록 재분석: 추론은 실행하지 않음
$PY src/analyze.py

# 전체 실험 실행: 장시간 실행 가능한 세션 사용
bash queue_main.sh
```

생성된 응답과 문항별 채점 결과는 `$WS/results/eval/<실험 이름>/`에 함께 저장한다. 평가가 끝나면 문항 수와 정확도 등을 `.done.json` 파일에 요약하고, 분석 단계에서 집계한 CSV는 `$WS/results/analysis/`에 저장한다. 파일명과 저장 경로의 예시는 다음과 같다.

```text
$WS/results/
├── eval/
│   ├── qwen3-8b__nvfp4a16__awq__trace/
│   │   ├── aime120__s0.jsonl          # 생성 응답과 문항별 채점 결과
│   │   └── aime120__s0.done.json      # 전체 문항의 평가 요약
│   └── qwen3-8b__bf16/
│       └── aime120__s0.limit30.jsonl  # --limit 30으로 평가한 결과
└── analysis/
    ├── main_table.csv                # 실험별 평가 요약
    ├── factor_effects.csv            # 설정 간 정확도 차이
    └── speed.csv                     # 처리량 요약
```

전체 실험은 양자화, 평가, INT2 진단, 처리량 측정, 분석 순서로 실행한다.

## 체크포인트
실험에 사용한 양자화 체크포인트의 다운로드 링크는 아래와 같다.

### Qwen3-8B

#### NVFP4 (A16)

| Quantization method | Calibration data | Link |
|---|---|---|
| RTN | - | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-8B-NVFP4A16-RTN) |
| GPTQ | WikiText2 | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-8B-NVFP4A16-GPTQ-WikiText2) |
| GPTQ | Trace | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-8B-NVFP4A16-GPTQ-Trace) |
| AWQ | WikiText2 | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-8B-NVFP4A16-AWQ-WikiText2) |
| AWQ | Trace | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-8B-NVFP4A16-AWQ-Trace) |

#### NVFP4 (A4)

| Quantization method | Calibration data | Link |
|---|---|---|
| RTN | WikiText2 | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-8B-NVFP4-RTN-WikiText2) |
| RTN | Trace | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-8B-NVFP4-RTN-Trace) |
| GPTQ | WikiText2 | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-8B-NVFP4-GPTQ-WikiText2) |
| GPTQ | Trace | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-8B-NVFP4-GPTQ-Trace) |
| AWQ | WikiText2 | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-8B-NVFP4-AWQ-WikiText2) |
| AWQ | Trace | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-8B-NVFP4-AWQ-Trace) |

#### INT2 (A16)

| Quantization method | Calibration data | Link |
|---|---|---|
| RTN | - | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-8B-W2A16-RTN) |
| GPTQ | WikiText2 | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-8B-W2A16-GPTQ-WikiText2) |
| GPTQ | Trace | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-8B-W2A16-GPTQ-Trace) |
| AWQ | WikiText2 | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-8B-W2A16-AWQ-WikiText2) |
| AWQ | Trace | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-8B-W2A16-AWQ-Trace) |

### Qwen3-1.7B

#### NVFP4 (A16)

| Quantization method | Calibration data | Link |
|---|---|---|
| RTN | - | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-1.7B-NVFP4A16-RTN) |
| GPTQ | WikiText2 | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-1.7B-NVFP4A16-GPTQ-WikiText2) |
| GPTQ | Trace | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-1.7B-NVFP4A16-GPTQ-Trace) |
| AWQ | WikiText2 | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-1.7B-NVFP4A16-AWQ-WikiText2) |
| AWQ | Trace | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-1.7B-NVFP4A16-AWQ-Trace) |

#### NVFP4 (A4)

| Quantization method | Calibration data | Link |
|---|---|---|
| RTN | WikiText2 | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-1.7B-NVFP4-RTN-WikiText2) |
| RTN | Trace | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-1.7B-NVFP4-RTN-Trace) |
| GPTQ | WikiText2 | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-1.7B-NVFP4-GPTQ-WikiText2) |
| GPTQ | Trace | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-1.7B-NVFP4-GPTQ-Trace) |
| AWQ | WikiText2 | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-1.7B-NVFP4-AWQ-WikiText2) |
| AWQ | Trace | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-1.7B-NVFP4-AWQ-Trace) |

#### INT2 (A16)

| Quantization method | Calibration data | Link |
|---|---|---|
| RTN | - | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-1.7B-W2A16-RTN) |
| GPTQ | WikiText2 | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-1.7B-W2A16-GPTQ-WikiText2) |
| GPTQ | Trace | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-1.7B-W2A16-GPTQ-Trace) |
| AWQ | WikiText2 | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-1.7B-W2A16-AWQ-WikiText2) |
| AWQ | Trace | [Hugging Face](https://huggingface.co/Minguinho-zeze/Qwen3-1.7B-W2A16-AWQ-Trace) |
