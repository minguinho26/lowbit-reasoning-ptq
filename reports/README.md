# 평가 결과 파일 안내

이 폴더에는 모델별 평가 결과와 처리량 측정값, INT2 진단 파일, 실험 설정이 있다. 모델 가중치는 [체크포인트 목록](../README.md#체크포인트)의 Hugging Face 링크에서 내려받을 수 있다.

- [Qwen3-8B 결과](qwen3-8b/)
- [Qwen3-1.7B 결과](qwen3-1.7b/)

## 파일 구성

아래 개수는 모델 하나를 기준으로 한다. BF16을 포함한 17개 실험에서 AIME-120과 GPQA-Diamond를 seed 0, 1, 2로 평가했다.

| 파일 또는 폴더 | 내용 | 모델별 데이터 수 |
|---|---|---|
| `main_table.csv` | 실험과 벤치마크별 정확도, 출력 길이, BF16 대비 차이 | 34행 |
| `seed_results.json` | 실험, 벤치마크, seed별 정확도와 평균 출력 길이 | 평가 102건 |
| `per_problem_metrics.csv` | 응답별 정답 여부, 토큰 수, 생성 종료 사유 | 16,218행 |
| `factor_effects.csv` | 알고리즘, calibration 데이터, weight와 activation 변경에 따른 정확도 차이 | 72행 |
| `speed.csv` | 실험과 batch size별 처리량, BF16 대비 처리량 비율 | 51행 |
| `speed_measurements/` | 실험별 세 번의 측정 시간과 처리량 | JSON 17개 |
| `diagnostics/` | INT2 체크포인트의 Transformers 및 vLLM 추론 결과와 비교 | INT2 실험 5개 |
| `checkpoints.json` | 공개한 HF 저장소와 실험 이름, 가중치 크기, 파일 목록 | 체크포인트 16개 |
| `collection.json` | HF collection 주소와 체크포인트 수 | Collection 1개 |
| `hf_commits.json` | HF 저장소별 업로드 확인 기록과 revision | 저장소 16개 |
| `experiment.yaml` | 실험 당시 모델, 양자화, calibration, 평가 설정 | 설정 파일 1개 |
| `environment.json` | 실험 당시 코드 commit과 평가 패키지 버전 | 환경 파일 1개 |

1.7B 폴더에는 다음 파일도 있다.

| 파일 | 내용 |
|---|---|
| `quantization_metadata.json` | 실험별 양자화 알고리즘 인자, calibration 설정, 소요 시간, GPU, 패키지 버전 |
| `validation.json` | 파일 수, 문항 ID, 요약 지표를 확인한 기록과 원본 파일의 SHA-256. CI는 재계산하지 않고 저장된 하한과 상한의 순서만 확인 |

## 평가 결과 설명

### 실험 이름

파일의 `run`은 실험 이름이며, `<model>__<format>[__<algorithm>__<calibration>]` 형식이다. 이름에 사용하는 값은 다음과 같다.

| 값 | 의미 |
|---|---|
| `bf16` | 양자화하지 않은 BF16 모델 |
| `nvfp4a16` | NVFP4 weight, BF16 activation |
| `nvfp4` | NVFP4 weight, FP4 activation |
| `w2a16` | INT2 weight, BF16 activation |
| `wikitext2`, `trace` | WikiText2 또는 BF16 모델이 생성한 reasoning trace로 calibration |
| `aime120`, `gpqa_diamond` | AIME-120 또는 GPQA-Diamond 벤치마크 |

Weight-only RTN은 calibration을 사용하지 않아 실험 이름에 calibration 항목이 없다.

### 정확도와 출력 길이

`main_table.csv`는 실험별로 세 seed의 응답을 모아 집계한 파일이며, 한 행은 벤치마크 하나의 결과다. `seed_results.json`에는 seed별 결과가 있고, 루트 README의 정확도 평균과 std도 이 파일로 계산했다.

| 열 | 의미 |
|---|---|
| `n_problems`, `n_samples` | 문항 수와 세 seed에서 생성한 전체 응답 수 |
| `acc` | 정답을 맞힌 응답의 비율. `0.4`는 40% |
| `median_tokens`, `p90_tokens` | 생성된 응답의 토큰 수 중앙값과 90백분위수. 입력 토큰은 제외 |
| `truncation_rate` | 최대 출력 길이에 도달해 중단된 응답의 비율 |
| `loop_rate` | 토큰 반복 규칙에 걸린 응답의 비율. 판별 코드는 [is_looping](../src/evaluate.py) 참고 |
| `acc_drop` | 양자화 모델의 정확도에서 BF16 정확도를 뺀 값. 음수이면 BF16보다 낮음 |
| `acc_drop_lo`, `acc_drop_hi` | 정확도 차이의 95% 신뢰구간 하한과 상한 |
| `acc_drop_distinguishable` | 위 신뢰구간이 0을 포함하지 않는지 여부 |
| `len_ratio_completed` | BF16과 양자화 모델 모두 길이 제한 전에 끝낸 응답이 있는 문항의 출력 길이 비율 |
| `len_ratio_both_correct` | BF16과 양자화 모델 모두 정답 응답이 있는 문항의 출력 길이 비율 |
| `quantized_param_fraction` | 전체 파라미터 중 양자화한 파라미터의 비율 |

출력 길이 비율은 조건에 맞는 응답의 문항별 평균 길이로 `양자화 모델 / BF16`을 계산한 뒤 기하평균한 값이다. `_lo`와 `_hi`는 95% 신뢰구간, `_n`은 계산에 사용한 문항 수다. BF16 행이나 계산할 응답이 없는 경우에는 값이 비어 있을 수 있다.

`per_problem_metrics.csv`는 응답 하나를 한 행에 저장한다. `run`, `bench`, `seed`, `uid`로 응답을 구분하며, `correct`는 정답 여부, `n_tokens`는 출력 토큰 수, `n_prompt_tokens`는 입력 토큰 수다. `truncated`는 길이 제한으로 중단됐는지, `looping`은 토큰 반복 규칙에 걸렸는지 나타낸다. `finish_reason`은 생성 종료 사유다.

### 정확도 차이

`factor_effects.csv`에는 비교한 실험을 `run_a`와 `run_b`, 고정한 조건을 `fixed`, 정확도 차이를 `diff`에 기록했다. 뺄셈 방향은 비교 종류에 따라 다르다.

| `factor` | 비교 내용 | `diff` 계산 |
|---|---|---|
| `algorithm` | 같은 weight와 activation type, calibration 데이터에서 비교한 알고리즘 중 최고 정확도와 최저 정확도의 차이 | `run_a - run_b` |
| `calibration` | 같은 weight와 activation type, 알고리즘에서 두 calibration 결과 중 최고 정확도와 최저 정확도의 차이 | `run_a - run_b` |
| `vs_rtn` | GPTQ 또는 AWQ의 정확도와 RTN 정확도의 차이 | `run_a - run_b` |
| `activation` | NVFP4 (A4)와 NVFP4 (A16)의 정확도 차이 | `run_b - run_a` |
| `weight_format` | INT2 (A16)과 NVFP4 (A16)의 정확도 차이 | `run_b - run_a` |

`diff=-0.05`는 위 뺄셈 방향을 기준으로 정확도 차이가 -5 percentage point(pp)라는 뜻이다. `diff_lo`와 `diff_hi`는 차이의 95% 신뢰구간(CI)이다. 정확도 차이의 CI는 문항별 세 seed의 평균 점수를 구한 뒤, 두 실험에서 같은 문항을 함께 다시 뽑는 paired bootstrap을 10,000회 수행해 계산했다. Calibration 샘플링이나 양자화 재실행의 변동은 포함하지 않았고, 최고와 최저 결과를 골라 비교한 영향과 다중 비교도 보정하지 않았다.

## 처리량 측정 파일

`speed.csv`의 한 행은 실험 하나와 batch size 하나의 결과다. `output_tok_s`는 모든 요청에서 생성한 출력 토큰 수를 합산한 초당 처리량이고, `speedup_vs_bf16`은 같은 batch size에서 측정한 `양자화 모델 처리량 / BF16 처리량`이다. `checkpoint_gb`는 가중치 파일 크기이다. `speed_measurements/<실험 이름>.json`에는 이 요약값을 계산한 반복별 실행 시간과 입력 및 출력 길이, GPU, 측정 시각이 있다.

| 항목 | 의미 |
|---|---|
| `seconds_all` | 같은 조건에서 세 번 측정한 실행 시간 |
| `seconds_median` | 세 실행 시간의 중앙값 |
| `output_tok_s` | `batch_size × 출력 길이 / seconds_median` |
| `per_seq_tok_s` | `출력 길이 / seconds_median` |
| `checkpoint_bytes` | 가중치 파일 크기. 단위는 byte |

`effective_speedup_aime120`과 `effective_speedup_gpqa_diamond`는 `speedup_vs_bf16 / len_ratio_completed`로 계산했다. 양자화 실험의 batch size 16 행에 기록했으며, 출력 길이 비율을 구할 수 없으면 비어 있다.
> 실제 벤치마크 응답 시간을 측정한 값은 아니다.

## INT2 진단 파일

`diagnostics/`에는 INT2 체크포인트를 Transformers와 vLLM으로 실행해 비교한 결과가 있다. 진단용 질문 3개에서 최대 48토큰을 생성했다.

| 파일 | 의미 |
|---|---|
| `reference.json` | 저장된 INT2 가중치를 BF16 텐서로 변환해 Transformers로 추론한 결과 |
| `kernel.json` | INT2 체크포인트를 vLLM 압축 커널로 추론한 결과 |
| `comparison.json` | 두 방식의 첫 토큰 일치 여부, 상위 20개 토큰의 겹치는 개수, 공통 토큰의 log probability 차이, 생성문 비교 |

1.7B의 `qwen3-1.7b__bf16/reference.json`은 원본 BF16 모델로 생성한 결과다. INT2 실험 폴더의 `reference.json`은 양자화된 가중치를 BF16으로 변환한 결과이므로 구분해야 한다. 이 진단만으로 INT2 정확도 저하의 원인을 확정하지는 못했다.
