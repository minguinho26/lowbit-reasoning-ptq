# 리즈닝 모델 저비트 PTQ: 양자화 알고리즘과 calibration 데이터의 영향

리즈닝 모델 저비트 PTQ 실험 계획. 작성일 2026-10-29.

## 1. 질문

리즈닝 모델의 post-training quantization에서 원본(BF16) 대비 성능 저하를 더 크게 좌우하는 것은 양자화 알고리즘인가, calibration 데이터인가. 그 답은 포맷(NVFP4 W4A4, INT2 W2A16)과 모델 계열에 따라 달라지는가.

세부 질문:
1. 데이터 의존도가 높은 알고리즘(GPTQ, AutoRound)일수록 calibration 도메인에 더 민감한가.
2. calibration 도메인이 맞지 않으면 데이터를 쓰는 알고리즘이 RTN보다 나빠지는가.
3. 저비트에서 출력이 길어지는가(길이, truncation, 루프). 길어진다면 토큰당 속도 향상이 얼마나 상쇄되는가.

## 2. 선행 연구와의 관계

- Liu et al., Quantization Hurts Reasoning? (COLM 2025, arXiv 2504.04823): R1-Distill 계열 1.5B에서 calibration 도메인(WikiText2 vs 자체 생성 Numina-Math trace)이 GPTQ에만 크게 영향을 준다고 보고함(W3, 평균 +9.8pt). AWQ, SmoothQuant, FlatQuant는 1pt 안팎. W4 수준에서는 출력이 길어지지 않는다고 보고함.
- Egiazarian et al., MR-GPTQ (ICLR 2026, arXiv 2509.23202): NVFP4는 group 16이라 RTN이 이미 강하고, 기존 기법이 RTN을 항상 이기지는 못한다고 보고함. Qwen3-8B W4A4에서 RTN 98.9%, GPTQ 98.1% recovery(비리즈닝 과제).
- Alimaskina et al., arXiv 2606.02011: 2-bit 리즈닝 모델의 정확도 하락이 루프, 예산 소진 같은 생성 과정 실패와 연결됨을 보고함.

이 프로젝트가 추가하는 것: 위 결과를 NVFP4 W4A4와 INT2 W2A16, 최신 두 계열(Qwen3.5, Ministral 3), 알고리즘 3개 × calibration 2개의 교차 설계로 확장하고, 길이와 실측 속도까지 같이 본다.

## 3. 모델

| 계열 | ~4B | ~8B | 구조 |
|---|---|---|---|
| Qwen3.5 | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | 하이브리드(Gated DeltaNet 3층 + Gated Attention 1층 반복), vision encoder 포함 |
| Ministral 3 Reasoning | mistralai/Ministral-3-3B-Reasoning-2512 | mistralai/Ministral-3-8B-Reasoning-2512 | 일반 dense transformer, vision encoder 포함 |

파이프라인 검증용: Qwen/Qwen3-8B (BF16, GPTQ W4A16 두 개만). 2504.04823 Table 5의 BF16 수치(AIME-120 68.6, GPQA-D 60.6)와 비교해서 평가 파이프라인이 공개 수치를 재현하는지 확인한다. 본 분석에는 넣지 않는다.

모델 공통 규칙:
- vision/audio 부분과 lm_head는 양자화하지 않는다. 평가는 텍스트 전용으로 띄운다(vLLM language_model_only).
- Qwen3.5의 Gated DeltaNet 투영(linear_attn)은 microscale 포맷과 호환되지 않아 llm-compressor 예제에서도 제외한다. 공정한 비교를 위해 모든 포맷에서 같은 층을 제외한다.
- 모델별로 실제로 양자화된 파라미터 비율을 기록해 결과 표에 같이 적는다.
- KV cache는 모든 설정에서 BF16으로 둔다(W4A4KV16, W2A16KV16). 양자화 대상은 Linear 층의 가중치와 입력뿐이다. 2504.04823의 4비트 weight-activation 결과는 KV까지 4비트(W4A4KV4)라 직접 비교하지 않는다.

## 4. 포맷 × 알고리즘 × calibration

| 포맷 | 정의 | 알고리즘 | calibration | 모델당 설정 수 |
|---|---|---|---|---|
| NVFP4 W4A4 | FP4 E2M1, group 16, FP8 scale. activation도 FP4 | RTN, GPTQ, AutoRound | WikiText2, reasoning trace | 6 |
| INT2 W2A16 | INT2 대칭, group 128, weight-only | RTN(1회), GPTQ, AutoRound | GPTQ/AutoRound만 2종 | 5 |
| BF16 | 원본 | - | - | 1 |

- NVFP4 RTN도 activation global scale을 calibration으로 정하므로 calibration 축에 포함한다.
- INT2 RTN은 calibration을 쓰지 않으므로 1회만 돌린다. 데이터를 쓰지 않는 기준선 역할이다.
- 알고리즘을 데이터 의존도 순으로 놓으면 RTN < GPTQ(Hessian 기반 오차 보정) < AutoRound(블록 출력 재구성으로 rounding, clipping 학습)이다.
- 합계: 4개 모델 × 12개 = 48개 설정. 검증용 Qwen3-8B 2개를 더해 50개.

calibration:
- WikiText2: wikitext-2-raw-v1 train 텍스트를 이어 붙여 2048토큰 윈도우 128개를 무작위 추출(2504.04823과 동일).
- reasoning trace: NuminaMath-1.5 문제 256개에 대해 각 원본 모델이 평가와 같은 디코딩 설정으로 직접 생성한 응답. 프롬프트와 응답을 이어 붙여 2048토큰 윈도우 128개를 무작위 추출(2504.04823과 동일).
- 오염 방지: NuminaMath-1.5에서 source가 amc_aime인 문제를 제외하고, AIME-120, GPQA-Diamond 문항과 13-gram이 겹치는 문제를 제외한다.

알고리즘 기본값: GPTQ(block 128, dampening 0.01, act-order static), AutoRound(iters 200, batch 8). llm-compressor 기본값을 그대로 쓴다.

## 5. 벤치마크와 디코딩

| 유형 | 벤치 | 문제 수 | seed |
|---|---|---|---|
| 수학 | AIME-120 (AIME 2022~2024 90문제 + AIME 2025 30문제) | 120 | 3 |
| 과학 | GPQA-Diamond | 198 | 1 |

- 프롬프트와 데이터셋은 2504.04823 코드와 같다. AIME: 문제 뒤에 "Please reason step by step, and put your final answer within \boxed{}." GPQA: simple-evals 템플릿, 선택지 순서는 고정 seed로 한 번만 섞어 모든 설정에서 동일하게 쓴다.
- 디코딩은 전 모델 동일: temperature 0.6, top-p 0.95, top-k 끔, penalty 없음, 최대 32,768 토큰. 모델 저장소의 generation_config는 적용하지 않는다.
- thinking 활성화는 각 모델의 공식 chat template을 따른다. Qwen3.5는 enable_thinking=True, Ministral은 템플릿에 들어 있는 기본 시스템 프롬프트([THINK] 형식).
- LiveCodeBench는 비용 때문에 제외했다. 설정 파일에서 추가할 수 있게 해둔다.

## 6. 지표

| 지표 | 정의 |
|---|---|
| 정확도 | 벤치별 pass@1. AIME는 seed 3개 평균 |
| 원본 대비 저하 폭 | 양자화 모델 정확도 − 같은 모델 BF16 정확도 |
| paired 길이 비율 | 같은 문제에서 양자화 모델 출력 토큰 수 / 원본 출력 토큰 수. 완료 샘플 기준, 둘 다 맞힌 문제 기준 두 가지 |
| 길이 분포 | 출력 토큰 수 중앙값, p90 |
| truncation rate | 32,768토큰 한도에 걸려 끝나지 못한 샘플 비율(오답 처리) |
| 루프 비율 | 마지막 2048토큰에서 32-gram 고유 비율이 0.5 미만인 샘플 비율(휴리스틱) |
| 토큰당 decode 속도 | 고정 입력/출력 길이에서 원본 대비 처리량 비율. 같은 pod에서 연달아 측정 |
| effective speedup | 토큰당 속도 향상 ÷ paired 길이 비율 |
| 체크포인트 크기 | safetensors 합계 |

## 7. 통계 처리

- 모든 비교(원본 대비, 설정끼리)는 문제 단위 paired bootstrap(10,000회)으로 차이의 95% 신뢰구간을 구한다. 구간이 0을 포함하면 "구분되지 않음"으로 판정한다.
- AIME는 문제별로 seed 3개 평균 정답률을 쓴 뒤 bootstrap한다. GPQA는 단일 seed라 seed 간 변동을 따로 분리하지 못한다는 점을 리포트에 적는다.
- 요인 효과: 모델 × 포맷별로, calibration을 고정하고 알고리즘을 바꿨을 때의 범위와 알고리즘을 고정하고 calibration을 바꿨을 때의 범위를 비교한다. 각 범위에 bootstrap 구간을 같이 적는다.

## 8. 하드웨어와 소프트웨어

- GPU: RunPod RTX 5090(32GB, sm_120) 한 종류로 양자화, 평가, 속도 측정을 모두 한다.
- 소프트웨어: vllm 0.32.0, llmcompressor 0.14.0.1(auto-round 포함), transformers 5.17.0. 양자화 환경과 평가 환경을 분리한다.
- 저장소: Network Volume(/workspace)에 HF cache, 체크포인트, 결과를 둔다. HF 업로드 후 로컬 체크포인트는 지운다.

## 9. 비용 추정

- 평가 출력 토큰: 설정당 AIME 3 seed(약 470만) + GPQA 1 seed(약 160만). INT2는 출력이 1.8배라고 가정. 합계 약 4.1억 토큰.
- 5090 토큰당 비용(메모리 대역폭 기준 이론치): 양자화 모델 약 $0.23/100만 토큰, BF16 약 $0.58/100만 토큰. 실제 vLLM은 이보다 느리므로 1.5~2배를 감안한다.
- 평가 약 $100~210, 양자화와 trace 생성과 smoke test 약 $30~50, 스토리지 월 $10~20. 합계 약 $150~280.
- 처리량과 INT2 출력 길이는 가정이다. 1주차 실측 후 다시 계산한다.

## 10. 일정

1주차: 환경 구성과 확인
- 5090에서 확인할 것: NVFP4 W4A4가 native FP4 GEMM 커널로 잡히는지(vLLM 로그의 "Using ... for NVFP4 GEMM"), W2A16이 Humming 커널로 도는지, Qwen3.5 하이브리드와 Ministral 3에서 GPTQ와 AutoRound가 끝까지 도는지.
- BF16 원본 4개와 Qwen3-8B 검증 실행. 실측 tok/s와 평균 출력 길이로 예산 재계산.
- INT2 설정 하나를 먼저 돌려 출력 길이 확인.

2주차: calibration trace 생성, 48개 설정 양자화

3주차: 평가와 속도 측정

4주차: 분석, HF 업로드(모델카드에 recipe, calibration, 원본 대비 표, 서빙 명령), 기술 리포트

## 11. 범위와 한계

- 계열이 2개라 계열 효과를 일반화할 수 없다. 두 계열은 구조도 달라서(하이브리드 vs 일반 attention) 계열 차이와 구조 차이를 분리할 수 없다.
- NVFP4 W4A4와 INT2 W2A16은 bit 수, 데이터 타입, activation 양자화 여부가 동시에 다르다. bit-width 효과로 해석하지 않고 실무에서 쓰이는 두 저비트 설정의 비교로 쓴다.
- GPQA는 단일 seed다.
- 모델별 권장 샘플링을 쓰지 않으므로 원본 정확도가 모델카드 수치보다 낮을 수 있다.

## 참고

- 2504.04823 코드: https://github.com/ruikangliu/Quantized-Reasoning-Models
- llm-compressor: https://github.com/vllm-project/llm-compressor
