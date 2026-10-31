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
- Lee et al., ReSET (arXiv 2606.13233): R1-Distill-Qwen-7B/14B, Qwen3-8B/14B/32B를 NVFP4 W4A4(KV BF16, ModelOpt, calibration C4 32개 고정)로 AIME-120, GPQA-D, LiveCodeBench에서 평가함. Qwen3-8B AIME-120 기준 BF16 70.4, RTN 62.5, MR-GPTQ 65.2. calibration 축은 다루지 않음.

이 프로젝트가 추가하는 것: NVFP4 W4A4와 INT2 W2A16에서 알고리즘 3개 × calibration 2개를 교차시켜 두 요인의 효과를 분리하고, 그 답이 포맷, 모델 크기, 학습 레시피(Qwen3 자체 학습 vs R1 증류)에 따라 달라지는지 본다. 길이와 실측 속도도 같이 본다. 모델은 2504.04823, ReSET과 겹치게 골라 공개 수치로 재현을 확인할 수 있게 한다.

## 3. 모델

| 계열 | 소형 | 중형 | base / post-training |
|---|---|---|---|
| Qwen3 | Qwen/Qwen3-1.7B | Qwen/Qwen3-8B | Qwen3 base / 자체 post-training(RL 포함) |
| R1-Distill | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | Qwen2.5-Math base / DeepSeek-R1 생성 데이터로 SFT 증류 |

선택: DeepSeek-R1-Distill-Llama-8B를 넣으면 레시피는 같고 base만 다른 대조군이 된다(설정 파일에 주석으로 둠).

모델 선택 근거:
- 모두 32K 예산 안에서 평가된 기록이 있는 모델이다. Qwen3.5-4B와 Ministral-3-3B는 원본부터 AIME 30~43%가 32K에서 잘려서 제외했다(2026-10-31 확인).
- 공개 수치가 있다: Qwen3-8B, R1-Distill-Qwen-1.5B/7B는 2504.04823, Qwen3-8B와 R1-Distill-Qwen-7B는 ReSET.
- 모두 텍스트 전용 일반 dense transformer라 양자화 대상 층이 모델 간에 같다.

추가 검증 실행: Qwen3-8B GPTQ W4A16을 2504.04823 Table 5(AIME-120 66.9, GPQA-D 59.4)와 비교한다. Qwen3-8B BF16은 같은 표(68.6, 60.6)와 ReSET(70.4, 54.4)과 비교한다.

모델 공통 규칙:
- lm_head는 양자화하지 않는다. 임베딩과 norm도 그대로 둔다.
- 모델별로 실제로 양자화된 파라미터 비율을 기록해 결과 표에 같이 적는다.
- KV cache는 모든 설정에서 BF16으로 둔다(W4A4KV16, W2A16KV16). 양자화 대상은 Linear 층의 가중치와 입력뿐이다. 2504.04823의 4비트 weight-activation 결과는 KV까지 4비트(W4A4KV4)라 직접 비교하지 않는다. ReSET은 같은 W4A4KV16이다.

## 4. 포맷 × 알고리즘 × calibration

| 포맷 | 정의 | 알고리즘 | calibration | 모델당 설정 수 |
|---|---|---|---|---|
| NVFP4 W4A4 | FP4 E2M1, group 16, FP8 scale. activation도 FP4 | RTN, GPTQ, AutoRound | WikiText2, reasoning trace | 6 |
| INT2 W2A16 | INT2 대칭, group 128, weight-only | RTN(1회), GPTQ, AutoRound | GPTQ/AutoRound만 2종 | 5 |
| BF16 | 원본 | - | - | 1 |

- NVFP4 RTN도 activation global scale을 calibration으로 정하므로 calibration 축에 포함한다.
- INT2 RTN은 calibration을 쓰지 않으므로 1회만 돌린다. 데이터를 쓰지 않는 기준선 역할이다.
- 알고리즘을 데이터 의존도 순으로 놓으면 RTN < GPTQ(Hessian 기반 오차 보정) < AutoRound(블록 출력 재구성으로 rounding, clipping 학습)이다.
- 합계: 4개 모델 × 12개 = 48개 설정. 검증용 Qwen3-8B GPTQ W4A16 1개를 더해 49개.

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
- thinking 활성화는 각 모델의 공식 chat template을 따른다. Qwen3는 enable_thinking=True, R1-Distill은 템플릿이 생성 프롬프트에 <think>를 넣는다. 시스템 프롬프트는 쓰지 않는다.
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

실측(RTX 5090, AIME 30문제, 원본): Qwen3-8B 평균 출력 1.78만 토큰, truncation 13%, 334 tok/s(30개 동시). Qwen3.5-4B NVFP4 10문제 965 tok/s.

- 8B 원본은 KV 공간이 부족해 처리량이 낮고, 양자화 모델과 1.5~1.7B 모델은 더 빠르다.
- 출력 토큰 총량은 대략 4~5억 토큰으로 본다(AIME 평균 1.5~2만 토큰 기준). 5090 기준 GPU 150~200시간, $180~250 수준.
- 1주차에 Qwen3-1.7B와 R1-Distill-Qwen-1.5B 원본 truncation과 처리량을 보고 다시 계산한다. RTX PRO 6000이 8B 원본에서 토큰당 더 싼지도 같이 확인한다.

## 10. 일정

1주차: 환경 구성과 확인
- 완료: NVFP4 W4A4가 5090에서 native FP4 GEMM(CutlassNvFp4LinearKernel)으로 도는 것을 확인함. calibration을 쓰는 양자화 경로 확인함.
- 남은 것: W2A16 로드와 커널, GPTQ와 AutoRound가 새 모델에서 끝까지 도는지, BF16 원본 4개의 truncation과 처리량, Qwen3-8B 공개 수치 재현.
- INT2 설정 하나를 먼저 돌려 출력 길이 확인.

2주차: calibration trace 생성, 48개 설정 양자화

3주차: 평가와 속도 측정

4주차: 분석, HF 업로드(모델카드에 recipe, calibration, 원본 대비 표, 서빙 명령), 기술 리포트

## 11. 범위와 한계

- 계열이 2개라 계열 효과를 일반화할 수 없다. 두 계열은 base 세대(Qwen3 vs Qwen2.5)와 학습 레시피(RL 포함 vs SFT 증류)가 동시에 다르다. R1-Distill-Llama-8B를 넣으면 레시피를 고정한 base 비교가 하나 생긴다.
- R1-Distill-Qwen-1.5B는 원본 AIME 정확도부터 낮아(2504.04823 기준 23%) INT2에서 바닥 효과가 예상된다.
- NVFP4 W4A4와 INT2 W2A16은 bit 수, 데이터 타입, activation 양자화 여부가 동시에 다르다. bit-width 효과로 해석하지 않고 실무에서 쓰이는 두 저비트 설정의 비교로 쓴다.
- GPQA는 단일 seed다.
- 모델별 권장 샘플링을 쓰지 않으므로 원본 정확도가 모델카드 수치보다 낮을 수 있다.
- 원본도 32K에서 일부 잘린다(Qwen3-8B AIME 약 13%). 정확도는 32K 예산 기준이다.

## 참고

- 2504.04823 코드: https://github.com/ruikangliu/Quantized-Reasoning-Models
- llm-compressor: https://github.com/vllm-project/llm-compressor
