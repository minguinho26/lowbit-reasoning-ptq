# 리즈닝 모델 저비트 PTQ: 양자화 알고리즘과 calibration 데이터의 영향

리즈닝 모델 저비트 PTQ 실험 계획. 작성일 2026-10-29.

## 1. 질문

리즈닝 모델의 post-training quantization에서 원본(BF16) 대비 성능 저하를 더 크게 좌우하는 것은 양자화 알고리즘인가, calibration 데이터인가. 그 답은 모델 크기(Qwen3 1.7B, 4B, 8B)에 따라, 그리고 activation 양자화 여부와 가중치 포맷에 따라 달라지는가. 포맷은 한 번에 요인 하나만 바뀌도록 BF16, NVFP4 W4A16, NVFP4 W4A4, INT2 W2A16 네 설정으로 구성한다(4절).

세부 질문:
1. 데이터 의존도가 높은 알고리즘(GPTQ, AutoRound)일수록 calibration 도메인에 더 민감한가.
2. calibration 도메인이 맞지 않으면 데이터를 쓰는 알고리즘이 RTN보다 나빠지는가.
3. 저비트에서 출력이 길어지는가(길이, truncation, 루프). 길어진다면 토큰당 속도 향상이 얼마나 상쇄되는가.

## 2. 선행 연구와의 관계

- Liu et al., Quantization Hurts Reasoning? (COLM 2025, arXiv 2504.04823): R1-Distill 계열 1.5B에서 calibration 도메인(WikiText2 vs 자체 생성 Numina-Math trace)이 GPTQ에만 크게 영향을 준다고 보고함(W3, 평균 +9.8pt). AWQ, SmoothQuant, FlatQuant는 1pt 안팎. W4 수준에서는 출력이 길어지지 않는다고 보고함.
- Egiazarian et al., MR-GPTQ (ICLR 2026, arXiv 2509.23202): NVFP4는 group 16이라 RTN이 이미 강하고, 기존 기법이 RTN을 항상 이기지는 못한다고 보고함. Qwen3-8B W4A4에서 RTN 98.9%, GPTQ 98.1% recovery(비리즈닝 과제).
- Alimaskina et al., arXiv 2606.02011: 2-bit 리즈닝 모델의 정확도 하락이 루프, 예산 소진 같은 생성 과정 실패와 연결됨을 보고함.
- Lee et al., ReSET (arXiv 2606.13233): R1-Distill-Qwen-7B/14B, Qwen3-8B/14B/32B를 NVFP4 W4A4(KV BF16, ModelOpt, calibration C4 32개 고정)로 AIME-120, GPQA-D, LiveCodeBench에서 평가함. Qwen3-8B AIME-120 기준 BF16 70.4, RTN 62.5, MR-GPTQ 65.2. calibration 축은 다루지 않음.

이 프로젝트가 추가하는 것: NVFP4 W4A4와 INT2 W2A16에서 알고리즘 3개 × calibration 2개를 교차시켜 두 요인의 효과를 분리하고, 그 답이 같은 학습 레시피 안에서 모델 크기에 따라 달라지는지 본다. 길이와 실측 속도도 같이 본다. Qwen3-8B는 2504.04823, ReSET과 겹쳐 공개 수치로 재현을 확인할 수 있다.

리즈닝 모델 양자화 연구의 모델 구성: 2504.04823은 R1-Distill-Qwen 1.5B~32B, R1-Distill-Llama 8B/70B, QwQ-32B, ReSET은 R1-Distill-Qwen 7B/14B와 Qwen3 8B/14B/32B, 2606.02011은 Qwen3 8B/32B만 쓴다. 대부분 Qwen 아키텍처이고, 단일 계열 구성도 선례가 있다.

## 3. 모델

| 모델 | 층 | KV head | base / post-training |
|---|---|---|---|
| Qwen/Qwen3-1.7B | 28 | 8 | Qwen3 base / 자체 post-training(RL 포함), thinking 모드 |
| Qwen/Qwen3-4B | 36 | 8 | 같음 |
| Qwen/Qwen3-8B | 36 | 8 | 같음 |

모델 선택 근거:
- 학습 레시피를 고정하고 크기만 바꾼다. 세 모델 모두 같은 Qwen3 post-training 파이프라인에서 나왔다.
- Qwen3.5-4B와 Ministral-3-3B는 원본부터 AIME 30~43%가 32K에서 잘려서 제외했다(2026-10-31 확인). R1-Distill-Qwen 1.5B/7B는 1주차 확인에 썼지만 크기 축에 집중하기 위해 본 실험에서 뺐다(2026-11-01). 1.5B는 원본 AIME 30문제 정확도가 13%로 바닥에 가까웠다.
- Qwen3-8B는 공개 수치가 있다(2504.04823, ReSET).

추가 검증 실행: Qwen3-8B GPTQ W4A16을 2504.04823 Table 5(AIME-120 66.9, GPQA-D 59.4)와 비교한다. Qwen3-8B BF16은 같은 표(68.6, 60.6)와 ReSET(70.4, 54.4)과 비교한다.

모델 공통 규칙:
- lm_head는 양자화하지 않는다. 임베딩과 norm도 그대로 둔다.
- 모델별로 실제로 양자화된 파라미터 비율을 기록해 결과 표에 같이 적는다.
- KV cache는 모든 설정에서 BF16으로 둔다(W4A4KV16, W2A16KV16). 양자화 대상은 Linear 층의 가중치와 입력뿐이다. 2504.04823의 4비트 weight-activation 결과는 KV까지 4비트(W4A4KV4)라 직접 비교하지 않는다. ReSET은 같은 W4A4KV16이다.

## 4. 포맷 × 알고리즘 × calibration

포맷은 이웃한 설정끼리 요인 하나만 다르게 둔다.

```
BF16 --(가중치 NVFP4)--> NVFP4 W4A16 --(activation FP4)--> NVFP4 W4A4
                              |
                       (가중치 포맷 INT2)
                              v
                         INT2 W2A16
```

| 비교 | 바뀌는 요인 | 고정되는 것 |
|---|---|---|
| NVFP4 W4A16 vs NVFP4 W4A4 | activation 양자화(16 -> FP4) | 가중치 포맷(NVFP4). RTN은 가중치가 완전히 같다 |
| NVFP4 W4A16 vs INT2 W2A16 | 가중치 포맷(NVFP4 -> INT2) | activation 16비트 |
| BF16 vs NVFP4 W4A16 | 가중치 양자화 | activation 16비트 |

가중치 포맷은 원소 타입, bit, group, scale 방식이 묶인 각 포맷의 표준 정의 그대로 하나의 요인으로 본다. 이 요인 안의 구성(bit 대 group 등)은 분리하지 않는다. group을 맞추면 실제 비트 수가 달라지고(INT2 g16은 가중치당 3비트), 둘을 다 맞추면 존재하지 않는 포맷이 된다. MR-GPTQ도 INT4, NVFP4, MXFP4를 각 규격 그대로 비교한다.

| 포맷 | 원소 | group / scale | 가중치당 실제 비트 | activation | 알고리즘 | calibration | 모델당 설정 수 |
|---|---|---|---|---|---|---|---|
| NVFP4 W4A16 | FP4 E2M1 | 16 / FP8 E4M3 + FP32 global | 4.5 | BF16 | RTN(1회), GPTQ, AutoRound | GPTQ/AutoRound만 2종 | 5 |
| NVFP4 W4A4 | FP4 E2M1 | 16 / FP8 E4M3 + FP32 global | 4.5 | NVFP4 | RTN, GPTQ, AutoRound | 3개 모두 2종 | 6 |
| INT2 W2A16 | INT2 대칭 | 128 / BF16 | 2.125 | BF16 | RTN(1회), GPTQ, AutoRound | GPTQ/AutoRound만 2종 | 5 |
| BF16 | 원본 | - | 16 | BF16 | - | - | 1 |

- NVFP4 W4A4 RTN은 activation global scale을 calibration으로 정하므로 calibration 축에 포함한다. weight-only RTN(NVFP4 W4A16, INT2)은 데이터를 쓰지 않으므로 1회만 돌린다.
- 알고리즘을 데이터 의존도 순으로 놓으면 RTN < GPTQ(Hessian 기반 오차 보정) < AutoRound(블록 출력 재구성으로 rounding, clipping 학습)이다.
- 합계: 3개 모델 × 17개 = 51개 설정. 검증용 Qwen3-8B INT4 GPTQ W4A16 1개를 더해 52개.
- INT2 W2A4(INT2 가중치 × FP4 activation)는 넣지 않는다. 커널이 없어 fake quantization으로만 돌릴 수 있고, INT2 W2A16에서 이미 RTN은 붕괴하고 AutoRound는 루프에 빠져(1주차, 1.5B) 바닥 효과가 예상된다.

calibration:
- WikiText2: wikitext-2-raw-v1 train 텍스트를 이어 붙여 2048토큰 윈도우 128개를 무작위 추출(2504.04823과 동일).
- reasoning trace: NuminaMath-1.5 문제 256개에 대해 각 원본 모델이 평가와 같은 디코딩 설정으로 직접 생성한 응답. 프롬프트와 응답을 이어 붙여 2048토큰 윈도우 128개를 무작위 추출(2504.04823과 동일).
- 오염 방지: NuminaMath-1.5에서 source가 amc_aime인 문제를 제외하고, AIME-120, GPQA-Diamond 문항과 13-gram이 겹치는 문제를 제외한다.

알고리즘 기본값: GPTQ(block 128, dampening 0.01, act-order static), AutoRound(iters 200, batch 8). llm-compressor 기본값을 그대로 쓴다.

## 5. 벤치마크와 디코딩

| 유형 | 벤치 | 문제 수 | seed |
|---|---|---|---|
| 수학 | AIME-120 (AIME 2022~2024 90문제 + AIME 2025 30문제) | 120 | 3 |
| 과학 | GPQA-Diamond | 198 | 3 |

- 프롬프트와 데이터셋은 2504.04823 코드와 같다. AIME: 문제 뒤에 "Please reason step by step, and put your final answer within \boxed{}." GPQA: simple-evals 템플릿, 선택지 순서는 고정 seed로 한 번만 섞어 모든 설정에서 동일하게 쓴다.
- 디코딩은 전 모델 동일: temperature 0.6, top-p 0.95, top-k 끔, penalty 없음, 최대 32,768 토큰. 모델 저장소의 generation_config는 적용하지 않는다.
- thinking 활성화는 공식 chat template의 enable_thinking=True를 쓴다. 시스템 프롬프트는 쓰지 않는다.
- LiveCodeBench는 비용 때문에 제외했다. 설정 파일에서 추가할 수 있게 해둔다.

## 6. 지표

| 지표 | 정의 |
|---|---|
| 정확도 | 벤치별 pass@1. seed 3개 평균 |
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
- 문제별로 seed 3개 평균 정답률을 쓴 뒤 bootstrap한다. seed 수는 2504.04823과 같다.
- 요인 효과: 모델 × 포맷별로, calibration을 고정하고 알고리즘을 바꿨을 때의 범위와 알고리즘을 고정하고 calibration을 바꿨을 때의 범위를 비교한다. 각 범위에 bootstrap 구간을 같이 적는다.
- 포맷 요인: 4절의 포맷 쌍(activation, 가중치 포맷)마다 같은 알고리즘, 같은 calibration끼리 짝지어 차이와 구간을 구한다.

## 8. 하드웨어와 소프트웨어

- GPU: RunPod RTX 5090(32GB, sm_120) 한 종류로 양자화, 평가, 속도 측정을 모두 한다.
- 소프트웨어: vllm 0.32.0, llmcompressor 0.14.0.1(auto-round 포함), transformers 5.17.0. 양자화 환경과 평가 환경을 분리한다.
- 저장소: Network Volume(/workspace)에 HF cache, 체크포인트, 결과를 둔다. HF 업로드 후 로컬 체크포인트는 지운다.

## 9. 비용 추정

실측(RTX 5090, AIME 앞 30문제, seed 0):

| 설정 | 정확도 | 평균 출력 토큰 | truncation | loop | 처리량(tok/s) | KV cache(토큰) |
|---|---|---|---|---|---|---|
| Qwen3-1.7B BF16 | 43.3% | 18,382 | 13.3% | 0% | 856 | 221,424 |
| Qwen3-8B BF16 | 73.3% | 17,775 | 13.3% | 0% | 334 | - |
| Qwen3-8B NVFP4 RTN (WikiText2) | 60.0% | 20,728 | 16.7% | 0% | 533 | 147,120 |
| R1-Distill-Qwen-1.5B BF16 (GPQA 198문제) | 33.3% | 10,393 | 1.5% | 1.5% | 5,086 | - |
| R1-Distill-Qwen-1.5B INT2 RTN | 0% | 320 | 0% | 0% | - | - |
| R1-Distill-Qwen-1.5B INT2 AutoRound trace (5문제) | 0% | 32,768 | 100% | 60% | - | - |

- 처리량은 KV cache 용량이 정한다. 평가 처리량은 속도 비교에 쓰지 않는다(속도는 bench_speed로만 비교).
- 가정: GPQA 출력 길이는 AIME의 0.53배(R1-1.5B 실측). NVFP4 출력은 원본의 1.17배(Qwen3-8B 실측). 데이터를 쓰는 INT2 설정은 모든 샘플이 32K까지 간다(최악). INT2 RTN은 짧게 붕괴해서 비용이 거의 없다. Qwen3-4B는 아직 안 재서 원본 450, 양자화 650 tok/s로 둔다.
- 평가 GPU 시간(5090, $1.19/h, AIME와 GPQA 모두 3 seed): BF16 21시간, NVFP4 W4A16 81시간, NVFP4 W4A4 109시간, INT2 W2A16 159시간, 검증용 W4A16 6시간. 합계 약 377시간, $448. INT2가 모든 샘플에서 32K까지 간다는 최악 가정이 가장 크다. Qwen3-8B INT2 AutoRound 30문제를 먼저 돌려 실제 길이로 다시 계산한다.
- 그 외(지금까지 쓴 비용 제외): 양자화 약 14시간($17), Qwen3-4B trace 생성 약 3시간($4), 속도 측정 약 3시간($4), 스토리지($10~20). 총액 약 $485.

## 10. 일정

1주차(완료): 환경 구성과 확인
- NVFP4 W4A4가 CutlassNvFp4LinearKernel로 도는 것, W2A16이 Humming 커널로 도는 것(NVRTC 라이브러리 경로 수정 후) 확인.
- RTN, GPTQ, AutoRound × NVFP4, INT2 양자화 11개 경로가 R1-Distill-Qwen-1.5B에서 끝까지 도는 것 확인.
- Qwen3-1.7B, Qwen3-8B calibration trace 생성 완료.

2주차: Qwen3-4B 원본 확인과 trace 생성, NVFP4 W4A16 커널 확인, 51개 설정 양자화

3주차: 평가와 속도 측정

4주차: 분석, HF 업로드(모델카드에 recipe, calibration, 원본 대비 표, 서빙 명령), 기술 리포트

## 11. 범위와 한계

- Qwen3 한 계열이라 다른 아키텍처나 학습 레시피(예: R1 증류)로 일반화할 수 없다.
- NVFP4와 INT2 가중치는 원소 타입, bit, group/scale 방식이 동시에 다르다(activation은 NVFP4 W4A16과 INT2 W2A16 비교에서 16비트로 고정). 가중치 포맷 요인(NVFP4 -> INT2)은 이 셋이 묶인 포맷 단위 비교이고, bit 감소 효과로 해석하지 않는다. 결과 표에 가중치당 실제 비트(4.5 vs 2.125)를 같이 적는다. bit만 다른 비교(INT4 W4A16 vs INT2 W2A16)는 이번 범위 밖이다.
- INT2 × FP4 activation 조합이 없어 가중치 포맷과 activation 양자화의 상호작용은 보지 않는다.
- 모델별 권장 샘플링을 쓰지 않으므로 원본 정확도가 모델카드 수치보다 낮을 수 있다.
- 원본도 32K에서 일부 잘린다(Qwen3-8B AIME 약 13%). 정확도는 32K 예산 기준이다.

## 참고

- 2504.04823 코드: https://github.com/ruikangliu/Quantized-Reasoning-Models
- llm-compressor: https://github.com/vllm-project/llm-compressor
