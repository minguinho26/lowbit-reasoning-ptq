"""vLLM 엔진과 샘플링 설정. 평가, calibration trace 생성, 속도 측정이 모두 이 설정을 쓴다.

원본과 양자화 모델의 조건을 같게 맞추기 위한 고정 사항:
  - HF config와 HF tokenizer 경로로만 로드(tokenizer_mode=hf, config_format=hf)
  - 모델 저장소의 generation_config 무시(generation_config=vllm). 디코딩은 설정 파일 값만 사용
  - 멀티모달 모델은 텍스트 전용으로 로드(language_model_only)
  - special token을 출력 텍스트에 남김. Ministral의 [THINK]/[/THINK]가 special token이라 지우면 답 부분을 못 자름
"""


def build_llm(cfg, model_dir, mcfg, **overrides):
    from vllm import LLM

    v, d = cfg["vllm"], cfg["decoding"]
    kw = dict(
        model=str(model_dir),
        tokenizer_mode="hf",
        config_format="hf",
        generation_config="vllm",
        max_model_len=d["max_tokens"] + v["prompt_budget"],
        gpu_memory_utilization=v["gpu_memory_utilization"],
        max_num_seqs=v["max_num_seqs"],
        enable_prefix_caching=v["enable_prefix_caching"],
        seed=0,
    )
    if mcfg.get("multimodal"):
        kw["language_model_only"] = True
    kw.update(overrides)
    return LLM(**kw)


def sampling_params(cfg, seed):
    from vllm import SamplingParams

    d = cfg["decoding"]
    return SamplingParams(
        n=1,
        temperature=d["temperature"],
        top_p=d["top_p"],
        top_k=d["top_k"],
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        max_tokens=d["max_tokens"],
        seed=seed,
        skip_special_tokens=False,
    )


def gpu_info():
    import subprocess

    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"
