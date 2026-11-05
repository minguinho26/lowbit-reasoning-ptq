"""INT2 체크포인트의 BF16 복원 경로와 vLLM 압축 커널을 같은 입력으로 비교한다.

본 평가를 바꾸지 않는 짧은 진단이다. baseline/reference와 kernel은 별도 프로세스로
실행해 GPU 메모리를 공유하지 않는다. 일치는 체크포인트 해석 경로의 일관성만 보여주며
양자화 알고리즘 자체의 정확성이나 전체 벤치 성능을 입증하지 않는다.
"""

import argparse
import json
import os
from pathlib import Path

from common import base_model_dir, load_config, model_cfg, parse_run_name, run_model_dir


PROMPTS = [
    "Calculate 13 times 17. Give the integer answer.",
    "All roses are flowers. Some flowers are red. Must all roses be red? Explain briefly.",
    "Solve for x: 3*x + 7 = 22. Show your calculation.",
]


def compare_outputs(reference, kernel):
    """각 prompt의 첫 next-token 분포와 greedy 첫 토큰을 비교한다."""
    result = []
    for ref, actual in zip(reference["samples"], kernel["samples"]):
        if ref["input_ids"] != actual["input_ids"]:
            raise ValueError("Reference/kernel token IDs differ")
        shared = sorted(set(ref["top_logprobs"]) & set(actual["top_logprobs"]))
        delta = [abs(ref["top_logprobs"][t] - actual["top_logprobs"][t]) for t in shared]
        result.append({
            "prompt": ref["prompt"],
            "first_token_agrees": ref["first_token"] == actual["first_token"],
            "top20_overlap": len(shared),
            "shared_top20_max_abs_logprob_delta": max(delta) if delta else None,
            "reference_text": ref["text"], "kernel_text": actual["text"],
        })
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--backend", required=True, choices=("reference", "kernel"))
    p.add_argument("--config")
    args = p.parse_args()
    cfg = load_config(args.config)
    run = parse_run_name(args.run)
    if run["fmt"] not in ("bf16", "w2a16"):
        raise ValueError("This diagnostic supports BF16 and W2A16 only")
    mcfg = model_cfg(cfg, run["model"])
    path = base_model_dir(cfg, run["model"]) if run["fmt"] == "bf16" else run_model_dir(cfg, args.run)
    out = cfg["_dirs"]["results"] / "diagnostics" / args.run
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{args.backend}.json"
    # 완료된 짧은 진단은 같은 Pod에서만 재사용한다. 새 GPU에서는 다시 검사한다.
    pod = os.environ.get("RUNPOD_POD_ID")
    if pod and target.exists() and json.loads(target.read_text()).get("pod_id") == pod:
        print(f"already diagnosed on this pod: {target}")
        return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir(cfg, run["model"]), local_files_only=True)
    inputs = [tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
        **(mcfg.get("chat_template_kwargs") or {}),
    ) for text in PROMPTS]
    samples = []
    if args.backend == "reference":
        import torch
        from transformers import AutoModelForCausalLM, CompressedTensorsConfig, GenerationConfig

        kw = {} if run["fmt"] == "bf16" else {"quantization_config": CompressedTensorsConfig(run_compressed=False)}
        # INT2를 BF16 tensor로 복원해 Humming을 거치지 않는 Transformers forward를 쓴다.
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation="eager", local_files_only=True, **kw,
        ).eval()
        with torch.inference_mode():
            for prompt, ids in zip(PROMPTS, inputs):
                ids_tensor = torch.tensor([ids], device="cuda")
                logits = model(input_ids=ids_tensor).logits[0, -1].float()
                logp = logits.log_softmax(-1)
                values, tokens = logp.topk(20)
                generated = model.generate(
                    input_ids=ids_tensor, attention_mask=torch.ones_like(ids_tensor),
                    generation_config=GenerationConfig(
                        do_sample=False, max_new_tokens=48, use_cache=True,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
                    ),
                )[0, len(ids):].tolist()
                samples.append({
                    "prompt": prompt, "input_ids": ids, "finite_logits": bool(torch.isfinite(logits).all()),
                    "first_token": int(logp.argmax()),
                    "top_logprobs": {str(t): float(v) for t, v in zip(tokens.tolist(), values.tolist())},
                    "text": tokenizer.decode(generated, skip_special_tokens=False),
                })
    else:
        from engine import build_llm
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        # 본 평가와 같은 기본 backend를 사용한다. 짧은 입력에 맞춰 메모리 크기만 제한한다.
        llm = build_llm(cfg, path, mcfg, max_model_len=512, max_num_seqs=1)
        for prompt, ids in zip(PROMPTS, inputs):
            response = llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(
                temperature=0.0, max_tokens=48, logprobs=20, skip_special_tokens=False,
            ), use_tqdm=False)[0].outputs[0]
            samples.append({
                "prompt": prompt, "input_ids": ids, "first_token": int(response.token_ids[0]),
                "top_logprobs": {str(t): float(v.logprob) for t, v in response.logprobs[0].items()},
                "text": response.text,
            })

    from engine import gpu_info
    result = {"run": args.run, "backend": args.backend, "pod_id": pod,
              "gpu": gpu_info(), "max_new_tokens": 48, "samples": samples}
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    reference = out / "reference.json"
    if args.backend == "kernel" and reference.exists():
        ref = json.loads(reference.read_text())
        if ref.get("pod_id") != pod:
            raise ValueError("Reference was generated on a different Pod")
        comparison = {"run": args.run, "pod_id": pod, "samples": compare_outputs(ref, result)}
        (out / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False))
    print(f"saved -> {target}", flush=True)


if __name__ == "__main__":
    main()
