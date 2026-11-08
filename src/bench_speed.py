"""토큰당 decode 속도 측정. 고정 입력/출력 길이(ignore_eos), 배치 크기별 처리량.

원본과 양자화 모델은 반드시 같은 pod에서 연달아 측정할 것(호스트마다 클럭, 전력 제한이 다를 수 있음).
run_matrix.py speed --model <m> 이 원본부터 순서대로 돌린다.

예: python src/bench_speed.py --run qwen3-8b__bf16
"""

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone

from common import base_model_dir, load_config, parse_run_name, run_model_dir
from engine import build_llm, gpu_info


def checkpoint_bytes(path):
    return sum(f.stat().st_size for f in path.glob("*.safetensors"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--config")
    args = p.parse_args()

    cfg = load_config(args.config)
    s = cfg["speed"]
    r = parse_run_name(args.run)
    model_dir = base_model_dir(cfg, r["model"]) if r["fmt"] == "bf16" else run_model_dir(cfg, args.run)
    out = cfg["_dirs"]["results"] / "speed" / f"{args.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    import vllm
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    llm = build_llm(cfg, model_dir, max_model_len=s["input_len"] + s["output_len"] + 16,
                    max_num_seqs=max(s["batch_sizes"]))
    sp = SamplingParams(temperature=0.0, max_tokens=s["output_len"], ignore_eos=True)
    rng = random.Random(0)

    def prompts(bs):
        return [TokensPrompt(prompt_token_ids=[rng.randint(1000, 20000) for _ in range(s["input_len"])]) for _ in range(bs)]

    llm.generate(prompts(2), SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True), use_tqdm=False)  # warmup

    results = []
    for bs in s["batch_sizes"]:
        times = []
        for _ in range(s["repeats"]):
            t0 = time.time()
            llm.generate(prompts(bs), sp, use_tqdm=False)
            times.append(time.time() - t0)
        t = sorted(times)[len(times) // 2]
        results.append({
            "batch_size": bs,
            "seconds_median": t,
            "seconds_all": times,
            "output_tok_s": bs * s["output_len"] / t,
            "per_seq_tok_s": s["output_len"] / t,
        })
        print(results[-1])

    rec = {
        "run": args.run,
        "input_len": s["input_len"],
        "output_len": s["output_len"],
        "results": results,
        "checkpoint_bytes": checkpoint_bytes(model_dir),
        "gpu": gpu_info(),
        "vllm": vllm.__version__,
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out.write_text(json.dumps(rec, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
