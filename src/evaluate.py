"""벤치마크 평가. 문제 단위로 jsonl에 바로 기록하므로, pod이 끊겨도 같은 명령으로 이어서 돌릴 수 있다.

기록 항목: 정오답, 출력 토큰 수, truncation(max_tokens 도달), 루프 여부, 출력 텍스트.
끝까지 돌면 <bench>__s<seed>.done.json 을 남긴다(run_matrix.py가 완료 여부 판단에 씀).

예: python src/evaluate.py --run qwen3-8b__nvfp4__gptq__trace --bench aime120 --seed 0
    python src/evaluate.py --run qwen3-8b__bf16 --bench aime120 --seed 0 --limit 20   # 빠른 확인
"""

import argparse
import json
import time

import benchmarks
from common import base_model_dir, eval_path, load_config, model_cfg, parse_run_name, run_model_dir
from engine import build_llm, gpu_info, sampling_params


def is_looping(token_ids, window, n, threshold):
    """마지막 window 토큰에서 n-gram 고유 비율이 threshold 미만이면 반복 루프로 본다(휴리스틱)."""
    tail = list(token_ids[-window:])
    if len(tail) < window // 2:
        return False
    grams = [tuple(tail[i : i + n]) for i in range(len(tail) - n + 1)]
    return len(set(grams)) / len(grams) < threshold


def load_done(path):
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["uid"] for line in f if line.strip()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--bench", required=True, choices=list(benchmarks.LOADERS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--limit", type=int, help="앞에서부터 N문제만(확인용). 완료 표시는 남기지 않음")
    p.add_argument("--config")
    args = p.parse_args()

    cfg = load_config(args.config)
    r = parse_run_name(args.run)
    mcfg = model_cfg(cfg, r["model"])
    model_dir = base_model_dir(cfg, r["model"]) if r["fmt"] == "bf16" else run_model_dir(cfg, args.run)
    out = eval_path(cfg, args.run, args.bench, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)

    items = benchmarks.load(args.bench)
    if args.limit:
        items = items[: args.limit]
    done = load_done(out)
    todo = [it for it in items if it["uid"] not in done]
    print(f"[{args.run} {args.bench} s{args.seed}] {len(done)} done, {len(todo)} to go")

    a = cfg["analysis"]
    gen_time, gen_tokens = 0.0, 0
    if todo:
        llm = build_llm(cfg, model_dir, mcfg)
        sp = sampling_params(cfg, args.seed)
        chunk = cfg["vllm"]["chunk"]
        for i in range(0, len(todo), chunk):
            batch = todo[i : i + chunk]
            msgs = [[{"role": "user", "content": it["prompt"]}] for it in batch]
            t0 = time.time()
            outs = llm.chat(msgs, sp, chat_template_kwargs=mcfg.get("chat_template_kwargs") or None, use_tqdm=True)
            gen_time += time.time() - t0
            with out.open("a") as f:
                for it, o in zip(batch, outs):
                    c = o.outputs[0]
                    truncated = c.finish_reason == "length"
                    gen_tokens += len(c.token_ids)
                    rec = {
                        "uid": it["uid"],
                        "gold": it["gold"],
                        "correct": benchmarks.grade(it, c.text, truncated),
                        "n_tokens": len(c.token_ids),
                        "n_prompt_tokens": len(o.prompt_token_ids or []),
                        "truncated": truncated,
                        "finish_reason": c.finish_reason,
                        "looping": is_looping(c.token_ids, a["loop_window"], a["loop_ngram"], a["loop_threshold"]),
                        "text": c.text,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with out.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    summary = {
        "run": args.run,
        "bench": args.bench,
        "seed": args.seed,
        "n": len(rows),
        "acc": sum(r["correct"] for r in rows) / max(len(rows), 1),
        "mean_tokens": sum(r["n_tokens"] for r in rows) / max(len(rows), 1),
        "truncation_rate": sum(r["truncated"] for r in rows) / max(len(rows), 1),
        "loop_rate": sum(r["looping"] for r in rows) / max(len(rows), 1),
        "throughput_tok_s_this_run": gen_tokens / gen_time if gen_time else None,
        "gpu": gpu_info(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.limit and len(rows) == len(items):
        out.with_suffix(".done.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
