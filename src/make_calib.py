"""calibration 데이터 생성. 2504.04823과 같은 방식.

traces  : NuminaMath-1.5 문제에 대해 원본(BF16) 모델이 평가와 같은 디코딩으로 응답을 생성 (서빙 환경에서 실행)
windows : 텍스트를 이어 붙여 토큰화한 뒤 seq_len 길이 윈도우를 n_samples개 무작위 추출 (양자화 환경에서 실행)
          source=wikitext2 : wikitext-2-raw-v1 train
          source=trace     : 위 traces 결과의 (chat template이 적용된 프롬프트 + 응답)

오염 방지: source가 trace_exclude_sources(amc_aime)인 문제, 평가 문항(AIME-120, GPQA-Diamond)과
decontam_ngram 단어 n-gram이 겹치는 문제를 뺀 뒤 앞에서부터 trace_problems개를 쓴다.

예: python src/make_calib.py traces  --model qwen3-8b
    python src/make_calib.py windows --model qwen3-8b --source wikitext2
    python src/make_calib.py windows --model qwen3-8b --source trace
"""

import argparse
import json
import random
import re

from common import base_model_dir, calib_path, load_config, model_cfg


def _words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(text, n):
    w = _words(text)
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def select_problems(cfg):
    import benchmarks
    from datasets import load_dataset

    c = cfg["calibration"]
    n = c["decontam_ngram"]
    banned = set()
    for bench in cfg["benchmarks"]:
        for it in benchmarks.load(bench):
            banned |= _ngrams(it["question"], n)

    ds = load_dataset("AI-MO/NuminaMath-1.5", split="train")
    picked, skipped = [], {"source": 0, "overlap": 0}
    for i, row in enumerate(ds):
        if row["source"] in c["trace_exclude_sources"]:
            skipped["source"] += 1
            continue
        if _ngrams(row["problem"], n) & banned:
            skipped["overlap"] += 1
            continue
        picked.append({"uid": f"numina-{i}", "problem": row["problem"]})
        if len(picked) == c["trace_problems"]:
            break
    print(f"picked {len(picked)} problems, skipped {skipped}")
    return picked


def traces(cfg, model):
    import benchmarks
    from engine import build_llm, sampling_params

    mcfg = model_cfg(cfg, model)
    out = cfg["_dirs"]["calib"] / model / "trace.generations.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"exists: {out}")
        return

    problems = select_problems(cfg)
    llm = build_llm(cfg, base_model_dir(cfg, model), mcfg)
    tok = llm.get_tokenizer()
    kw = mcfg.get("chat_template_kwargs") or {}
    msgs = [[{"role": "user", "content": p["problem"] + benchmarks.MATH_SUFFIX}] for p in problems]
    outs = llm.chat(msgs, sampling_params(cfg, cfg["calibration"]["seed"]), chat_template_kwargs=kw or None)
    with out.open("w") as f:
        for p, m, o in zip(problems, msgs, outs):
            prompt = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw)
            c = o.outputs[0]
            f.write(json.dumps({
                "uid": p["uid"],
                "full_prompt": prompt,
                "generated_text": c.text,
                "n_tokens": len(c.token_ids),
                "finish_reason": c.finish_reason,
            }, ensure_ascii=False) + "\n")
    print(f"saved -> {out}")


def _source_text(cfg, model, source):
    if source == "wikitext2":
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        return "\n\n".join(t.strip() for t in ds["text"] if t.strip())
    if source == "trace":
        path = cfg["_dirs"]["calib"] / model / "trace.generations.jsonl"
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return "".join(r["full_prompt"] + r["generated_text"] + "\n\n" for r in rows)
    raise ValueError(source)


def windows(cfg, model, source):
    from transformers import AutoTokenizer

    c = cfg["calibration"]
    out = calib_path(cfg, model, source)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base_model_dir(cfg, model))
    ids = tok(_source_text(cfg, model, source), add_special_tokens=False)["input_ids"]
    L = c["seq_len"]
    assert len(ids) > L + 1, f"{source} 토큰이 {len(ids)}개뿐임"
    rng = random.Random(c["seed"])
    with out.open("w") as f:
        for _ in range(c["n_samples"]):
            i = rng.randint(0, len(ids) - L - 1)
            f.write(json.dumps({"input_ids": ids[i : i + L]}) + "\n")
    print(f"{source}: {len(ids)} tokens -> {c['n_samples']} x {L} windows -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["traces", "windows"])
    p.add_argument("--model", required=True)
    p.add_argument("--source", choices=["wikitext2", "trace"])
    p.add_argument("--config")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.step == "traces":
        traces(cfg, args.model)
    else:
        assert args.source, "--source 필요"
        windows(cfg, args.model, args.source)


if __name__ == "__main__":
    main()
