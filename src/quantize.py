"""llm-compressor PTQ. 실행 단위 이름 하나를 받아 체크포인트(compressed-tensors 형식)를 만든다.

포맷(scheme): NVFP4(W4A4), W2A16, W4A16(sanity 전용). compressed-tensors preset 그대로 사용
알고리즘: rtn(QuantizationModifier), gptq(GPTQModifier), autoround(AutoRoundModifier)
calibration: make_calib.py windows 결과(토큰 윈도우)를 그대로 넣음. 섞지 않음

예: python src/quantize.py --run qwen35-4b__nvfp4__gptq__trace
    python src/quantize.py --run ministral3-8b__w2a16__rtn
"""

import argparse
import json
import re
import shutil
import time

from common import base_model_dir, calib_path, load_config, model_cfg, parse_run_name, run_model_dir

WEIGHT_FILES = re.compile(r".*\.safetensors$|.*\.safetensors\.index\.json$|^config\.json$")


def build_recipe(cfg, algo, scheme, ignore):
    a = cfg["algorithms"].get(algo) or {}
    if algo == "rtn":
        from llmcompressor.modifiers.quantization import QuantizationModifier

        return QuantizationModifier(targets="Linear", scheme=scheme, ignore=ignore)
    if algo == "gptq":
        from llmcompressor.modifiers.gptq import GPTQModifier

        return GPTQModifier(targets="Linear", scheme=scheme, ignore=ignore, **a)
    if algo == "autoround":
        from llmcompressor.modifiers.autoround import AutoRoundModifier

        return AutoRoundModifier(targets="Linear", scheme=scheme, ignore=ignore, **a)
    raise ValueError(algo)


def load_windows(path):
    from datasets import Dataset

    with path.open() as f:
        ids = [json.loads(line)["input_ids"] for line in f if line.strip()]
    return Dataset.from_dict({"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]})


def is_ignored(name, ignore):
    for pat in ignore:
        if pat.startswith("re:"):
            if re.match(pat[3:], name):
                return True
        elif name == pat or name.endswith("." + pat):
            return True
    return False


def quantized_fraction(model, ignore):
    """양자화 대상(Linear 중 ignore에 안 걸린 것) 파라미터 수 / 전체 파라미터 수."""
    import torch

    total = sum(p.numel() for p in model.parameters())
    q = sum(
        m.weight.numel()
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and not is_ignored(n, ignore)
    )
    return q / total, q, total


def copy_aux_files(src, out):
    """tokenizer, chat template, processor 설정 등 가중치가 아닌 파일을 원본에서 복사."""
    for f in src.iterdir():
        if f.is_file() and not WEIGHT_FILES.match(f.name) and not (out / f.name).exists():
            shutil.copy2(f, out / f.name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--config")
    args = p.parse_args()

    cfg = load_config(args.config)
    r = parse_run_name(args.run)
    assert r["fmt"] != "bf16", "bf16은 양자화 대상이 아님"
    mcfg = model_cfg(cfg, r["model"])
    scheme = cfg["formats"][r["fmt"]]["scheme"]
    ignore = mcfg["ignore"]
    src, out = base_model_dir(cfg, r["model"]), run_model_dir(cfg, args.run)
    if (out / "quant_meta.json").exists():
        print(f"exists: {out}")
        return

    import llmcompressor
    import torch
    import transformers
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    cls = AutoModelForImageTextToText if mcfg.get("multimodal") else AutoModelForCausalLM
    model = cls.from_pretrained(src, dtype="auto")
    frac, n_q, n_total = quantized_fraction(model, ignore)
    recipe = build_recipe(cfg, r["algo"], scheme, ignore)

    t0 = time.time()
    # 토크나이저 객체를 넘긴다. 경로(str)를 넘기면 llm-compressor가 AutoProcessor로 멀티모달 processor 전체를
    # 로드하려 하고, Qwen3.5의 video processor가 torchvision을 요구해서 실패한다. 데이터는 이미 토큰화돼 있음
    kw = {"model": model, "recipe": recipe, "tokenizer": AutoTokenizer.from_pretrained(src)}
    if mcfg.get("sequential_targets"):
        kw["sequential_targets"] = mcfg["sequential_targets"]
    if r["calib"]:
        ds = load_windows(calib_path(cfg, r["model"], r["calib"]))
        oneshot(
            dataset=ds,
            max_seq_length=cfg["calibration"]["seq_len"],
            num_calibration_samples=len(ds),
            shuffle_calibration_samples=False,
            **kw,
        )
    else:
        oneshot(**kw)
    elapsed = time.time() - t0

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, save_compressed=True)
    copy_aux_files(src, out)
    meta = {
        "run": args.run,
        "hf_id": mcfg["hf_id"],
        "scheme": scheme,
        "algorithm": r["algo"],
        "algorithm_args": cfg["algorithms"].get(r["algo"]) or {},
        "calibration": r["calib"],
        "calibration_args": {k: cfg["calibration"][k] for k in ("n_samples", "seq_len", "seed")} if r["calib"] else None,
        "ignore": ignore,
        "quantized_param_fraction": frac,
        "quantized_params": n_q,
        "total_params": n_total,
        "quantize_seconds": elapsed,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {"llmcompressor": llmcompressor.__version__, "transformers": transformers.__version__, "torch": torch.__version__},
    }
    (out / "quant_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
