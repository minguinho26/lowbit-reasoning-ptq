"""llm-compressor PTQ. 실행 단위 이름 하나를 받아 체크포인트(compressed-tensors 형식)를 만든다.

포맷(scheme): NVFP4(W4A4), NVFP4A16, W2A16, W4A16(sanity 전용). compressed-tensors preset 그대로 사용
알고리즘: rtn(QuantizationModifier), gptq(GPTQModifier), awq(AWQModifier + QuantizationModifier)
calibration: make_calib.py windows 결과(토큰 윈도우)를 그대로 넣음. 섞지 않음

예: python src/quantize.py --run qwen3-8b__nvfp4__gptq__trace
    python src/quantize.py --run r1-qwen-7b__w2a16__rtn
"""

import argparse
import inspect
import json
import re
import shutil
import time

from common import base_model_dir, calib_path, load_config, model_cfg, parse_run_name, run_model_dir

WEIGHT_FILES = re.compile(r".*\.safetensors$|.*\.safetensors\.index\.json$|^config\.json$")


def patch_awq_output_cache(modifier_cls):
    """AWQ가 사용하지 않는 attention 출력까지 모든 샘플에 걸쳐 보관하는 문제를 피한다.

    llm-compressor의 _run_samples는 tuple 전체를 목록에 모은 뒤 첫 tensor만 고른다.
    Qwen3 eager attention은 큰 attention matrix도 반환하므로 길이 2048의 128개
    샘플에서 불필요한 GPU 텐서가 누적된다. 각 forward 직후 같은 첫 tensor만
    남기면 반환값과 loss 계산 순서는 유지하면서 보조 출력은 즉시 해제된다.
    """
    import torch
    if getattr(modifier_cls._run_samples, "_primary_output_cache_patched", False):
        return
    original = inspect.getsource(modifier_cls._run_samples)
    if "outputs = [module(**batch) for batch in batch_iter]" not in original:
        raise RuntimeError("AWQ _run_samples 구현이 바뀜. 출력 cache 패치 호환성을 확인할 것")

    @torch.no_grad()
    def run_samples(self, module):
        from llmcompressor.core.session_functions import active_session

        cache = self._parent_args_cache[module]
        use_prefetch = active_session().state.sequential_prefetch
        batch_iter = cache.iter_prefetch() if use_prefetch else cache
        outputs = []
        for batch in batch_iter:
            result = module(**batch)
            outputs.append(result[0] if isinstance(result, tuple) else result)
            del result  # 다음 forward가 시작되기 전에 attention matrix 등 보조 출력을 해제한다.
        return outputs

    run_samples._primary_output_cache_patched = True
    modifier_cls._run_samples = run_samples


def build_recipe(cfg, algo, scheme, ignore):
    a = cfg["algorithms"].get(algo) or {}
    if algo == "rtn":
        from llmcompressor.modifiers.quantization import QuantizationModifier

        return QuantizationModifier(targets="Linear", scheme=scheme, ignore=ignore)
    if algo == "gptq":
        from llmcompressor.modifiers.gptq import GPTQModifier

        return GPTQModifier(targets="Linear", scheme=scheme, ignore=ignore, **a)
    if algo == "awq":
        import torch
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor.modifiers.transform.awq import AWQModifier

        patch_awq_output_cache(AWQModifier)
        # AWQ는 채널 스케일 변환을 하고, 다음 modifier가 선택한 포맷으로 양자화한다.
        # CPU offload는 calibration 데이터와 탐색 횟수를 유지하면서 GPU cache 사용량을 줄인다.
        awq_args = dict(a)
        offload_device = awq_args.pop("offload_device", "cpu")
        return [
            AWQModifier(offload_device=torch.device(offload_device) if offload_device else None, **awq_args),
            QuantizationModifier(targets="Linear", scheme=scheme, ignore=ignore),
        ]
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
    if r["algo"] not in cfg["formats"][r["fmt"]]["algorithms"]:
        raise ValueError(f"설정에서 제외된 알고리즘: {args.run}")
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
        "compatibility_patches": ["awq_primary_output_cache"] if r["algo"] == "awq" else [],
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
