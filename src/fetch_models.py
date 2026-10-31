"""BF16 원본을 HF 형식 파일만 받아 <workspace>/models/<model>__bf16 에 저장.

Ministral 저장소에는 Mistral 형식(consolidated.safetensors, params.json, tekken.json)도 같이 들어 있다.
원본과 양자화 모델을 vLLM에서 같은 경로(HF config, HF tokenizer)로 띄우기 위해 Mistral 형식 파일은 받지 않는다.

예: python src/fetch_models.py            # 설정의 모든 모델
    python src/fetch_models.py --model qwen35-4b
"""

import argparse

from huggingface_hub import snapshot_download

from common import base_model_dir, load_config, model_cfg

IGNORE = ["consolidated.safetensors", "params.json", "tekken.json", "original/*", "*.pth"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", help="여러 번 지정 가능. 생략하면 전부")
    p.add_argument("--config")
    args = p.parse_args()
    cfg = load_config(args.config)

    models = args.model or list(cfg["models"]) + list(cfg.get("sanity", {}))
    for m in models:
        out = base_model_dir(cfg, m)
        print(f"{model_cfg(cfg, m)['hf_id']} -> {out}")
        snapshot_download(model_cfg(cfg, m)["hf_id"], local_dir=out, ignore_patterns=IGNORE)


if __name__ == "__main__":
    main()
