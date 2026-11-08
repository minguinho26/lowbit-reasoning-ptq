"""BF16 원본을 <workspace>/models/<model>__bf16 에 저장.

예: python src/fetch_models.py            # 설정의 모든 모델
    python src/fetch_models.py --model qwen3-8b
"""

import argparse

from huggingface_hub import snapshot_download

from common import base_model_dir, load_config, model_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", help="여러 번 지정 가능. 생략하면 전부")
    p.add_argument("--config")
    args = p.parse_args()
    cfg = load_config(args.config)

    models = args.model or list(cfg["models"])
    for m in models:
        out = base_model_dir(cfg, m)
        print(f"{model_cfg(cfg, m)['hf_id']} -> {out}")
        snapshot_download(model_cfg(cfg, m)["hf_id"], local_dir=out)


if __name__ == "__main__":
    main()
