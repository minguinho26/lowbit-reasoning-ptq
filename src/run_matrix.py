"""실험 매트릭스 실행기. 설정 파일에서 실행 단위를 펼쳐서, 아직 안 끝난 것만 순서대로 돌린다.

각 단계는 맞는 가상환경의 python으로 해당 스크립트를 subprocess 호출한다.
  quant 환경: make_calib windows, quantize
  serve 환경: make_calib traces, evaluate, bench_speed

예: python src/run_matrix.py list
    python src/run_matrix.py calib    --model qwen35-4b
    python src/run_matrix.py quantize --model qwen35-4b
    python src/run_matrix.py evaluate --model qwen35-4b --bench aime120
    python src/run_matrix.py speed    --model qwen35-4b
    python src/run_matrix.py all      --model qwen35-4b      # calib -> quantize -> evaluate -> speed
"""

import argparse
import subprocess
import sys
from pathlib import Path

from common import calib_path, enumerate_runs, eval_path, load_config, run_model_dir

SRC = Path(__file__).resolve().parent


def call(cfg, env, script, *args, dry=False):
    cmd = [str(_python(cfg, env)), str(SRC / script), *map(str, args)]
    print("$", " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def _python(cfg, env):
    from common import venv_python

    py = venv_python(cfg, env)
    return py if py.exists() else Path(sys.executable)


def select(cfg, model=None, fmt=None):
    return [r for r in enumerate_runs(cfg) if (not model or r["model"] == model) and (not fmt or r["fmt"] == fmt)]


def is_quantized(cfg, r):
    return r["fmt"] == "bf16" or (run_model_dir(cfg, r["name"]) / "quant_meta.json").exists()


def eval_done(cfg, r, bench, seed):
    return eval_path(cfg, r["name"], bench, seed).with_suffix(".done.json").exists()


def cmd_list(cfg, a):
    for r in select(cfg, a.model, a.fmt):
        ev = [f"{b}:s{s}" for b, bc in cfg["benchmarks"].items() for s in bc["seeds"] if eval_done(cfg, r, b, s)]
        sp = (cfg["_dirs"]["results"] / "speed" / f"{r['name']}.json").exists()
        print(f"{r['name']:45s} quant={'Y' if is_quantized(cfg, r) else '-'}  speed={'Y' if sp else '-'}  eval={','.join(ev) or '-'}")
    print(f"total runs: {len(select(cfg, a.model, a.fmt))}")


def cmd_calib(cfg, a):
    models = sorted({r["model"] for r in select(cfg, a.model) if r["calib"]})
    for m in models:
        needed = {r["calib"] for r in select(cfg, m) if r["calib"]}
        if "trace" in needed and not (cfg["_dirs"]["calib"] / m / "trace.generations.jsonl").exists():
            call(cfg, "serve", "make_calib.py", "traces", "--model", m, dry=a.dry)
        for src in sorted(needed):
            if not calib_path(cfg, m, src).exists():
                call(cfg, "quant", "make_calib.py", "windows", "--model", m, "--source", src, dry=a.dry)


def cmd_quantize(cfg, a):
    for r in select(cfg, a.model, a.fmt):
        if not is_quantized(cfg, r):
            call(cfg, "quant", "quantize.py", "--run", r["name"], dry=a.dry)


def cmd_evaluate(cfg, a):
    for r in select(cfg, a.model, a.fmt):
        for bench, bc in cfg["benchmarks"].items():
            if a.bench and bench != a.bench:
                continue
            for seed in bc["seeds"]:
                if not eval_done(cfg, r, bench, seed):
                    call(cfg, "serve", "evaluate.py", "--run", r["name"], "--bench", bench, "--seed", seed, dry=a.dry)


def cmd_speed(cfg, a):
    # 원본을 먼저 재고 같은 모델의 양자화 설정을 연달아 잰다(같은 pod에서 돌릴 것)
    runs = sorted(select(cfg, a.model, a.fmt), key=lambda r: (r["model"], r["fmt"] != "bf16", r["name"]))
    for r in runs:
        if not (cfg["_dirs"]["results"] / "speed" / f"{r['name']}.json").exists() or a.force:
            call(cfg, "serve", "bench_speed.py", "--run", r["name"], dry=a.dry)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["list", "calib", "quantize", "evaluate", "speed", "all"])
    p.add_argument("--model")
    p.add_argument("--fmt")
    p.add_argument("--bench")
    p.add_argument("--dry", action="store_true", help="명령만 출력")
    p.add_argument("--force", action="store_true", help="speed: 이미 있어도 다시 측정")
    p.add_argument("--config")
    a = p.parse_args()
    cfg = load_config(a.config)
    steps = {"list": [cmd_list], "calib": [cmd_calib], "quantize": [cmd_quantize], "evaluate": [cmd_evaluate],
             "speed": [cmd_speed], "all": [cmd_calib, cmd_quantize, cmd_evaluate, cmd_speed]}[a.step]
    for fn in steps:
        fn(cfg, a)


if __name__ == "__main__":
    main()
