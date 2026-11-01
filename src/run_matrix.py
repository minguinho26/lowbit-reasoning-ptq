"""실험 매트릭스 실행기. 설정 파일에서 실행 단위를 펼쳐서, 아직 안 끝난 것만 순서대로 돌린다.

각 단계는 맞는 가상환경의 python으로 해당 스크립트를 subprocess 호출한다.
  quant 환경: make_calib windows, quantize
  serve 환경: make_calib traces, evaluate, bench_speed

예: python src/run_matrix.py list
    python src/run_matrix.py calib    --model qwen3-8b
    python src/run_matrix.py quantize --model qwen3-8b
    python src/run_matrix.py evaluate --model qwen3-8b --bench aime120
    python src/run_matrix.py speed    --model qwen3-8b
    python src/run_matrix.py all      --model qwen3-8b      # calib -> quantize -> evaluate -> speed
    python src/run_matrix.py queue    --gpus 0,1 --models qwen3-8b,qwen3-4b,qwen3-1.7b --log-dir /workspace/logs/x

queue: GPU마다 워커를 하나씩 두고, 작업 단위(원본 다운로드, calibration, 양자화 1개, 평가 1개 = 실행 단위 x 벤치 x seed)를
       목록 순서대로 가져가 실행한다. 앞 단계(다운로드 -> calibration -> 양자화 -> 평가)가 끝난 단위만 가져가고,
       이미 끝난 단위는 건너뛴다. 실패한 단위에 의존하는 단위는 건너뛴다. 속도 측정은 하지 않는다(다른 GPU가 돌고 있으면
       측정이 흔들리므로 queue가 끝난 뒤 speed로 따로 잰다).
       목록 순서: 모든 모델 다운로드 -> 첫 모델 원본 평가(공개 수치 재현 확인) -> 모든 모델 calibration -> 모델 순서대로
       [원본 평가, 실행 단위마다 양자화 -> 평가]. 실행 단위는 검증용 W4A16, NVFP4 W4A4 RTN을 앞에 둔다.
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from common import calib_path, enumerate_runs, eval_path, load_config, parse_run_name, run_model_dir

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


def _calib_done(cfg, m):
    needed = {r["calib"] for r in select(cfg, m) if r["calib"]}
    return all(calib_path(cfg, m, s).exists() for s in needed)


def build_units(cfg, models):
    """(이름, 환경, 스크립트, 인자, 완료 여부 함수, 의존 단위 이름들)의 순서 있는 목록."""
    from common import base_model_dir

    units = []

    def add(name, env, script, args, done, deps):
        units.append({"name": name, "env": env, "script": script, "args": [str(a) for a in args], "done": done, "deps": deps})

    def add_evals(r, dep):
        for bench, bc in cfg["benchmarks"].items():
            for seed in bc["seeds"]:
                add(f"eval:{r['name']}:{bench}:s{seed}", "serve", "evaluate.py",
                    ["--run", r["name"], "--bench", bench, "--seed", seed],
                    lambda r=r, b=bench, s=seed: eval_done(cfg, r, b, s), [dep])

    def ordered_runs(m):
        runs = [r for r in select(cfg, m) if r["fmt"] != "bf16"]
        first = [r for r in runs if r["fmt"] == "w4a16"] + [r for r in runs if r["fmt"] == "nvfp4" and r["algo"] == "rtn"]
        return first + [r for r in runs if r not in first]

    for m in models:
        add(f"fetch:{m}", "serve", "fetch_models.py", ["--model", m],
            lambda m=m: (base_model_dir(cfg, m) / "config.json").exists(), [])
    bf16 = {m: parse_run_name(f"{m}__bf16") for m in models}
    add_evals(bf16[models[0]], f"fetch:{models[0]}")
    for m in models:
        add(f"calib:{m}", "serve", "run_matrix.py", ["calib", "--model", m], lambda m=m: _calib_done(cfg, m), [f"fetch:{m}"])
    for i, m in enumerate(models):
        if i > 0:
            add_evals(bf16[m], f"fetch:{m}")
        for r in ordered_runs(m):
            add(f"quant:{r['name']}", "quant", "quantize.py", ["--run", r["name"]],
                lambda r=r: is_quantized(cfg, r), [f"calib:{m}"])
            add_evals(r, f"quant:{r['name']}")
    return units


def cmd_queue(cfg, a):
    models = a.models.split(",") if a.models else list(cfg["models"])
    gpus = a.gpus.split(",")
    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    units = build_units(cfg, models)
    by_name = {u["name"]: u for u in units}
    state = {}  # 이름 -> running / ok / fail / blocked
    lock = threading.Lock()
    status_file = log_dir / "status.tsv"
    if a.dry:
        for u in units:
            print(f"{'done' if u['done']() else 'todo':4s}  {u['name']}  <- {','.join(u['deps']) or '-'}")
        return

    def dep_state(d):
        if state.get(d) in ("fail", "blocked"):
            return "fail"
        return "ok" if by_name[d]["done"]() else "pending"

    def take():
        """다음에 실행할 단위. 없으면 (None, 기다릴 게 남았는지)."""
        with lock:
            waiting = False
            for u in units:
                if u["name"] in state or u["done"]():
                    continue
                ds = [dep_state(d) for d in u["deps"]]
                if "fail" in ds:
                    state[u["name"]] = "blocked"
                    continue
                if all(s == "ok" for s in ds):
                    state[u["name"]] = "running"
                    return u, True
                waiting = True
            return None, waiting and any(s == "running" for s in state.values())

    def worker(gpu):
        while True:
            u, waiting = take()
            if u is None:
                if not waiting:
                    return
                time.sleep(30)
                continue
            cmd = [str(_python(cfg, u["env"])), str(SRC / u["script"]), *u["args"]]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            safe = u["name"].replace(":", "__")
            t0 = time.time()
            print(f"[{time.strftime('%F %T')}] gpu{gpu} start {u['name']}", flush=True)
            with open(log_dir / f"{safe}.log", "w") as f:
                f.write("$ " + " ".join(cmd) + "\n")
                f.flush()
                try:
                    rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=a.timeout).returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            ok = rc == 0 and u["done"]()
            mins = (time.time() - t0) / 60
            with lock:
                state[u["name"]] = "ok" if ok else "fail"
                with status_file.open("a") as f:
                    f.write(f"{u['name']}\tgpu{gpu}\t{'ok' if ok else f'FAIL({rc})'}\t{mins:.0f}min\n")
            print(f"[{time.strftime('%F %T')}] gpu{gpu} {'ok' if ok else 'FAIL'} {u['name']} ({mins:.0f}min)", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for i, t in enumerate(threads):
        t.start()
        time.sleep(5 * (i < len(threads) - 1))  # 두 워커가 같은 시각에 같은 볼륨에서 모델을 읽기 시작하지 않게
    for t in threads:
        t.join()
    left = [u["name"] for u in units if not u["done"]()]
    print(f"queue finished. not done: {len(left)}")
    for n in left:
        print(f"  {n} ({state.get(n, '-')})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["list", "calib", "quantize", "evaluate", "speed", "all", "queue"])
    p.add_argument("--model")
    p.add_argument("--fmt")
    p.add_argument("--bench")
    p.add_argument("--dry", action="store_true", help="명령만 출력")
    p.add_argument("--force", action="store_true", help="speed: 이미 있어도 다시 측정")
    p.add_argument("--gpus", default="0", help="queue: 쓸 GPU 번호, 쉼표로 구분")
    p.add_argument("--models", help="queue: 모델 순서, 쉼표로 구분. 앞의 모델부터 처리")
    p.add_argument("--log-dir", default="logs/queue", help="queue: 단위별 로그와 status.tsv 위치")
    p.add_argument("--timeout", type=int, default=6 * 3600, help="queue: 단위 하나의 최대 실행 시간(초)")
    p.add_argument("--config")
    a = p.parse_args()
    cfg = load_config(a.config)
    steps = {"list": [cmd_list], "calib": [cmd_calib], "quantize": [cmd_quantize], "evaluate": [cmd_evaluate],
             "speed": [cmd_speed], "all": [cmd_calib, cmd_quantize, cmd_evaluate, cmd_speed], "queue": [cmd_queue]}[a.step]
    for fn in steps:
        fn(cfg, a)


if __name__ == "__main__":
    main()
