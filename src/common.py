"""설정 파일 로딩, 실행 단위(run) 이름 규칙, 경로."""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "experiment.yaml"


def load_config(path=None):
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    ws = Path(os.environ.get("WS", cfg["paths"]["workspace"]))
    cfg["_ws"] = ws
    cfg["_dirs"] = {k: ws / v for k, v in cfg["paths"]["subdirs"].items()}
    return cfg


def run_name(model, fmt="bf16", algo=None, calib=None):
    return "__".join([model, fmt] + [p for p in (algo, calib) if p])


def parse_run_name(name):
    parts = name.split("__")
    return {
        "model": parts[0],
        "fmt": parts[1],
        "algo": parts[2] if len(parts) > 2 else None,
        "calib": parts[3] if len(parts) > 3 else None,
        "name": name,
    }


def model_cfg(cfg, model):
    if model in cfg["models"]:
        return cfg["models"][model]
    return cfg["sanity"][model]


def enumerate_runs(cfg, include_sanity=True):
    runs = []
    for model in cfg["models"]:
        runs.append(run_name(model))
        for fmt, f in cfg["formats"].items():
            if not f.get("main", True):
                continue
            for algo in f["algorithms"]:
                calibs = cfg["calibration"]["sources"] if algo in f["calib_dependent"] else [None]
                for calib in calibs:
                    runs.append(run_name(model, fmt, algo, calib))
    if include_sanity:
        for model, s in cfg.get("sanity", {}).items():
            for r in s["runs"]:
                runs.append(run_name(model, **r))
    return [parse_run_name(n) for n in runs]


def base_model_dir(cfg, model):
    return cfg["_dirs"]["models"] / run_name(model)


def run_model_dir(cfg, name):
    return cfg["_dirs"]["models"] / name


def calib_path(cfg, model, source):
    return cfg["_dirs"]["calib"] / model / f"{source}.windows.jsonl"


def eval_path(cfg, name, bench, seed):
    return cfg["_dirs"]["results"] / "eval" / name / f"{bench}__s{seed}.jsonl"


def venv_python(cfg, which):
    return cfg["_ws"] / cfg["paths"][f"venv_{which}"] / "bin" / "python"
