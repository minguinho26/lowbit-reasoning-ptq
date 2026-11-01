"""결과 분석. 평가 jsonl과 속도 json을 읽어 표(csv)를 만든다.

main_table.csv     : 실행 단위 × 벤치별 정확도, 원본 대비 저하 폭과 95% CI, 길이, truncation, 루프
factor_effects.csv : 모델 × 포맷 × 벤치별 요인 효과. diff = 정확도(run_a) - 정확도(run_b), 95% CI 포함
                     algorithm   : calibration 고정, 알고리즘 간 최고 - 최저
                     calibration : 알고리즘 고정, calibration 간 최고 - 최저
                     vs_rtn      : 데이터를 쓰는 알고리즘 - RTN (음수면 RTN보다 나쁨)
                     activation, weight_format : 설정 파일 format_comparisons의 포맷 쌍. 같은 알고리즘, calibration끼리 b - a
speed.csv          : 배치 크기별 처리량, 원본 대비 속도 향상, effective speedup(= 속도 향상 / paired 길이 비율)

통계 처리: 문제 단위 paired bootstrap. 문제별 값은 seed 평균(정답률, 출력 길이).
길이 비율은 문제별 log(양자화/원본)의 평균을 bootstrap한 뒤 exp(기하평균 비율).

예: python src/analyze.py
"""

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from common import enumerate_runs, load_config, run_name


def load_eval(cfg):
    """{(run, bench): {uid: [rec per seed]}}"""
    root = cfg["_dirs"]["results"] / "eval"
    data = defaultdict(lambda: defaultdict(list))
    for f in sorted(root.glob("*/*.jsonl")):
        run, bench = f.parent.name, f.stem.split("__s")[0]
        with f.open() as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    data[(run, bench)][rec["uid"]].append(rec)
    return data


def per_problem(recs):
    """문제별 seed 평균: 정답률, 완료 샘플 평균 길이, 정답 샘플 평균 길이."""
    acc = np.mean([r["correct"] for r in recs])
    done = [r["n_tokens"] for r in recs if not r["truncated"]]
    corr = [r["n_tokens"] for r in recs if r["correct"]]
    return acc, (np.mean(done) if done else np.nan), (np.mean(corr) if corr else np.nan)


def bootstrap_ci(x, B, rng):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    idx = rng.integers(0, len(x), size=(B, len(x)))
    stats = x[idx].mean(axis=1)
    return x.mean(), np.percentile(stats, 2.5), np.percentile(stats, 97.5), len(x)


def describe(problems):
    toks = [r["n_tokens"] for recs in problems.values() for r in recs]
    flat = [r for recs in problems.values() for r in recs]
    return {
        "n_problems": len(problems),
        "n_samples": len(flat),
        "acc": float(np.mean([r["correct"] for r in flat])),
        "median_tokens": float(np.median(toks)),
        "p90_tokens": float(np.percentile(toks, 90)),
        "truncation_rate": float(np.mean([r["truncated"] for r in flat])),
        "loop_rate": float(np.mean([r["looping"] for r in flat])),
    }


def compare(q, b, B, rng):
    """양자화(q) vs 원본(b), 같은 문제끼리 짝지음."""
    uids = sorted(set(q) & set(b))
    pq = {u: per_problem(q[u]) for u in uids}
    pb = {u: per_problem(b[u]) for u in uids}
    d_acc = [pq[u][0] - pb[u][0] for u in uids]
    log_len_done = [np.log(pq[u][1] / pb[u][1]) for u in uids]
    log_len_corr = [np.log(pq[u][2] / pb[u][2]) for u in uids]
    m, lo, hi, _ = bootstrap_ci(d_acc, B, rng)
    lm, llo, lhi, ln = bootstrap_ci(log_len_done, B, rng)
    cm, clo, chi, cn = bootstrap_ci(log_len_corr, B, rng)
    return {
        "acc_drop": m, "acc_drop_lo": lo, "acc_drop_hi": hi,
        "acc_drop_distinguishable": bool(lo > 0 or hi < 0),
        "len_ratio_completed": np.exp(lm), "len_ratio_completed_lo": np.exp(llo), "len_ratio_completed_hi": np.exp(lhi),
        "len_ratio_completed_n": ln,
        "len_ratio_both_correct": np.exp(cm), "len_ratio_both_correct_lo": np.exp(clo), "len_ratio_both_correct_hi": np.exp(chi),
        "len_ratio_both_correct_n": cn,
    }


def pairwise_range(cfg, data, runs_by_key, bench, B, rng):
    """주어진 실행 단위 묶음 안에서 정확도 최대-최소 차이와, 그 두 설정 간 paired bootstrap CI."""
    accs = {n: np.mean([per_problem(v)[0] for v in data[(n, bench)].values()]) for n in runs_by_key if (n, bench) in data}
    if len(accs) < 2:
        return None
    hi_run = max(accs, key=accs.get)
    lo_run = min(accs, key=accs.get)
    a, b = data[(hi_run, bench)], data[(lo_run, bench)]
    uids = sorted(set(a) & set(b))
    d = [per_problem(a[u])[0] - per_problem(b[u])[0] for u in uids]
    m, lo, hi, _ = bootstrap_ci(d, B, rng)
    return {"run_a": hi_run, "run_b": lo_run, "diff": m, "diff_lo": lo, "diff_hi": hi}


def write_csv(path, rows):
    if not rows:
        print(f"(empty) {path}")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {path} ({len(rows)} rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    args = p.parse_args()
    cfg = load_config(args.config)
    B = cfg["analysis"]["bootstrap"]
    rng = np.random.default_rng(0)
    out = cfg["_dirs"]["results"] / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    np.seterr(divide="ignore", invalid="ignore")
    data = load_eval(cfg)
    runs = enumerate_runs(cfg)
    benches = list(cfg["benchmarks"])

    # 1) main table
    main_rows, len_ratio = [], {}
    for r in runs:
        base = run_name(r["model"])
        for bench in benches:
            if (r["name"], bench) not in data:
                continue
            row = {"run": r["name"], "model": r["model"], "fmt": r["fmt"], "algo": r["algo"], "calib": r["calib"], "bench": bench}
            row.update(describe(data[(r["name"], bench)]))
            if r["fmt"] != "bf16" and (base, bench) in data:
                row.update(compare(data[(r["name"], bench)], data[(base, bench)], B, rng))
                len_ratio[(r["name"], bench)] = row["len_ratio_completed"]
            meta = cfg["_dirs"]["models"] / r["name"] / "quant_meta.json"
            if meta.exists():
                row["quantized_param_fraction"] = json.loads(meta.read_text())["quantized_param_fraction"]
            main_rows.append(row)
    write_csv(out / "main_table.csv", main_rows)

    # 2) factor effects
    fac_rows = []
    for model in cfg["models"]:
        for fmt, f in cfg["formats"].items():
            if not f.get("main", True):
                continue
            for bench in benches:
                for calib in cfg["calibration"]["sources"]:
                    group = [run_name(model, fmt, a, calib) for a in f["algorithms"] if a in f["calib_dependent"]]
                    res = pairwise_range(cfg, data, group, bench, B, rng)
                    if res:
                        fac_rows.append({"model": model, "fmt": fmt, "bench": bench, "factor": "algorithm", "fixed": f"calib={calib}", **res})
                for algo in f["calib_dependent"]:
                    group = [run_name(model, fmt, algo, c) for c in cfg["calibration"]["sources"]]
                    res = pairwise_range(cfg, data, group, bench, B, rng)
                    if res:
                        fac_rows.append({"model": model, "fmt": fmt, "bench": bench, "factor": "calibration", "fixed": f"algo={algo}", **res})
                # 데이터를 쓰는 알고리즘 vs RTN (RTN보다 나빠지는 경우가 있는지)
                for algo in f["algorithms"]:
                    if algo == "rtn":
                        continue
                    for calib in cfg["calibration"]["sources"]:
                        a_run = run_name(model, fmt, algo, calib)
                        rtn_run = run_name(model, fmt, "rtn", calib if "rtn" in f["calib_dependent"] else None)
                        if (a_run, bench) not in data or (rtn_run, bench) not in data:
                            continue
                        a, b = data[(a_run, bench)], data[(rtn_run, bench)]
                        uids = sorted(set(a) & set(b))
                        d = [per_problem(a[u])[0] - per_problem(b[u])[0] for u in uids]
                        m, lo, hi, _ = bootstrap_ci(d, B, rng)
                        fac_rows.append({"model": model, "fmt": fmt, "bench": bench, "factor": "vs_rtn",
                                         "fixed": f"algo={algo},calib={calib}", "run_a": a_run, "run_b": rtn_run,
                                         "diff": m, "diff_lo": lo, "diff_hi": hi})
    # 요인 하나만 다른 포맷 쌍
    def paired_diff(a_run, b_run, bench):
        if (a_run, bench) not in data or (b_run, bench) not in data:
            return None
        a, b = data[(a_run, bench)], data[(b_run, bench)]
        uids = sorted(set(a) & set(b))
        d = [per_problem(b[u])[0] - per_problem(a[u])[0] for u in uids]
        m, lo, hi, _ = bootstrap_ci(d, B, rng)
        return {"run_a": a_run, "run_b": b_run, "diff": m, "diff_lo": lo, "diff_hi": hi}

    for comp in cfg.get("format_comparisons", []):
        fa, fb = cfg["formats"][comp["a"]], cfg["formats"][comp["b"]]
        for model in cfg["models"]:
            for bench in benches:
                for algo in fa["algorithms"]:
                    if algo not in fb["algorithms"]:
                        continue
                    dep_a, dep_b = algo in fa["calib_dependent"], algo in fb["calib_dependent"]
                    for calib in (cfg["calibration"]["sources"] if dep_a or dep_b else [None]):
                        res = paired_diff(run_name(model, comp["a"], algo, calib if dep_a else None),
                                          run_name(model, comp["b"], algo, calib if dep_b else None), bench)
                        if res:
                            fac_rows.append({"model": model, "fmt": f"{comp['a']}->{comp['b']}", "bench": bench,
                                             "factor": comp["factor"], "fixed": f"algo={algo},calib={calib}", **res})
    write_csv(out / "factor_effects.csv", fac_rows)

    # 3) speed
    sp_rows = []
    speed_dir = cfg["_dirs"]["results"] / "speed"
    rb = cfg["speed"]["report_batch"]
    for r in runs:
        f = speed_dir / f"{r['name']}.json"
        fb = speed_dir / f"{run_name(r['model'])}.json"
        if not f.exists():
            continue
        rec = json.loads(f.read_text())
        base = json.loads(fb.read_text()) if fb.exists() else None
        for res in rec["results"]:
            row = {"run": r["name"], "batch_size": res["batch_size"], "output_tok_s": res["output_tok_s"],
                   "checkpoint_gb": rec["checkpoint_bytes"] / 1e9, "gpu": rec["gpu"]}
            if base:
                b = next(x for x in base["results"] if x["batch_size"] == res["batch_size"])
                row["speedup_vs_bf16"] = res["output_tok_s"] / b["output_tok_s"]
                if res["batch_size"] == rb:
                    for bench in benches:
                        lr = len_ratio.get((r["name"], bench))
                        if lr and lr == lr:
                            row[f"effective_speedup_{bench}"] = row["speedup_vs_bf16"] / lr
            sp_rows.append(row)
    write_csv(out / "speed.csv", sp_rows)


if __name__ == "__main__":
    main()
