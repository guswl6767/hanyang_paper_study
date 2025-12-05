#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
paper_viz_roles.py
- Teacher / Student JSON 경로를 분리 지정해서 안전하게 매칭
- 표(CSV/LaTeX) + 그림(절대/상대/코스트/파레토/스윕) 한 번에 생성
- Matplotlib only (no seaborn), 색상 지정 없음
"""

import os
import re
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

DATASETS_DEFAULT = [
    "exchange_rate", "solar_nips", "electricity_nips",
    "traffic_nips", "taxi_30min", "wiki-rolling_nips"
]

# ------------------------------
# Utilities for loading
# ------------------------------

def _iter_paths(paths):
    """Yield json file paths from a list of dirs/files/globs."""
    for p in paths:
        if os.path.isdir(p):
            for fp in glob.glob(os.path.join(p, "**", "*.json"), recursive=True):
                yield fp
        else:
            for fp in glob.glob(p):
                if fp.endswith(".json"):
                    yield fp

def _guess_dataset_from_name(fp, datasets):
    base = os.path.basename(fp)
    for d in datasets:
        if base.startswith(d):
            return d
    return None

def _load_by_role(paths, role, datasets):
    """
    paths: list[str] (dir/file/glob).
    role : "teacher" | "student"
    Return: dict[dataset] -> picked json (latest by mtime)
    """
    bucket = {ds: [] for ds in datasets}
    for fp in _iter_paths(paths):
        try:
            with open(fp, "r") as f:
                j = json.load(f)
        except Exception:
            continue
        ds = j.get("dataset") or _guess_dataset_from_name(fp, datasets)
        if ds not in bucket:
            continue

        # decide role
        r = j.get("model_role")
        base = os.path.basename(fp).lower()
        name_has_teacher = ("teacher" in base)
        name_has_student = ("student" in base)

        accept = False
        if r in ("teacher", "student"):
            accept = (r == role)
        else:
            if role == "teacher" and name_has_teacher:
                accept = True
            elif role == "student" and name_has_student:
                accept = True

        if not accept:
            continue

        mtime = os.path.getmtime(fp)
        bucket[ds].append((mtime, fp, j))

    # pick latest for each dataset
    picked = {}
    for ds, items in bucket.items():
        if not items:
            continue
        items.sort(key=lambda x: x[0], reverse=True)
        picked[ds] = items[0][2]
    return picked

def _f(m, key, default=np.nan):
    try:
        return float(m[key])
    except Exception:
        return default

# ------------------------------
# Rows computation (with stds)
# ------------------------------

def compute_rows(teach_by_ds, stud_by_ds, order):
    """
    Return list of dict rows with T/S metrics (means, stds) + deltas
    """
    rows = []
    for ds in order:
        t = teach_by_ds.get(ds)
        s = stud_by_ds.get(ds)
        if not t or not s:
            continue

        crpsT = _f(t, "crps_meanD_mean");  crpsT_std = _f(t, "crps_meanD_std")
        crpsS = _f(s, "crps_meanD_mean");  crpsS_std = _f(s, "crps_meanD_std")
        timeT = _f(t, "infer_time_sec_mean"); timeT_std = _f(t, "infer_time_sec_std")
        timeS = _f(s, "infer_time_sec_mean"); timeS_std = _f(s, "infer_time_sec_std")
        parT  = int(t.get("params", 0))
        parS  = int(s.get("params", 0))
        windows = int(s.get("windows", t.get("windows", 0)))
        D = int(s.get("D", t.get("D", 0)))
        pred_len = int(s.get("pred_len", t.get("pred_len", 0)))

        speedup  = timeT / max(timeS, 1e-9) if (timeT>0 and timeS>0) else np.nan
        par_red  = (1 - parS / max(parT, 1)) * 100.0 if parT > 0 else np.nan
        dcrps_pct = (crpsS / max(crpsT, 1e-9) - 1.0) * 100.0 if crpsT > 0 else np.nan

        rows.append({
            "ds": ds, "D": D, "pred_len": pred_len,
            "parT": parT, "parS": parS, "par_red_pct": par_red,
            "crpsT": crpsT, "crpsT_std": crpsT_std,
            "crpsS": crpsS, "crpsS_std": crpsS_std,
            "d_crps_pct": dcrps_pct,
            "timeT": timeT, "timeT_std": timeT_std,
            "timeS": timeS, "timeS_std": timeS_std,
            "speedup": speedup, "windows": windows
        })

    return rows

# ------------------------------
# Tables (CSV / LaTeX)
# ------------------------------

def save_csv_tex(rows, outdir):
    import csv
    os.makedirs(outdir, exist_ok=True)

    csv_path = os.path.join(outdir, "table_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Dataset","Params(T)","Params(S)","↓Params%",
            "CRPS_meanD(T)","CRPS_meanD(S)","ΔS−T",
            "Time(T)","Time(S)","Speedup(T/S)","Windows"
        ])
        for r in rows:
            w.writerow([
                r["ds"], r["parT"], r["parS"], f"{r['par_red_pct']:.1f}%",
                f"{r['crpsT']:.6f}", f"{r['crpsS']:.6f}", f"{r['crpsS']-r['crpsT']:+.6f}",
                f"{r['timeT']:.3f}", f"{r['timeS']:.3f}", f"{r['speedup']:.2f}",
                r["windows"]
            ])

    tex_path = os.path.join(outdir, "table_results.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n")
        f.write("Dataset & Par(T) & Par(S) & $\\downarrow$Par\\% & CRPS(T) & CRPS(S) & $\\Delta$ & Time(T) & Time(S) & Speedup\\\\\n\\midrule\n")
        for r in rows:
            f.write(
                f"{r['ds']} & {r['parT']:,} & {r['parS']:,} & {r['par_red_pct']:.1f}\\% & "
                f"{r['crpsT']:.6f} & {r['crpsS']:.6f} & {r['crpsS']-r['crpsT']:+.6f} & "
                f"{r['timeT']:.3f} & {r['timeS']:.3f} & {r['speedup']:.2f}\\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"[OK] saved: {csv_path}, {tex_path}")

# ------------------------------
# Figures
# ------------------------------

def bar_crps_abs(rows, outdir, use_log=True, fname="fig_crps_meanD_bar"):
    labels = [r["ds"] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    T = [r["crpsT"] for r in rows]
    S = [r["crpsS"] for r in rows]
    plt.figure(figsize=(10,4))
    plt.bar(x-w/2, T, width=w, label="Teacher")
    plt.bar(x+w/2, S, width=w, label="Student")
    plt.ylabel("CRPS_meanD (lower is better)")
    if use_log:
        plt.yscale("log")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.legend(); plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(outdir, f"{fname}.{ext}"), dpi=200 if ext=="png" else None)
    plt.close()

def bar_crps_abs_err(rows, outdir, use_log=True, fname="fig_crps_meanD_bar_err"):
    labels = [r["ds"] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    T  = [r["crpsT"] for r in rows]; Tsd = [r["crpsT_std"] for r in rows]
    S  = [r["crpsS"] for r in rows]; Ssd = [r["crpsS_std"] for r in rows]
    plt.figure(figsize=(10,4))
    plt.bar(x-w/2, T, width=w, label="Teacher", yerr=Tsd, capsize=3)
    plt.bar(x+w/2, S, width=w, label="Student", yerr=Ssd, capsize=3)
    plt.ylabel("CRPS_meanD (lower is better)")
    if use_log:
        plt.yscale("log")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.legend(); plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(outdir, f"{fname}.{ext}"), dpi=200 if ext=="png" else None)
    plt.close()

def bar_crps_abs_small(rows, outdir, exclude=("wiki-rolling_nips",), ymax=None, fname="fig_crps_meanD_bar_small"):
    rows_small = [r for r in rows if r["ds"] not in exclude]
    labels = [r["ds"] for r in rows_small]
    x = np.arange(len(rows_small)); w = 0.38
    T = [r["crpsT"] for r in rows_small]
    S = [r["crpsS"] for r in rows_small]
    plt.figure(figsize=(10,4))
    plt.bar(x-w/2, T, width=w, label="Teacher")
    plt.bar(x+w/2, S, width=w, label="Student")
    plt.ylabel("CRPS_meanD (lower is better)")
    if ymax is not None:
        plt.ylim(0, ymax)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.legend(); plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(outdir, f"{fname}.{ext}"), dpi=200 if ext=="png" else None)
    plt.close()

def _annotate_bars(ax, vals, fmt="{:.2f}"):
    for i, v in enumerate(vals):
        ax.text(i, v + (0.02 if v >= 0 else -0.02),
                fmt.format(v),
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

def bar_crps_ratio_labeled(rows, outdir, fname="fig_crps_ratio_bar_labeled"):
    labels = [r["ds"] for r in rows]
    x = np.arange(len(rows))
    ratio = [r["crpsS"] / max(r["crpsT"], 1e-9) for r in rows]
    fig = plt.figure(figsize=(10,4)); ax = plt.gca()
    ax.bar(x, ratio); ax.axhline(1.0, ls="--", lw=1)
    ax.set_ylabel("CRPS_meanD ratio (Student / Teacher, lower is better)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    _annotate_bars(ax, ratio, "{:.2f}")
    plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(outdir, f"{fname}.{ext}"), dpi=200 if ext=="png" else None)
    plt.close()

def bar_crps_delta_labeled(rows, outdir, fname="fig_crps_delta_bar_labeled"):
    labels = [r["ds"] for r in rows]
    x = np.arange(len(rows))
    dperc = [(r["crpsS"] / max(r["crpsT"], 1e-9) - 1.0) * 100.0 for r in rows]
    fig = plt.figure(figsize=(10,4)); ax = plt.gca()
    ax.bar(x, dperc); ax.axhline(0.0, ls="--", lw=1)
    ax.set_ylabel("Δ CRPS_meanD (%) vs Teacher (lower is better)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    _annotate_bars(ax, dperc, "{:+.1f}%")
    plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(outdir, f"{fname}.{ext}"), dpi=200 if ext=="png" else None)
    plt.close()

def bar_speedup(rows, outdir, fname="fig_speedup_bar"):
    labels = [r["ds"] for r in rows]
    x = np.arange(len(rows))
    sp = [r["speedup"] for r in rows]
    plt.figure(figsize=(10,4))
    plt.bar(x, sp)
    plt.ylabel("Speedup (Teacher Time / Student Time)")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(outdir, f"{fname}.{ext}"), dpi=200 if ext=="png" else None)
    plt.close()

def bar_param_reduction(rows, outdir, fname="fig_param_reduction_bar"):
    labels = [r["ds"] for r in rows]
    x = np.arange(len(rows))
    pr = [r["par_red_pct"] for r in rows]
    plt.figure(figsize=(10,4))
    plt.bar(x, pr)
    plt.ylabel("Param reduction (%)")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(outdir, f"{fname}.{ext}"), dpi=200 if ext=="png" else None)
    plt.close()

def bar_total_time(rows, outdir, fname="fig_total_infer_time_bar"):
    labels = [r["ds"] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    TT = [r["timeT"] * r["windows"] for r in rows]
    TS = [r["timeS"] * r["windows"] for r in rows]
    plt.figure(figsize=(10,4))
    plt.bar(x - w/2, TT, width=w, label="Teacher total")
    plt.bar(x + w/2, TS, width=w, label="Student total")
    plt.ylabel("Total inference time over windows (s)")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.legend(); plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(outdir, f"{fname}.{ext}"), dpi=200 if ext=="png" else None)
    plt.close()

def pareto_tradeoff(rows, outdir, fname="fig_tradeoff_pareto"):
    x = [r["speedup"] for r in rows]
    y = [r["d_crps_pct"] for r in rows]
    sizes = [max(50, r["par_red_pct"] * 5) for r in rows]  # bubble ~ param reduction
    labels = [r["ds"] for r in rows]
    plt.figure(figsize=(6,5))
    plt.scatter(x, y, s=sizes, alpha=0.8)
    for xi, yi, lab in zip(x, y, labels):
        plt.text(xi * 1.01, yi, lab, fontsize=9)
    plt.axhline(0, ls="--", lw=1); plt.axvline(1, ls="--", lw=1)
    plt.xlabel("Speedup (×, higher is better)")
    plt.ylabel("Δ CRPS_meanD (%) vs Teacher (lower is better)")
    plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(outdir, f"{fname}.{ext}"), dpi=200 if ext=="png" else None)
    plt.close()

# Optional: sweep curve (CRPS vs time) for multiple student configs in one dataset
def pareto_sweep(student_paths, dataset, outdir, fname=None):
    all_paths = []
    for p in student_paths:
        if os.path.isdir(p):
            all_paths += glob.glob(os.path.join(p, "**", "*.json"), recursive=True)
        else:
            all_paths += glob.glob(p)

    pts = []
    for fp in all_paths:
        try:
            j = json.load(open(fp))
        except Exception:
            continue
        if j.get("dataset") != dataset or j.get("model_role") != "student":
            continue
        crpsS = _f(j, "crps_meanD_mean")
        timeS = _f(j, "infer_time_sec_mean")
        cfg = j.get("model_cfg", {})
        n = int(cfg.get("n_diffusion_steps", 0))
        ch = int(cfg.get("residual_channels", 0))
        pts.append((n, ch, crpsS, timeS, fp))

    if not pts:
        print(f"[WARN] no student sweeps for {dataset}")
        return

    pts.sort(key=lambda x: (x[0], x[1]))  # n, ch order
    x = [p[3] for p in pts]  # time
    y = [p[2] for p in pts]  # crps
    plt.figure(figsize=(6,5))
    plt.plot(x, y, marker="o")
    for (n, ch, crpsS, timeS, fp), xi, yi in zip(pts, x, y):
        plt.text(xi * 1.01, yi, f"n={n},ch={ch}", fontsize=8)
    plt.xlabel("Student time per window (s)")
    plt.ylabel("CRPS_meanD (lower is better)")
    plt.tight_layout()
    if not fname:
        fname = f"fig_sweep_{dataset}"
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(outdir, f"{fname}.{ext}"), dpi=200 if ext=="png" else None)
    plt.close()

# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Paper visualization (teacher/student separated roots)")
    ap.add_argument("--teacher", nargs="+", required=True,
                    help="Teacher json 경로(들): 디렉터리/파일/글롭 혼용 가능, 재귀 탐색")
    ap.add_argument("--student", nargs="+", required=True,
                    help="Student json 경로(들): 디렉터리/파일/글롭 혼용 가능, 재귀 탐색")
    ap.add_argument("--out", required=True, help="그림/표 저장 디렉터리")
    ap.add_argument("--datasets", nargs="*", default=DATASETS_DEFAULT)
    ap.add_argument("--strict", action="store_true",
                    help="모든 데이터셋이 teacher/student 둘 다 있어야 진행(없으면 에러)")
    # optional sweep
    ap.add_argument("--sweep_student", nargs="*", default=[],
                    help="스윕용 student json 경로(여러 개 가능). 지정되면 fig_sweep_<ds> 생성")
    ap.add_argument("--sweep_datasets", nargs="*", default=[],
                    help="스윕 곡선을 만들 데이터셋 목록 (예: solar_nips traffic_nips)")

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    teach = _load_by_role(args.teacher, "teacher", args.datasets)
    stud  = _load_by_role(args.student,  "student", args.datasets)

    # report missing
    missing = []
    for ds in args.datasets:
        if ds not in teach:
            print(f"[WARN] teacher missing: {ds}")
        if ds not in stud:
            print(f"[WARN] student missing: {ds}")
        if ds not in teach or ds not in stud:
            missing.append(ds)
    if args.strict and missing:
        raise SystemExit(f"[ERROR] missing datasets: {', '.join(missing)}")

    order = [d for d in args.datasets if d in teach and d in stud]
    if not order:
        print("[WARN] no matched teacher/student datasets. Check paths.")
        return

    rows = compute_rows(teach, stud, order)
    if not rows:
        print("[WARN] no matched pairs after compute_rows.")
        return

    # Tables
    save_csv_tex(rows, args.out)

    # Absolute (log-scale) + error bars + small-range
    bar_crps_abs(rows, args.out, use_log=True,  fname="fig_crps_meanD_bar")
    bar_crps_abs_err(rows, args.out, use_log=True, fname="fig_crps_meanD_bar_err")
    bar_crps_abs_small(rows, args.out, exclude=("wiki-rolling_nips",), ymax=None, fname="fig_crps_meanD_bar_small")

    # Relative (ratio & delta) with labels
    bar_crps_ratio_labeled(rows, args.out, fname="fig_crps_ratio_bar_labeled")
    bar_crps_delta_labeled(rows, args.out, fname="fig_crps_delta_bar_labeled")

    # Speed/Params/Cost
    bar_speedup(rows, args.out, fname="fig_speedup_bar")
    bar_param_reduction(rows, args.out, fname="fig_param_reduction_bar")
    bar_total_time(rows, args.out, fname="fig_total_infer_time_bar")

    # Pareto scatter
    pareto_tradeoff(rows, args.out, fname="fig_tradeoff_pareto")

    # Optional sweeps
    if args.sweep_student and args.sweep_datasets:
        for ds in args.sweep_datasets:
            pareto_sweep(args.sweep_student, ds, args.out, fname=f"fig_sweep_{ds}")

    print(f"[OK] Figures & tables saved to: {args.out}")

if __name__ == "__main__":
    main()
