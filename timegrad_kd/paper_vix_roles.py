#!/usr/bin/env python3
import os, json, glob, argparse, time
import numpy as np
import matplotlib.pyplot as plt

DATASETS_DEFAULT = [
    "exchange_rate","solar_nips","electricity_nips",
    "traffic_nips","taxi_30min","wiki-rolling_nips"
]
# --- 절대값(Teacher vs Student) 막대: wiki 때문에 로그 스케일 옵션 지원 ---
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
    plt.legend()
    plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(outdir, f"{fname}.{ext}"), dpi=200 if ext=="png" else None)
    plt.close()

# --- 절대값(작은 값 전용) : wiki 제외 또는 y축 상한 지정 ---
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
    plt.legend()
    plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(outdir, f"{fname}.{ext}"), dpi=200 if ext=="png" else None)
    plt.close()

# --- 상대값(비율) : S/T (lower is better) ---
def bar_crps_ratio(rows, outdir, fname="fig_crps_ratio_bar"):
    labels = [r["ds"] for r in rows]
    x = np.arange(len(rows))
    ratio = [r["crpsS"]/max(r["crpsT"],1e-9) for r in rows]
    plt.figure(figsize=(10,4))
    plt.bar(x, ratio)
    plt.axhline(1.0, ls="--", lw=1, color="gray")
    plt.ylabel("CRPS_meanD ratio (Student / Teacher, lower is better)")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(outdir, f"{fname}.{ext}"), dpi=200 if ext=="png" else None)
    plt.close()

# --- 상대값(퍼센트 변화) : Δ% = (S/T−1)*100 (lower is better) ---
def bar_crps_delta(rows, outdir, fname="fig_crps_delta_bar"):
    labels = [r["ds"] for r in rows]
    x = np.arange(len(rows))
    dperc = [ (r["crpsS"]/max(r["crpsT"],1e-9) - 1.0)*100.0 for r in rows ]
    plt.figure(figsize=(10,4))
    plt.bar(x, dperc)
    plt.axhline(0.0, ls="--", lw=1, color="gray")
    plt.ylabel("Δ CRPS_meanD (%) vs Teacher (lower is better)")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(outdir, f"{fname}.{ext}"), dpi=200 if ext=="png" else None)
    plt.close()

def _iter_paths(paths):
    for p in paths:
        if os.path.isdir(p):
            for fp in glob.glob(os.path.join(p, "**", "*.json"), recursive=True):
                yield fp
        else:
            # 파일/글롭 패턴도 허용
            for fp in glob.glob(p):
                if fp.endswith(".json"): yield fp

def _load_by_role(paths, role, datasets):
    """paths: list[str] (dir/file/glob). role: 'teacher'|'student'."""
    bucket = {ds: [] for ds in datasets}
    for fp in _iter_paths(paths):
        try:
            with open(fp, "r") as f: j = json.load(f)
        except Exception:
            continue
        ds = j.get("dataset")
        if ds not in bucket:  # 파일명 프리픽스로 유추
            base = os.path.basename(fp)
            for d in datasets:
                if base.startswith(d):
                    ds = d; break
        if ds not in bucket: 
            continue
        r = j.get("model_role")
        # 파일명에 역할이 표시돼 있으면 보조 검증
        name_has_role = ("teacher" in os.path.basename(fp).lower(), "student" in os.path.basename(fp).lower())
        if r not in ("teacher","student"):
            # model_role이 없으면 파일명으로 추정
            if role=="teacher" and name_has_role[0]: pass
            elif role=="student" and name_has_role[1]: pass
            else: 
                # 역할 모호 → 건너뜀
                continue
        else:
            if r != role: 
                continue
        mtime = os.path.getmtime(fp)
        bucket[ds].append((mtime, fp, j))
    # 각 데이터셋별로 최신 파일 선택
    picked = {}
    for ds, lst in bucket.items():
        if not lst: 
            continue
        lst.sort(key=lambda x: x[0], reverse=True)
        picked[ds] = lst[0][2]  # json
    return picked

def _get(m, key, default=np.nan):
    try: return float(m[key])
    except Exception: return default

def compute_rows(teach_by_ds, stud_by_ds, order):
    rows = []
    for ds in order:
        t = teach_by_ds.get(ds); s = stud_by_ds.get(ds)
        if not t or not s: 
            continue
        crpsT = _get(t, "crps_meanD_mean")
        crpsS = _get(s, "crps_meanD_mean")
        timeT = _get(t, "infer_time_sec_mean")
        timeS = _get(s, "infer_time_sec_mean")
        parT  = int(t.get("params", 0))
        parS  = int(s.get("params", 0))
        speedup = timeT / max(timeS, 1e-9) if (timeT>0 and timeS>0) else np.nan
        par_red = (1 - parS/max(parT,1)) * 100.0 if parT>0 else np.nan
        dcrps_pct = (crpsS/max(crpsT,1e-9) - 1) * 100.0 if crpsT>0 else np.nan
        rows.append({
            "ds": ds,
            "D": int(s.get("D", t.get("D", 0))),
            "pred_len": int(s.get("pred_len", t.get("pred_len", 0))),
            "parT": parT, "parS": parS, "par_red_pct": par_red,
            "crpsT": crpsT, "crpsS": crpsS, "d_crps_pct": dcrps_pct,
            "timeT": timeT, "timeS": timeS, "speedup": speedup,
            "windows": int(s.get("windows", t.get("windows", 0)))
        })
    return rows

def save_csv_tex(rows, outdir):
    import csv
    csv_path = os.path.join(outdir, "table_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset","Params(T)","Params(S)","↓Params%","CRPS_meanD(T)","CRPS_meanD(S)","ΔS−T",
                    "Time(T)","Time(S)","Speedup(T/S)","Windows"])
        for r in rows:
            w.writerow([r["ds"], r["parT"], r["parS"], f"{r['par_red_pct']:.1f}%",
                        f"{r['crpsT']:.6f}", f"{r['crpsS']:.6f}", f"{r['crpsS']-r['crpsT']:+.6f}",
                        f"{r['timeT']:.3f}", f"{r['timeS']:.3f}", f"{r['speedup']:.2f}", r["windows"]])
    tex_path = os.path.join(outdir, "table_results.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n")
        f.write("Dataset & Par(T) & Par(S) & $\\downarrow$Par\\% & CRPS(T) & CRPS(S) & $\\Delta$ & Time(T) & Time(S) & Speedup\\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['ds']} & {r['parT']:,} & {r['parS']:,} & {r['par_red_pct']:.1f}\\% & "
                    f"{r['crpsT']:.6f} & {r['crpsS']:.6f} & {r['crpsS']-r['crpsT']:+.6f} & "
                    f"{r['timeT']:.3f} & {r['timeS']:.3f} & {r['speedup']:.2f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[OK] saved: {csv_path}, {tex_path}")

def bar(rows, outdir, values, ylabel, fname):
    labels = [r["ds"] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    plt.figure(figsize=(10,4))
    if isinstance(values[0], tuple):
        A = [v[0] for v in values]; B = [v[1] for v in values]
        plt.bar(x-w/2, A, width=w, label="Teacher")
        plt.bar(x+w/2, B, width=w, label="Student")
        plt.legend()
    else:
        plt.bar(x, values)
    plt.ylabel(ylabel)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(outdir, f"{fname}.{ext}"), dpi=200 if ext=="png" else None)
    plt.close()

def pareto(rows, outdir):
    x = [r["speedup"] for r in rows]
    y = [r["d_crps_pct"] for r in rows]
    sizes = [max(50, r["par_red_pct"]*5) for r in rows]
    labels = [r["ds"] for r in rows]
    plt.figure(figsize=(6,5))
    plt.scatter(x, y, s=sizes, alpha=0.8)
    for xi, yi, lab in zip(x,y,labels):
        plt.text(xi*1.01, yi, lab, fontsize=9)
    plt.axhline(0, ls="--", lw=1); plt.axvline(1, ls="--", lw=1)
    plt.xlabel("Speedup (×, higher is better)")
    plt.ylabel("ΔCRPS_meanD (%) vs Teacher (lower is better)")
    plt.tight_layout()
    for ext in ("png","pdf"):
        plt.savefig(os.path.join(outdir, f"fig_tradeoff_pareto.{ext}"), dpi=200 if ext=="png" else None)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", nargs="+", required=True,
                    help="teacher json 경로(들). 디렉터리/파일/글롭 혼용 가능, 재귀 탐색")
    ap.add_argument("--student", nargs="+", required=True,
                    help="student json 경로(들). 디렉터리/파일/글롭 혼용 가능, 재귀 탐색")
    ap.add_argument("--out", required=True, help="그림/표 저장 디렉터리")
    ap.add_argument("--datasets", nargs="*", default=DATASETS_DEFAULT)
    ap.add_argument("--strict", action="store_true",
                    help="모든 데이터셋이 teacher/student 둘 다 있어야 진행(없으면 에러)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    teach = _load_by_role(args.teacher, "teacher", args.datasets)
    stud  = _load_by_role(args.student,  "student", args.datasets)

    # 매칭 리포트
    missing = []
    for ds in args.datasets:
        if ds not in teach: print(f"[WARN] teacher missing: {ds}")
        if ds not in stud:  print(f"[WARN] student missing: {ds}")
        if ds not in teach or ds not in stud:
            missing.append(ds)
    if args.strict and missing:
        raise SystemExit(f"[ERROR] missing datasets: {', '.join(missing)}")

    order = [d for d in args.datasets if d in teach and d in stud]
    rows = compute_rows(teach, stud, order)
    if not rows:
        print("[WARN] no matched pairs; check --teacher/--student paths.")
        return

    save_csv_tex(rows, args.out)
    # bar(rows, args.out, [(r["crpsT"], r["crpsS"]) for r in rows], "CRPS_meanD (lower is better)", "fig_crps_meanD_bar")
    # bar(rows, args.out, [r["speedup"] for r in rows], "Speedup (Teacher/Student)", "fig_speedup_bar")
    # bar(rows, args.out, [r["par_red_pct"] for r in rows], "Param reduction (%)", "fig_param_reduction_bar")
    # pareto(rows, args.out)

    # 절대값(Teacher vs Student)
    bar_crps_abs(rows, args.out, use_log=True,  fname="fig_crps_meanD_bar")        # 로그 스케일
    bar_crps_abs_small(rows, args.out, exclude=("wiki-rolling_nips",), ymax=None)  # 작은 값 전용

    # 상대값(비율/퍼센트) — 데이터셋 간 스케일 공정 비교
    bar_crps_ratio(rows, args.out, fname="fig_crps_ratio_bar")      # S/T (↓가 좋음)
    bar_crps_delta(rows, args.out, fname="fig_crps_delta_bar")      # Δ% (↓가 좋음)

    # 기존 파레토/스피드/파라메타 감소 그림도 유지
    bar(rows, args.out, [r["speedup"] for r in rows], "Speedup (Teacher/Student)", "fig_speedup_bar")
    bar(rows, args.out, [r["par_red_pct"] for r in rows], "Param reduction (%)", "fig_param_reduction_bar")
    pareto(rows, args.out)
    print(f"[OK] Figures & tables saved to: {args.out}")

if __name__ == "__main__":
    main()
