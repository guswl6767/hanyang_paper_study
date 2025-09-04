# run_experiments.py
import argparse, os, subprocess, json, time, csv
from datetime import datetime
from typing import List, Dict
from configs import DATASETS

RESULT_HEADER = [
    "dataset", "model", "params", 
    "crps_sum_mean", "crps_sum_std", 
    "infer_time_sec_mean", "infer_time_sec_std", 
    "windows"
]

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def run(cmd: List[str]):
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return r

def eval_model(ckpt: str, dataset: str, device: str, num_samples: int, rolling: bool = True) -> Dict:
    cmd = ["python", "evaluate.py", "--dataset", dataset, "--ckpt", ckpt, "--device", device, "--num_samples", str(num_samples)]
    if rolling:
        cmd.append("--rolling")
    r = run(cmd)
    # parse last JSON object from stdout
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip().startswith("{")]
    out = json.loads(lines[-1])
    return out

def write_csv_row(csv_path: str, row: List):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(RESULT_HEADER)
        w.writerow(row)

def to_md_table(rows: List[Dict]) -> str:
    # rows: [{"dataset":..,"model":"teacher"...}, {"dataset":..,"model":"student"...}, ...]
    # Make paired table per dataset with deltas
    datasets = sorted(set(r["dataset"] for r in rows))
    md = []
    md.append("| Dataset | Params (T) | Params (S) | ↓Params% | CRPS_sum (T) | CRPS_sum (S) | ΔS−T | Inference s (T) | Inference s (S) | ↓Time% | Windows |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ds in datasets:
        t = next(r for r in rows if r["dataset"]==ds and r["model"]=="teacher")
        s = next(r for r in rows if r["dataset"]==ds and r["model"]=="student")
        params_t = t["params"]; params_s = s["params"]
        crps_t = t["crps_sum_mean"]; crps_s = s["crps_sum_mean"]
        time_t = t["infer_time_sec_mean"]; time_s = s["infer_time_sec_mean"]
        windows = t["windows"]  # same as s
        p_red = 100.0*(1.0 - params_s/max(params_t,1))
        t_red = 100.0*(1.0 - time_s/max(time_t,1e-9))
        md.append(f"| {ds} | {params_t:,} | {params_s:,} | {p_red:.1f}% | {crps_t:.4f} | {crps_s:.4f} | {crps_s-crps_t:+.4f} | {time_t:.3f} | {time_s:.3f} | {t_red:.1f}% | {windows} |")
    return "\n".join(md)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()),
                    help="기본: 논문 6종 데이터셋 전부")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs_teacher", type=int, default=50)
    ap.add_argument("--epochs_student", type=int, default=50)
    ap.add_argument("--num_samples", type=int, default=100, help="평가 시 샘플 개수")
    ap.add_argument("--skip_train", action="store_true", help="이미 학습된 ckpt를 사용")
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.results_dir, stamp)
    ensure_dir(out_dir)

    all_rows = []

    for ds in args.datasets:
        # paths
        teacher_dir = os.path.join("checkpoints", "teacher", ds)
        student_dir = os.path.join("checkpoints", "student", ds)
        teacher_ckpt = os.path.join(teacher_dir, "teacher.pt")
        student_ckpt = os.path.join(student_dir, "student.pt")
        ensure_dir(teacher_dir); ensure_dir(student_dir)

        # Train
        if not args.skip_train:
            run(["python", "train_teacher.py", "--dataset", ds, "--epochs", str(args.epochs_teacher),
                 "--device", args.device, "--save_dir", teacher_dir])
            run(["python", "train_student.py", "--dataset", ds, "--epochs", str(args.epochs_student),
                 "--device", args.device, "--teacher_ckpt", teacher_ckpt, "--save_dir", student_dir])

        # Evaluate (rolling)
        t_metrics = eval_model(teacher_ckpt, ds, args.device, args.num_samples, rolling=True)
        s_metrics = eval_model(student_ckpt, ds, args.device, args.num_samples, rolling=True)

        # write per-model csv
        t_row = [ds, "teacher", t_metrics["params"], t_metrics["crps_sum_mean"], t_metrics["crps_sum_std"],
                 t_metrics["infer_time_sec_mean"], t_metrics.get("infer_time_sec_std", 0.0), t_metrics["windows"]]
        s_row = [ds, "student", s_metrics["params"], s_metrics["crps_sum_mean"], s_metrics["crps_sum_std"],
                 s_metrics["infer_time_sec_mean"], s_metrics.get("infer_time_sec_std", 0.0), s_metrics["windows"]]
        csv_path = os.path.join(out_dir, "summary.csv")
        write_csv_row(csv_path, t_row)
        write_csv_row(csv_path, s_row)

        # also JSON dumps per dataset
        with open(os.path.join(out_dir, f"{ds}_teacher.json"), "w") as f: json.dump(t_metrics, f, indent=2)
        with open(os.path.join(out_dir, f"{ds}_student.json"), "w") as f: json.dump(s_metrics, f, indent=2)

        all_rows.append({"dataset": ds, "model": "teacher", **t_metrics})
        all_rows.append({"dataset": ds, "model": "student", **s_metrics})

    # Markdown summary
    md = to_md_table(all_rows)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md + "\n")

    print("\n=== Summary (markdown) ===\n")
    print(md)
    print(f"\nSaved to: {out_dir}")

if __name__ == "__main__":
    main()
