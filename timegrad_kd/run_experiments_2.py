# # run_experiments.py
# import argparse, os, subprocess, json, csv
# from datetime import datetime
# from typing import List, Dict
# from configs import DATASETS

# RESULT_HEADER = [
#     "dataset", "model", "params",
#     "crps_sum_mean", "crps_sum_std",
#     "infer_time_sec_mean", "infer_time_sec_std",
#     "windows",
# ]

# def ensure_dir(p): os.makedirs(p, exist_ok=True)

# def run_live(cmd: List[str]):
#     """학습용: 자식 프로세스 stdout/stderr를 그대로 터미널에 스트리밍."""
#     print(">>", " ".join(cmd), flush=True)
#     env = os.environ.copy()
#     env["PYTHONUNBUFFERED"] = "1"   # 자식 파이썬 즉시 flush
#     r = subprocess.run(cmd, env=env)
#     if r.returncode != 0:
#         raise RuntimeError(f"Command failed: {' '.join(cmd)}")

# def run_capture(cmd: List[str]) -> str:
#     """
#     평가용: 실시간으로 출력하면서 전체 텍스트를 반환.
#     마지막 JSON 블록을 파싱하기 위해 전체 로그를 돌려준다.
#     """
#     print(">>", " ".join(cmd), flush=True)
#     env = os.environ.copy()
#     env["PYTHONUNBUFFERED"] = "1"
#     p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
#                          text=True, bufsize=1, env=env)
#     out_lines: List[str] = []
#     try:
#         for line in iter(p.stdout.readline, ''):
#             print(line, end='')     # 터미널로 즉시 표시
#             out_lines.append(line)
#     finally:
#         if p.stdout:
#             p.stdout.close()
#     ret = p.wait()
#     if ret != 0:
#         raise RuntimeError(f"Command failed: {' '.join(cmd)}")
#     return "".join(out_lines)

# def _extract_last_json_block(text: str) -> str:
#     """출력 전체에서 마지막 JSON 객체 { ... }만 정확히 잘라낸다."""
#     end = text.rfind("}")
#     if end == -1:
#         raise RuntimeError("No closing '}' found in output.")
#     depth = 0
#     i = end
#     while i >= 0:
#         c = text[i]
#         if c == "}":
#             depth += 1
#         elif c == "{":
#             depth -= 1
#             if depth == 0:
#                 return text[i:end+1]
#         i -= 1
#     raise RuntimeError("No matching '{' for the last JSON object.")

# def eval_model(ckpt: str, dataset: str, device: str, num_samples: int, rolling: bool = True) -> Dict:
#     cmd = ["python", "evaluate.py", "--dataset", dataset, "--ckpt", ckpt,
#            "--device", device, "--num_samples", str(num_samples)]
#     if rolling:
#         cmd.append("--rolling")
#     out_txt = run_capture(cmd)      # 실시간 출력 + 전체 문자열 반환
#     json_str = _extract_last_json_block(out_txt)
#     return json.loads(json_str)

# def write_csv_row(csv_path: str, row: List):
#     write_header = not os.path.exists(csv_path)
#     with open(csv_path, "a", newline="") as f:
#         w = csv.writer(f)
#         if write_header:
#             w.writerow(RESULT_HEADER)
#         w.writerow(row)

# def to_md_table(rows: List[Dict]) -> str:
#     datasets = sorted(set(r["dataset"] for r in rows))
#     md = []
#     md.append("| Dataset | Params (T) | Params (S) | ↓Params% | CRPS_sum (T) | CRPS_sum (S) | ΔS−T | Inference s (T) | Inference s (S) | ↓Time% | Windows |")
#     md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
#     for ds in datasets:
#         t = next(r for r in rows if r["dataset"]==ds and r["model"]=="teacher")
#         s = next(r for r in rows if r["dataset"]==ds and r["model"]=="student")
#         params_t, params_s = t["params"], s["params"]
#         crps_t, crps_s = t["crps_sum_mean"], s["crps_sum_mean"]
#         time_t, time_s = t["infer_time_sec_mean"], s["infer_time_sec_mean"]
#         windows = t["windows"]
#         p_red = 100.0*(1.0 - params_s/max(params_t,1))
#         t_red = 100.0*(1.0 - time_s/max(time_t,1e-12))
#         md.append(f"| {ds} | {params_t:,} | {params_s:,} | {p_red:.1f}% | {crps_t:.4f} | {crps_s:.4f} | {crps_s-crps_t:+.4f} | {time_t:.3f} | {time_s:.3f} | {t_red:.1f}% | {windows} |")
#     return "\n".join(md)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
#     ap.add_argument("--device", default="cuda")
#     ap.add_argument("--epochs_teacher", type=int, default=50)
#     ap.add_argument("--epochs_student", type=int, default=50)
#     ap.add_argument("--num_samples", type=int, default=100)
#     ap.add_argument("--skip_train", action="store_true")
#     ap.add_argument("--results_dir", default="results")
#     args = ap.parse_args()

#     stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#     out_dir = os.path.join(args.results_dir, stamp)
#     ensure_dir(out_dir)

#     all_rows: List[Dict] = []

#     for ds in args.datasets:
#         print(f"\n==== [{ds}] START ====", flush=True)

#         teacher_dir = os.path.join("checkpoints", "teacher", ds)
#         student_dir = os.path.join("checkpoints", "student", ds)
#         teacher_ckpt = os.path.join(teacher_dir, "teacher.pt")
#         student_ckpt = os.path.join(student_dir, "student.pt")
#         ensure_dir(teacher_dir); ensure_dir(student_dir)

#         if not args.skip_train:
#             run_live(["python", "train_teacher.py", "--dataset", ds,
#                       "--epochs", str(args.epochs_teacher),
#                       "--device", args.device, "--save_dir", teacher_dir])
#             run_live(["python", "train_student.py", "--dataset", ds,
#                       "--epochs", str(args.epochs_student),
#                       "--device", args.device,
#                       "--teacher_ckpt", teacher_ckpt, "--save_dir", student_dir])

#         print(f"\n-- [{ds}] EVAL: teacher --", flush=True)
#         t_metrics = eval_model(teacher_ckpt, ds, args.device, args.num_samples, rolling=True)
#         print(f"\n-- [{ds}] EVAL: student --", flush=True)
#         s_metrics = eval_model(student_ckpt, ds, args.device, args.num_samples, rolling=True)

#         # CSV 저장
#         csv_path = os.path.join(out_dir, "summary.csv")
#         t_row = [ds, "teacher", t_metrics["params"], t_metrics["crps_sum_mean"], t_metrics["crps_sum_std"],
#                  t_metrics["infer_time_sec_mean"], t_metrics.get("infer_time_sec_std", 0.0), t_metrics["windows"]]
#         s_row = [ds, "student", s_metrics["params"], s_metrics["crps_sum_mean"], s_metrics["crps_sum_std"],
#                  s_metrics["infer_time_sec_mean"], s_metrics.get("infer_time_sec_std", 0.0), s_metrics["windows"]]
#         write_csv_row(csv_path, t_row)
#         write_csv_row(csv_path, s_row)

#         # JSON 저장
#         with open(os.path.join(out_dir, f"{ds}_teacher.json"), "w") as f: json.dump(t_metrics, f, indent=2)
#         with open(os.path.join(out_dir, f"{ds}_student.json"), "w") as f: json.dump(s_metrics, f, indent=2)

#         all_rows.append({"dataset": ds, "model": "teacher", **t_metrics})
#         all_rows.append({"dataset": ds, "model": "student", **s_metrics})
#         print(f"==== [{ds}] DONE ====\n", flush=True)

#     md = to_md_table(all_rows)
#     with open(os.path.join(out_dir, "summary.md"), "w") as f:
#         f.write(md + "\n")

#     print("\n=== Summary (markdown) ===\n")
#     print(md)
#     print(f"\nSaved to: {out_dir}")

# if __name__ == "__main__":
#     main()
# run_experiments.py
import argparse, os, subprocess, json, csv, time
from datetime import datetime
from typing import List, Dict
from configs import DATASETS

RESULT_HEADER = [
    "dataset", "model", "params",
    "crps_sum_mean", "crps_sum_std",
    "infer_time_sec_mean", "infer_time_sec_std",
    "windows",
    "hparams",  
]

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def run_live(cmd: List[str], gpu_id: str = None):
    print(">>", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU] set CUDA_VISIBLE_DEVICES={gpu_id}", flush=True)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def run_capture(cmd: List[str], gpu_id: str = None) -> str:
    print(">>", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU] set CUDA_VISIBLE_DEVICES={gpu_id}", flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=env)
    out_lines: List[str] = []
    try:
        for line in iter(p.stdout.readline, ''):
            print(line, end='')
            out_lines.append(line)
    finally:
        if p.stdout:
            p.stdout.close()
    ret = p.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return "".join(out_lines)

def _extract_last_json_block(text: str) -> str:
    end = text.rfind("}")
    if end == -1: raise RuntimeError("No closing '}' found in output.")
    depth = 0; i = end
    while i >= 0:
        c = text[i]
        if c == "}": depth += 1
        elif c == "{":
            depth -= 1
            if depth == 0:
                return text[i:end+1]
        i -= 1
    raise RuntimeError("No matching '{' for the last JSON object.")

def eval_model(ckpt: str, dataset: str, device: str, num_samples: int,
               rolling: bool = True, max_windows: int = None, batch_windows: int = 1,
               gpu_id: str = None) -> Dict:
    cmd = ["python", "evaluate_2.py", "--dataset", dataset, "--ckpt", ckpt,
           "--device", device, "--num_samples", str(num_samples)]
    if rolling: cmd.append("--rolling")
    if max_windows is not None: cmd += ["--max_windows", str(max_windows)]
    if batch_windows is not None and batch_windows > 1:
        cmd += ["--batch_windows", str(batch_windows)]
    out_txt = run_capture(cmd, gpu_id=gpu_id)
    json_str = _extract_last_json_block(out_txt)
    return json.loads(json_str)

def write_csv_row(csv_path: str, row: List):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(RESULT_HEADER)
        w.writerow(row)

def to_md_table(rows: List[Dict]) -> str:
    datasets = sorted(set(r["dataset"] for r in rows))
    md = []
    md.append("| Dataset | Params (T) | Params (S) | ↓Params% | CRPS_sum (T) | CRPS_sum (S) | ΔS−T | Inference s (T) | Inference s (S) | ↓Time% | Windows |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ds in datasets:
        t = next(r for r in rows if r["dataset"]==ds and r["model"]=="teacher")
        s = next(r for r in rows if r["dataset"]==ds and r["model"]=="student")
        params_t, params_s = t["params"], s["params"]
        crps_t, crps_s = t["crps_sum_mean"], s["crps_sum_mean"]
        time_t, time_s = t["infer_time_sec_mean"], s["infer_time_sec_mean"]
        windows = t["windows"]
        p_red = 100.0*(1.0 - params_s/max(params_t,1))
        t_red = 100.0*(1.0 - time_s/max(time_t,1e-12))
        md.append(f"| {ds} | {params_t:,} | {params_s:,} | {p_red:.1f}% | {crps_t:.4f} | {crps_s:.4f} | {crps_s-crps_t:+.4f} | {time_t:.3f} | {time_s:.3f} | {t_red:.1f}% | {windows} |")
    return "\n".join(md)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs_teacher", type=int, default=50)
    ap.add_argument("--epochs_student", type=int, default=50)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--max_windows", type=int, default=None, help="롤링 평가 윈도우 제한(디버그)")
    ap.add_argument("--batch_windows", type=int, default=1, help="윈도우 배치 크기(B)")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--gpus", default=None, help="예: '0,1,2,3' (없으면 현재 설정 유지)")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--skip_teacher_train", action="store_true",
                help="teacher 학습 스킵(ckpt만 사용)")
    ap.add_argument("--skip_teacher_eval", action="store_true",
                    help="teacher 평가 스킵(JSON 있으면 그대로 사용)")
    ap.add_argument("--reuse_teacher_metrics", action="store_true",
                    help="teacher 평가 JSON이 있으면 재평가 대신 재사용")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",")] if args.gpus else [None]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # out_dir = os.path.join(args.results_dir, stamp)
    out_dir = args.results_dir if args.skip_train else os.path.join(args.results_dir, stamp)

    ensure_dir(out_dir)

    all_rows: List[Dict] = []

    for idx, ds in enumerate(args.datasets):
        print(f"\n==== [{ds}] START ====", flush=True)

        teacher_dir = os.path.join(out_dir, "teacher", ds)
        student_dir = os.path.join(out_dir, "student", ds)
        teacher_ckpt = os.path.join(teacher_dir, "teacher.pt")
        student_ckpt = os.path.join(student_dir, "student.pt")

        ensure_dir(teacher_dir); ensure_dir(student_dir)

        # 학습은 기본적으로 순차 (원하면 여기도 GPU 분배 가능)
        # if not args.skip_train:
        #     run_live(["python", "train_teacher.py", "--dataset", ds,
        #               "--epochs", str(args.epochs_teacher),
        #               "--device", args.device, "--save_dir", teacher_dir],
        #              gpu_id=gpus[idx % len(gpus)])
        #     run_live(["python", "train_student.py", "--dataset", ds,
        #               "--epochs", str(args.epochs_student),
        #               "--device", args.device,
        #               "--teacher_ckpt", teacher_ckpt, "--save_dir", student_dir],
        #              gpu_id=gpus[idx % len(gpus)])

        if not args.skip_train and not args.skip_teacher_train:
            if not os.path.isfile(teacher_ckpt):
                run_live(["python","train_teacher_2.py","--dataset",ds,
                        "--epochs",str(args.epochs_teacher),
                        "--device",args.device,"--save_dir",teacher_dir],
                        gpu_id=gpus[idx % len(gpus)])
            else:
                print(f"[SKIP] teacher train: ckpt exists -> {teacher_ckpt}", flush=True)
        else:
            print("[SKIP] teacher train (flag)", flush=True)

        # STUDENT TRAIN (teacher_ckpt만 필요)
        if not args.skip_train:
            run_live(["python","train_student_2.py","--dataset",ds,
                    "--epochs",str(args.epochs_student),
                    "--device",args.device,"--teacher_ckpt",teacher_ckpt,
                    "--save_dir",student_dir],
                    gpu_id=gpus[idx % len(gpus)])

        # 평가는 GPU 분배 (teacher / student를 서로 다른 GPU에 배치 시 병렬 가능)
        gpu_t = gpus[idx % len(gpus)]
        gpu_s = gpus[(idx + 1) % len(gpus)] if len(gpus) > 1 else gpus[0]

        # print(f"\n-- [{ds}] EVAL: teacher --", flush=True)
        # t0 = time.time()
        # t_metrics = eval_model(teacher_ckpt, ds, args.device, args.num_samples,
        #                        rolling=True, max_windows=args.max_windows,
        #                        batch_windows=args.batch_windows, gpu_id=gpu_t)
        # print(f"[TIME] eval_teacher {ds}: {time.time()-t0:.1f}s", flush=True)
        teacher_json = os.path.join(out_dir, f"{ds}_teacher.json")

        # TEACHER EVAL (조건부 실행/재사용)
        if args.skip_teacher_eval:
            if args.reuse_teacher_metrics and os.path.isfile(teacher_json):
                with open(teacher_json) as f:
                    t_metrics = json.load(f)
                print(f"[SKIP] teacher eval (reuse metrics): {teacher_json}", flush=True)
            else:
                print("[SKIP] teacher eval (flag) — no JSON to reuse; evaluating once", flush=True)
                t0 = time.time()
                t_metrics = eval_model(
                    teacher_ckpt, ds, args.device, args.num_samples,
                    rolling=True, max_windows=args.max_windows,
                    batch_windows=args.batch_windows, gpu_id=gpu_t
                )
                print(f"[TIME] eval_teacher {ds}: {time.time()-t0:.1f}s", flush=True)
        else:
            print(f"\n-- [{ds}] EVAL: teacher --", flush=True)
            t0 = time.time()
            t_metrics = eval_model(
                teacher_ckpt, ds, args.device, args.num_samples,
                rolling=True, max_windows=args.max_windows,
                batch_windows=args.batch_windows, gpu_id=gpu_t
            )
            print(f"[TIME] eval_teacher {ds}: {time.time()-t0:.1f}s", flush=True)

        print(f"\n-- [{ds}] EVAL: student --", flush=True)
        t0 = time.time()
        s_metrics = eval_model(student_ckpt, ds, args.device, args.num_samples,
                               rolling=True, max_windows=args.max_windows,
                               batch_windows=args.batch_windows, gpu_id=gpu_s)
        print(f"[TIME] eval_student {ds}: {time.time()-t0:.1f}s", flush=True)

        # CSV 저장
        csv_path = os.path.join(out_dir, "summary.csv")
        t_row = [ds, "teacher", t_metrics["params"], t_metrics["crps_sum_mean"], t_metrics["crps_sum_std"],
                 t_metrics["infer_time_sec_mean"], t_metrics.get("infer_time_sec_std", 0.0), t_metrics["windows"],
                 json.dumps(t_metrics.get("model_cfg", {}), ensure_ascii=False)]
        s_row = [ds, "student", s_metrics["params"], s_metrics["crps_sum_mean"], s_metrics["crps_sum_std"],
                 s_metrics["infer_time_sec_mean"], s_metrics.get("infer_time_sec_std", 0.0), s_metrics["windows"],
                 json.dumps(s_metrics.get("model_cfg", {}), ensure_ascii=False)]
        write_csv_row(csv_path, t_row); write_csv_row(csv_path, s_row)

        # JSON 저장
        with open(os.path.join(out_dir, f"{ds}_teacher.json"), "w") as f: json.dump(t_metrics, f, indent=2)
        with open(os.path.join(out_dir, f"{ds}_student.json"), "w") as f: json.dump(s_metrics, f, indent=2)

        all_rows.append({"dataset": ds, "model": "teacher", **t_metrics})
        all_rows.append({"dataset": ds, "model": "student", **s_metrics})
        print(f"==== [{ds}] DONE ====\n", flush=True)

    # MD 요약
    def to_md_table(rows: List[Dict]) -> str:
        datasets = sorted(set(r["dataset"] for r in rows))
        md = []
        md.append("| Dataset | Params (T) | Params (S) | ↓Params% | CRPS_sum (T) | CRPS_sum (S) | ΔS−T | Inference s (T) | Inference s (S) | ↓Time% | Windows |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for ds in datasets:
            t = next(r for r in rows if r["dataset"]==ds and r["model"]=="teacher")
            s = next(r for r in rows if r["dataset"]==ds and r["model"]=="student")
            params_t, params_s = t["params"], s["params"]
            crps_t, crps_s = t["crps_sum_mean"], s["crps_sum_mean"]
            time_t, time_s = t["infer_time_sec_mean"], s["infer_time_sec_mean"]
            windows = t["windows"]
            p_red = 100.0*(1.0 - params_s/max(params_t,1))
            t_red = 100.0*(1.0 - time_s/max(time_t,1e-12))
            md.append(f"| {ds} | {params_t:,} | {params_s:,} | {p_red:.1f}% | {crps_t:.4f} | {crps_s:.4f} | {crps_s-crps_t:+.4f} | {time_t:.3f} | {time_s:.3f} | {t_red:.1f}% | {windows} |")
        return "\n".join(md)

    md = to_md_table(all_rows)
    with open(os.path.join(out_dir, "summary.md"), "w") as f: f.write(md + "\n")
    print("\n=== Summary (markdown) ===\n"); print(md); print(f"\nSaved to: {out_dir}")

if __name__ == "__main__":
    main()
