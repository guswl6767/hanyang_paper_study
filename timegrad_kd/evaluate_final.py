# evaluate.py
import argparse, time, json
import os, platform, datetime

import numpy as np
import torch
from typing import Dict, List, Tuple
from configs import DATASETS
from data import load_multivariate, scale_by_context_mean
from model_timegrad import TimeGrad
from metrics import crps_sum_over_horizon,crps_mean_over_horizon


def _env_info(device: str):
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_deterministic": getattr(torch.backends.cudnn, "deterministic", None),
        "device": device,
    }
    try:
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda
    except Exception:
        pass
    return info

def count_params(m): return sum(p.numel() for p in m.parameters())
@torch.no_grad()
def _infer_student_cfg_from_state(state_model: Dict, D_model: int):
    sd = state_model
    # residual_channels: eps_net.c_in.weight -> (C_out, 1, 1)
    rc = sd["eps_net.c_in.weight"].shape[0]
    # residual_blocks: eps_net.blocks.<i>.skip.weight 의 i 최대값 + 1
    import re
    idxs = []
    for k in sd.keys():
        m = re.match(r"eps_net\.blocks\.(\d+)\.skip\.weight", k)
        if m: idxs.append(int(m.group(1)))
    rb = (max(idxs) + 1) if idxs else 4
    # lstm_hidden: rnn.weight_hh_l0 -> (4H, H)
    H = sd["rnn.weight_hh_l0"].shape[0] // 4
    # noise_emb_dim은 보통 32, n_steps는 ckpt에 없으면 디폴트(20 사용)
    return dict(lstm_hidden=H, lstm_layers=2, noise_emb_dim=32,
                residual_channels=rc, residual_blocks=rb,
                n_steps=20, beta_start=1e-4, beta_end=1e-1)

@torch.no_grad()
def _build_model_from_ckpt(state: Dict, D_infer: int, device: str):
    D_model = int(state.get("D", D_infer))
    if "cfg" in state:
        # teacher(또는 cfg 저장된 student)는 cfg대로 복원
        cfg = state["cfg"]
        model = TimeGrad(
            D=D_model,
            lstm_hidden=cfg.get("lstm_hidden", 40),
            lstm_layers=cfg.get("lstm_layers", 2),
            noise_emb_dim=cfg.get("noise_emb_dim", 32),
            residual_channels=cfg.get("residual_channels", 8),
            residual_blocks=cfg.get("residual_blocks", 8),
            n_steps=cfg.get("n_diffusion_steps", cfg.get("n_steps", 100)),
            beta_start=cfg.get("beta_start", 1e-4),
            beta_end=cfg.get("beta_end", 1e-1),
        ).to(device)
        model.load_state_dict(state["model"])   # strict=True
        model.eval()
        return model, D_model
    else:
        # cfg가 없으면(대부분 student) state_dict에서 구조를 추론
        stu_cfg = _infer_student_cfg_from_state(state["model"], D_model)
        model = TimeGrad(
            D=D_model,
            lstm_hidden=stu_cfg["lstm_hidden"],
            lstm_layers=stu_cfg["lstm_layers"],
            noise_emb_dim=stu_cfg["noise_emb_dim"],
            residual_channels=stu_cfg["residual_channels"],
            residual_blocks=stu_cfg["residual_blocks"],
            n_steps=stu_cfg["n_steps"],
            beta_start=stu_cfg["beta_start"],
            beta_end=stu_cfg["beta_end"],
        ).to(device)
        model.load_state_dict(state["model"])   # 구조를 맞췄으므로 strict 로드 가능
        model.eval()
        return model, D_model
# @torch.no_grad()
# def _build_model_from_ckpt(state: Dict, D_infer: int, device: str):
#     D_model = int(state.get("D", D_infer))
#     if "cfg" in state:  # teacher
#         cfg = state["cfg"]
#         model = TimeGrad(
#             D=D_model,
#             lstm_hidden=cfg["lstm_hidden"], lstm_layers=cfg["lstm_layers"],
#             noise_emb_dim=cfg["noise_emb_dim"],
#             residual_channels=cfg["residual_channels"], residual_blocks=cfg["residual_blocks"],
#             n_steps=cfg["n_diffusion_steps"], beta_start=cfg["beta_start"], beta_end=cfg["beta_end"]
#         ).to(device)
#     else:  # student
#         model = TimeGrad(
#             D=D_model, lstm_hidden=24, lstm_layers=2,
#             noise_emb_dim=32, residual_channels=4, residual_blocks=4,
#             n_steps=10, beta_start=1e-4, beta_end=1e-1
#         ).to(device)
#     model.load_state_dict(state["model"])
#     model.eval()
#     return model, D_model

# (B) 함수 추가
def _seed_all(s: int):
    import random, numpy as np, torch
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _force_2d(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 2:
        raise ValueError(f"target must be 2D, got shape={arr.shape}")
    return arr

def _orient_to_DT(arr: np.ndarray, D_model: int) -> np.ndarray:
    """(D,T) 또는 (T,D*k) 등 무엇이 와도 (D*k, T)로 맞춤."""
    arr = _force_2d(arr)
    r, c = arr.shape
    if r % D_model == 0 and r >= D_model:
        return arr  # (D*k, T)
    if c % D_model == 0 and c >= D_model:
        return arr.T  # -> (D*k, T)
    raise ValueError(f"target shape {arr.shape} incompatible with model D={D_model}")

def _iter_ctx_fut_slices(target: np.ndarray, D_model: int, context_len: int, pred_len: int):
    DT = _orient_to_DT(target, D_model)          # (D*k, T)
    Dk, T = DT.shape
    if T < context_len + pred_len: return []
    k = Dk // D_model
    out = []
    C, P = context_len, pred_len
    for i in range(k):
        sl = DT[i*D_model:(i+1)*D_model, :]      # (D, T)
        ctx = sl[:, T-(C+P):T-P]
        fut = sl[:, T-P:T]
        out.append((ctx.astype(np.float32), fut.astype(np.float32)))
    return out

def _ctx_to_tensor(ctx_np: np.ndarray, D_model: int, device: str) -> torch.Tensor:
    if ctx_np.shape[0] != D_model:
        raise ValueError(f"ctx first dim must be D={D_model}, got {ctx_np.shape}")
    return torch.tensor(ctx_np.T, device=device).unsqueeze(0)  # (1,C,D)

def _count_total_windows(test_mv, D_model: int, context_len: int, pred_len: int) -> int:
    total = 0
    for item in test_mv:
        total += len(_iter_ctx_fut_slices(item["target"], D_model, context_len, pred_len))
    return total

@torch.no_grad()
def run_eval(ckpt_path: str, dataset: str, device: str="cuda", num_samples: int=100):
    t_start = time.time()
    freq, pred_len = DATASETS[dataset]
    train_mv, test_mv, meta = load_multivariate(dataset)

    item0 = next(iter(train_mv))
    D_infer = int(item0["target"].shape[0])

    # 로드는 항상 CPU로 (CUDA 초기화 지연 회피)
    # state = torch.load(ckpt_path, map_location="cpu")
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # PyTorch 2.4+에서만 지원
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    print(f"[EVAL] ckpt loaded on CPU: {ckpt_path}", flush=True)
    model, D_model = _build_model_from_ckpt(state, D_infer, device)

    context_len = pred_len
    item = next(iter(test_mv))
    pairs = _iter_ctx_fut_slices(item["target"], D_model, context_len, pred_len)
    if not pairs: raise RuntimeError("No valid window in this test item.")

    ctx, fut = pairs[-1]
    ctx_s, fut_s, mean = scale_by_context_mean(ctx, fut)
    x_ctx = _ctx_to_tensor(ctx_s, D_model, device)

    t0 = time.time()
    samples = model.forecast(x_ctx, pred_len, num_samples=num_samples)[0]
    infer_time = time.time() - t0

    scale = torch.tensor(mean.T, device=device)
    samples = samples * scale
    crps = crps_sum_over_horizon(samples, torch.tensor(fut.T, device=device))
    crps_meanD = crps_mean_over_horizon(samples, torch.tensor(fut.T, device=device))
    # ★ cfg/role 추출
    model_role = state.get("role", "teacher" if "cfg" in state else "student")
    model_cfg = state.get("cfg", {})

    print(f"[EVAL] CRPS_sum={crps:.6f} | CRPS_meanD={crps_meanD:.6f}", flush=True)
    t_end = time.time()
    env = _env_info(device)
    source_ckpt = os.path.abspath(ckpt_path)
    return {
        "dataset": dataset,
        "D": D_model,
        "pred_len": pred_len,
        "freq": DATASETS[dataset][0],           # 빈도도 같이
        "model_role": model_role,
        "model_cfg": model_cfg,                   # ★ 어떤 파라미터로 모델이 만들어졌는지
        "eval_cfg": {
            "num_samples": num_samples,
            "batch_windows": 1,                 # 단일 윈도우 평가
            "max_windows": None,
            "skip_windows": 0,
            "seed": None,                       # 단일 평가 경로엔 args 접근이 어려우면 None로
            "start_ts": datetime.datetime.utcfromtimestamp(t_start).isoformat()+"Z",
            "end_ts":   datetime.datetime.utcfromtimestamp(t_end).isoformat()+"Z",
            "duration_sec": t_end - t_start,
    },
        "env": env,
        "source_ckpt": source_ckpt,
        "params": count_params(model),
        "infer_time_sec_mean": float(infer_time),
        "infer_time_windows": 1,
        "crps_sum_mean": float(crps),
        "crps_sum_std": 0.0,
        "windows": 1,
        "crps_meanD_mean": float(crps_meanD),
    }

@torch.no_grad()
def run_eval_rolling(ckpt_path: str, dataset: str, device: str="cuda",
                     num_samples: int=100, max_windows: int=None,
                     skip_windows: int=0, batch_windows: int=1):
    t_start = time.time()
    freq, pred_len = DATASETS[dataset]
    train_mv, test_mv, meta = load_multivariate(dataset)

    item0 = next(iter(train_mv))
    D_infer = int(item0["target"].shape[0])

    # CPU 로드 통일
    state = torch.load(ckpt_path, map_location="cpu")
    model, D_model = _build_model_from_ckpt(state, D_infer, device)
    params = count_params(model)
    context_len = pred_len

    
    # 모든 윈도우 평탄화 → 샤딩/배치
    all_pairs = []
    for item in test_mv:
        all_pairs.extend(_iter_ctx_fut_slices(item["target"], D_model, context_len, pred_len))

    total = len(all_pairs)
    if skip_windows > 0: all_pairs = all_pairs[skip_windows:]
    if max_windows is not None: all_pairs = all_pairs[:max_windows]
    N = len(all_pairs)
    if N == 0: raise RuntimeError("No valid rolling windows found in test set.")

    print(f"[EVAL] total_windows={N}", flush=True)


    bw = max(1, int(batch_windows))
    crps_list: List[float] = []
    crps_meanD_list: List[float] = []   # ← 추가
    times: List[float] = []
    done = 0
    i = 0
    while i < N:
        batch = all_pairs[i:i+bw]

        ctxs, futs, means = [], [], []
        for (ctx, fut) in batch:
            ctx_s, fut_s, mean = scale_by_context_mean(ctx, fut)
            ctxs.append(ctx_s.T)   # (C,D)
            futs.append(fut.T)     # (P,D)
            means.append(mean.T)   # (1,D)

        x_ctx = torch.tensor(np.stack(ctxs), device=device)  # (B,C,D)

        t0 = time.time()
        samp = model.forecast(x_ctx, pred_len, num_samples=num_samples)  # (B,S,P,D)
        t1 = time.time()
        per_win = float(t1 - t0) / len(batch)     # 윈도우당 시간
        times.extend([per_win] * len(batch))

        for b in range(len(batch)):
            scale = torch.tensor(means[b], device=device)  # (1,D)
            s_b = samp[b] * scale                          # (S,P,D)
            crps = crps_sum_over_horizon(s_b, torch.tensor(futs[b], device=device))
            crps_list.append(float(crps))

            # 추가: meanD도 함께 저장
            crps_meanD = crps_mean_over_horizon(s_b, torch.tensor(futs[b], device=device))
            try:
                crps_meanD_list.append(float(crps_meanD))
            except NameError:
                crps_meanD_list = [float(crps_meanD)]
        done += len(batch)
        i += bw
        # if done % (5*bw) == 0 or done == N:
        #     avg_t = np.mean(times[-min(len(times),5):])
        #     print(f"[EVAL] {done}/{N} windows | last_batch={t1-t0:.2f}s avg~{avg_t:.2f}s", flush=True)
        if done % (5*bw) == 0 or done == N:
            # per-window 평균시간(위에서 times에 윈도우 수만큼 추가하고 있으니 OK)
            avg = sum(times) / len(times)
            remain = max(N - done, 0)
            eta_sec = int(remain * avg)
            print(
                f"[EVAL] progress {done}/{N} ({done/N*100:.1f}%) | "
                f"last_batch={t1-t0:.2f}s avg/win~{avg:.2f}s | ETA~{eta_sec//60}m{eta_sec%60:02d}s",
                flush=True,
            )
    crps_meanD_mean = float(np.mean(crps_meanD_list)) if 'crps_meanD_list' in locals() else None
    crps_meanD_std  = float(np.std(crps_meanD_list))  if 'crps_meanD_list' in locals() else None
    print(f"[EVAL] CRPS_sum(mean±std)={np.mean(crps_list):.6f}±{np.std(crps_list):.6f} | "
      f"CRPS_meanD(mean±std)={crps_meanD_mean:.6f}±{crps_meanD_std:.6f}", flush=True)
    model_role = state.get("role", "teacher" if "cfg" in state else "student")
    model_cfg = state.get("cfg", {})
    t_end = time.time()
    env = _env_info(device)
    source_ckpt = os.path.abspath(ckpt_path)
    return {
        "dataset": dataset,
        "D": D_model,
        "pred_len": pred_len,
        "freq": DATASETS[dataset][0],
        "model_role": model_role,
        "model_cfg": model_cfg,                          # ★
        "eval_cfg": {
            "num_samples": num_samples,
            "batch_windows": batch_windows,
            "max_windows": max_windows,
            "skip_windows": skip_windows,
            "total_windows_all": total,              # 샤딩/절삭 전 전체 윈도우 수
            "seed": getattr(args, "seed", None),     # main에서 전달된 seed
            "start_ts": datetime.datetime.utcfromtimestamp(t_start).isoformat()+"Z",
            "end_ts":   datetime.datetime.utcfromtimestamp(t_end).isoformat()+"Z",
            "duration_sec": t_end - t_start,
    },
        "env": env,
        "source_ckpt": source_ckpt,
        "params": params,
        "infer_time_sec_mean": float(np.mean(times)),
        "infer_time_sec_std": float(np.std(times)),
        "infer_time_windows": N,
        "crps_sum_mean": float(np.mean(crps_list)),
        "crps_sum_std": float(np.std(crps_list)),
        "windows": N,
        # "crps_meanD_mean": crps_meanD_mean,   # ← 선택
        # "crps_meanD_std":  crps_meanD_std,    # ← 선택
        "crps_meanD_mean": float(np.mean(crps_meanD_list)),
        "crps_meanD_std":  float(np.std(crps_meanD_list)),
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rolling", action="store_true")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--max_windows", type=int, default=None)
    ap.add_argument("--skip_windows", type=int, default=0, help="앞의 윈도우 N개 스킵(샤딩용)")
    ap.add_argument("--batch_windows", type=int, default=1, help="한 번에 처리할 윈도우 수(B)")
    ap.add_argument("--save_json", default=None)
    args = ap.parse_args()
    _seed_all(args.seed)
    if args.rolling:
        out = run_eval_rolling(args.ckpt, args.dataset, args.device,
                               num_samples=args.num_samples,
                               max_windows=args.max_windows,
                               skip_windows=args.skip_windows,
                               batch_windows=args.batch_windows)
    else:
        out = run_eval(args.ckpt, args.dataset, args.device, num_samples=args.num_samples)

    if args.save_json:
        with open(args.save_json, "w") as f: json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, ensure_ascii=False))
