# evaluate.py
import argparse, time, os, json
import numpy as np
import torch
from typing import Dict, List
from configs import DATASETS, TimeGradPaperCfg
from data import load_multivariate, split_context_pred, scale_by_context_mean
from model_timegrad import TimeGrad
from metrics import crps_sum_over_horizon

def count_params(model): 
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def _build_model_from_ckpt(state: Dict, D: int, device: str):
    if "cfg" in state:  # teacher
        cfg = state["cfg"]
        model = TimeGrad(D=D,
                         lstm_hidden=cfg["lstm_hidden"], lstm_layers=cfg["lstm_layers"],
                         noise_emb_dim=cfg["noise_emb_dim"],
                         residual_channels=cfg["residual_channels"], residual_blocks=cfg["residual_blocks"],
                         n_steps=cfg["n_diffusion_steps"], beta_start=cfg["beta_start"], beta_end=cfg["beta_end"]).to(device)
    else:  # student (경량 고정 구성)
        model = TimeGrad(D=D, lstm_hidden=24, lstm_layers=2,
                         noise_emb_dim=32, residual_channels=4, residual_blocks=4,
                         n_steps=10, beta_start=1e-4, beta_end=1e-1).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model

@torch.no_grad()
def run_eval(ckpt_path: str, dataset: str, device: str = "cuda", num_samples: int = 100):
    """마지막 윈도우만 평가(빠른 스팟체크)."""
    freq, pred_len = DATASETS[dataset]
    train_mv, test_mv, meta = load_multivariate(dataset)
    item = next(iter(test_mv))
    target = item["target"]  # (D, T)
    D, T = target.shape
    context_len = pred_len

    state = torch.load(ckpt_path, map_location=device)
    model = _build_model_from_ckpt(state, D, device)

    ctx, fut = split_context_pred(target, context_len, pred_len)   # (D,C),(D,P)
    ctx_s, fut_s, mean = scale_by_context_mean(ctx, fut)
    x_ctx = torch.tensor(ctx_s.T, device=device).unsqueeze(0)  # (1,C,D)

    t0 = time.time()
    samples = model.forecast(x_ctx, pred_len, num_samples=num_samples)[0]  # (S,P,D)
    infer_time = time.time() - t0

    # 복원
    samples = samples * torch.tensor(mean.T, device=device)  # (S,P,D)
    crps = crps_sum_over_horizon(samples, torch.tensor(fut.T, device=device))

    return {"params": count_params(model),
            "infer_time_sec_mean": infer_time,
            "infer_time_windows": 1,
            "crps_sum_mean": float(crps),
            "crps_sum_std": 0.0,
            "windows": 1}

@torch.no_grad()
def run_eval_rolling(ckpt_path: str, dataset: str, device: str = "cuda", num_samples: int = 100, max_windows: int = None):
    """GluonTS test의 모든 롤링 윈도우에 대해 평균/표준편차 계산."""
    freq, pred_len = DATASETS[dataset]
    train_mv, test_mv, meta = load_multivariate(dataset)
    first = next(iter(train_mv))
    D = first["target"].shape[0]
    context_len = pred_len

    state = torch.load(ckpt_path, map_location=device)
    model = _build_model_from_ckpt(state, D, device)
    params = count_params(model)

    crps_list: List[float] = []
    times: List[float] = []
    n_windows = 0

    for k, item in enumerate(test_mv):
        if (max_windows is not None) and (n_windows >= max_windows):
            break
        target = item["target"]  # (D,T_i)
        D_i, T_i = target.shape
        if T_i < context_len + pred_len:
            continue  # 짧으면 스킵

        ctx, fut = split_context_pred(target, context_len, pred_len)   # (D,C),(D,P)
        ctx_s, fut_s, mean = scale_by_context_mean(ctx, fut)
        x_ctx = torch.tensor(ctx_s.T, device=device).unsqueeze(0)  # (1,C,D)

        t0 = time.time()
        samples = model.forecast(x_ctx, pred_len, num_samples=num_samples)[0]  # (S,P,D)
        t1 = time.time()
        samples = samples * torch.tensor(mean.T, device=device)
        crps = crps_sum_over_horizon(samples, torch.tensor(fut.T, device=device))

        crps_list.append(float(crps))
        times.append(float(t1 - t0))
        n_windows += 1

    if n_windows == 0:
        raise RuntimeError("No valid rolling windows found in test set.")

    return {
        "params": params,
        "infer_time_sec_mean": float(np.mean(times)),
        "infer_time_sec_std": float(np.std(times)),
        "infer_time_windows": n_windows,
        "crps_sum_mean": float(np.mean(crps_list)),
        "crps_sum_std": float(np.std(crps_list)),
        "windows": n_windows,
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rolling", action="store_true", help="롤링 윈도우 전체 평가")
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--max_windows", type=int, default=None)
    args = ap.parse_args()

    if args.rolling:
        out = run_eval_rolling(args.ckpt, args.dataset, args.device, num_samples=args.num_samples, max_windows=args.max_windows)
    else:
        out = run_eval(args.ckpt, args.dataset, args.device, num_samples=args.num_samples)

    print(json.dumps(out, indent=2, ensure_ascii=False))
