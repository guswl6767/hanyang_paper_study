#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_final.py
- Teacher/Student ckpt를 로드해 GluonTS-style rolling 평가 수행
- H1/H2/H3 검증을 위한 확장:
  * H1: infer_time_sec_mean/std, params 기록 (전력/메모리는 별도 래퍼에서 nvidia-smi 사용 추천)
  * H2: --save_per_window_crps 로 per-window CRPS_sum / CRPS_meanD 리스트 저장 (Wilcoxon, TOST, Bootstrap에 사용)
  * H3-Calibration: --save_pit_cov 로 PIT 히스토그램(ks_D), Coverage(ECE_cov) 저장
  * H3-Structure: --save_struct 로 부분차원 평균 상관행렬 요약 저장 (Teacher/Student 간 Frobenius 거리 계산용)
"""

import argparse
import time
import json
import os
import platform
import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch

# === 프로젝트 의존 (사용자 코드) ===
from configs import DATASETS
from data import load_multivariate, scale_by_context_mean
from model_timegrad import TimeGrad
from metrics import crps_sum_over_horizon, crps_mean_over_horizon


# ------------------------------
# Utils
# ------------------------------
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

def _orient_to_DT(arr: np.ndarray, D_model: int) -> np.ndarray:
    """
    target 배열을 (D*k, T) 형태로 정규화해 돌려준다.
    - 이미 (D*k, T)이면 그대로
    - (T, D*k)이면 전치
    - 아니면 에러
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"target must be 2D, got shape={arr.shape}")
    r, c = arr.shape
    if r % D_model == 0 and r >= D_model:
        return arr  # (D*k, T)
    if c % D_model == 0 and c >= D_model:
        return arr.T  # -> (D*k, T)
    raise ValueError(f"target shape {arr.shape} incompatible with model D={D_model}")

def _iter_ctx_fut_slices(target: np.ndarray, D_model: int, context_len: int, pred_len: int):
    """
    target이 (D*k, T)이면 k개로 분할해 각 분할마다 (ctx(D,C), fut(D,P))를 만든다.
    마지막 C/P는 뒤쪽 꼬리 기준으로 자른다.
    """
    DT = _orient_to_DT(target, D_model)   # (D*k, T)
    Dk, T = DT.shape
    if T < context_len + pred_len:
        return []
    k = Dk // D_model
    out = []
    C, P = context_len, pred_len
    for i in range(k):
        sl = DT[i*D_model:(i+1)*D_model, :]      # (D, T)
        ctx = sl[:, T-(C+P):T-P]
        fut = sl[:, T-P:T]
        out.append((ctx, fut))                   # (D,C), (D,P)
    return out

def _seed_all(s: int):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ------------------------------
# Model 복원
# ------------------------------
@torch.no_grad()
def _infer_student_cfg_from_state(state_model: Dict, D_model: int):
    sd = state_model
    # residual channels
    rc = sd["eps_net.c_in.weight"].shape[0]
    # residual blocks: eps_net.blocks.<i>.skip.weight 의 최대 i + 1
    import re
    idxs = []
    for k in sd.keys():
        m = re.match(r"eps_net\.blocks\.(\d+)\.skip\.weight", k)
        if m:
            idxs.append(int(m.group(1)))
    rb = (max(idxs) + 1) if idxs else 4
    # lstm hidden: rnn.weight_hh_l0 -> (4H, H)
    H = sd["rnn.weight_hh_l0"].shape[0] // 4
    return dict(lstm_hidden=H, lstm_layers=2, noise_emb_dim=32,
                residual_channels=rc, residual_blocks=rb,
                n_steps=20, beta_start=1e-4, beta_end=1e-1)


@torch.no_grad()
def _build_model_from_ckpt(state: Dict, D_infer: int, device: str):
    D_model = int(state.get("D", D_infer))
    if "cfg" in state:
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
        model.load_state_dict(state["model"])
        model.eval()
        return model, D_model
    else:
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
        model.load_state_dict(state["model"])
        model.eval()
        return model, D_model


# ------------------------------
# PIT / Coverage 계산 유틸
# ------------------------------
def _pit_from_samples(samp: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    """
    samp: (S,P,D), y: (P,D)
    return: (P,D) PIT 값 = mean( samp <= y, axis=0 )
    """
    with torch.no_grad():
        pit = (samp <= y).float().mean(dim=0)  # (P,D)
    return pit.cpu().numpy()


def _coverage_hits(samp_np: np.ndarray, y_np: np.ndarray, levels: List[float]):
    """
    samp_np: (S,P,D), y_np: (P,D)
    return: dict[level -> hits], total_count
    """
    cov_hits = {a: 0 for a in levels}
    total = y_np.size
    for a in levels:
        lo = np.quantile(samp_np, (1.0 - a) / 2.0, axis=0)  # (P,D)
        hi = np.quantile(samp_np, (1.0 + a) / 2.0, axis=0)
        inside = ((y_np >= lo) & (y_np <= hi)).astype(np.int32)
        cov_hits[a] += int(inside.sum())
    return cov_hits, total


def _pit_hist_ks(pit_values: List[np.ndarray], n_bins: int = 20):
    """
    pit_values: 각 윈도우의 (P,D) PIT numpy 리스트
    return: bins(길이 n_bins+1), hist(길이 n_bins), KS distance 근사
    """
    if not pit_values:
        bins = np.linspace(0, 1, n_bins + 1)
        hist = np.zeros(n_bins, dtype=int)
        return bins, hist, float("nan")

    pit_all = np.concatenate([p.reshape(-1) for p in pit_values], axis=0)
    pit_all = np.clip(pit_all, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    hist, _ = np.histogram(pit_all, bins=bins)
    cum = np.cumsum(hist) / max(hist.sum(), 1)  # bin cumulative
    # 균등 CDF와의 차이: 간단 근사(중심/끝점 보정 무시)
    ks_D = float(np.max(np.abs(cum - np.linspace(1/len(hist), 1.0, len(hist)))))
    return bins, hist, ks_D


# ------------------------------
# 단일 윈도우 평가 (빠른 체크용)
# ------------------------------
@torch.no_grad()
def run_eval(ckpt_path: str, dataset: str, device: str="cuda", num_samples: int=100):
    freq, pred_len = DATASETS[dataset]
    train_mv, test_mv, meta = load_multivariate(dataset)

    item0 = next(iter(train_mv))
    D_infer = int(item0["target"].shape[0])

    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    model, D_model = _build_model_from_ckpt(state, D_infer, device)
    context_len = pred_len

    item = next(iter(test_mv))
    target = item["target"]  # (D,T)
     # 🔽 기존 ctx/fut 자르지 말고 분할해서 가져온 뒤 마지막 슬라이스 하나만 사용
    pairs = _iter_ctx_fut_slices(target, D_model, context_len, pred_len)
    if not pairs:
        raise RuntimeError("No valid window in this test item.")
    ctx, fut = pairs[-1]  # 마지막 슬라이스 하나로 빠른 체크
    ctx_s, fut_s, mean = scale_by_context_mean(ctx, fut)
    x_ctx = torch.tensor(ctx_s.T, device=device).unsqueeze(0)  

    t0 = time.time()
    samp = model.forecast(x_ctx, pred_len, num_samples=num_samples)[0]  # (S,P,D)
    infer_time = time.time() - t0

    scale = torch.tensor(mean.T, device=device)
    samp = samp * scale  # 복원
    y = torch.tensor(fut.T, device=device)

    crps = crps_sum_over_horizon(samp, y)
    crps_md = crps_mean_over_horizon(samp, y)

    out = {
        "dataset": dataset,
        "D": D_model,
        "pred_len": pred_len,
        "freq": freq,
        "model_role": state.get("role", "teacher" if "cfg" in state else "student"),
        "model_cfg": state.get("cfg", {}),
        "eval_cfg": {"num_samples": num_samples, "batch_windows": 1, "max_windows": None,
                     "skip_windows": 0, "seed": None,
                     "start_ts": datetime.datetime.utcnow().isoformat()+"Z",
                     "end_ts": None, "duration_sec": None},
        "env": _env_info(device),
        "source_ckpt": os.path.abspath(ckpt_path),
        "params": count_params(model),
        "infer_time_sec_mean": float(infer_time),
        "infer_time_windows": 1,
        "crps_sum_mean": float(crps),
        "crps_sum_std": 0.0,
        "windows": 1,
        "crps_meanD_mean": float(crps_md),
    }
    return out


# ------------------------------
# 롤링 평가 (H1/H2/H3용 완전판)
# ------------------------------
@torch.no_grad()
def run_eval_rolling(ckpt_path: str, dataset: str, device: str="cuda",
                     num_samples: int=100, max_windows: int=None,
                     skip_windows: int=0, batch_windows: int=1,
                     save_pit_cov: bool=False, pit_bins: int=20,
                     cov_levels: List[float]=None,
                     save_per_window_crps: bool=False,
                     save_struct: bool=False, struct_sample_dims: int=64,
                     struct_sample_windows: int=50, struct_random_seed: int=2025):

    if cov_levels is None:
        cov_levels = [0.5, 0.8, 0.9, 0.95]

    freq, pred_len = DATASETS[dataset]
    train_mv, test_mv, meta = load_multivariate(dataset)
    item0 = next(iter(train_mv))
    D_infer = int(item0["target"].shape[0])

    # ckpt 로드 → 모델 to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    model, D_model = _build_model_from_ckpt(state, D_infer, device)
    params = count_params(model)
    context_len = pred_len

    # rolling windows 준비 (여기서는 GluonTS test item의 "마지막" 윈도우만 사용)
    all_pairs = []
    for item in test_mv:
        target = item["target"]  # (D,T)
        slc = _iter_ctx_fut_slices(target, D_model, context_len, pred_len)
        all_pairs.extend(slc)

    total = len(all_pairs)
    if skip_windows > 0:
        all_pairs = all_pairs[skip_windows:]
    if max_windows is not None:
        all_pairs = all_pairs[:max_windows]
    N = len(all_pairs)
    if N == 0:
        raise RuntimeError("No valid rolling windows found in test set.")

    # 결과 통계
    crps_list: List[float] = []
    crps_md_list: List[float] = []
    times: List[float] = []

    # per-window 저장(옵션)
    pw_sum_list: List[float] = []
    pw_md_list: List[float] = []

    # PIT/coverage
    pit_values: List[np.ndarray] = []
    cov_hits_sum = {a: 0 for a in cov_levels}
    cov_total_sum = 0

    # 구조 요약(부분차원 평균 상관행렬)
    if save_struct:
        rng = np.random.default_rng(struct_random_seed)
        dim_idx = None
        struct_sum = None  # 누적 합 (k,k)
        struct_count = 0

    # 배치 처리
    bw = max(1, int(batch_windows))
    i = 0
    done = 0
    t_start = time.time()
    while i < N:
        batch = all_pairs[i:i+bw]
        ctxs, futs, means = [], [], []
        for (ctx, fut) in batch:
            ctx_s, fut_s, mean = scale_by_context_mean(ctx, fut)
            ctxs.append(ctx_s.T)   # (C,D)
            futs.append(fut.T)     # (P,D)
            means.append(mean.T)   # (1,D)
        x_ctx = torch.tensor(np.stack(ctxs), device=device)  # (B,C,D)

        # 샘플
        t0 = time.time()
        samp = model.forecast(x_ctx, pred_len, num_samples=num_samples)  # (B,S,P,D)
        t1 = time.time()
        per_win = float(t1 - t0) / len(batch)
        times.extend([per_win] * len(batch))

        # 각 윈도우 처리
        for b in range(len(batch)):
            scale = torch.tensor(means[b], device=device)
            s_b = samp[b] * scale                          # (S,P,D)
            y_b = torch.tensor(futs[b], device=device)     # (P,D)

            crps = crps_sum_over_horizon(s_b, y_b)
            crps_md = crps_mean_over_horizon(s_b, y_b)
            crps_list.append(float(crps))
            crps_md_list.append(float(crps_md))
            if save_per_window_crps:
                pw_sum_list.append(float(crps))
                pw_md_list.append(float(crps_md))

            if save_pit_cov:
                # PIT
                pit = _pit_from_samples(s_b, y_b)  # (P,D)
                pit_values.append(pit)
                # Coverage
                s_np = s_b.cpu().numpy()
                y_np = y_b.cpu().numpy()
                hits, tot = _coverage_hits(s_np, y_np, cov_levels)
                for a in cov_levels:
                    cov_hits_sum[a] += hits[a]
                cov_total_sum += tot

            if save_struct and struct_count < struct_sample_windows:
                # 부분차원 상관행렬 평균
                s_np = s_b.cpu().numpy()  # (S,P,D)
                S,P,D = s_np.shape
                if dim_idx is None:
                    k = min(int(struct_sample_dims), D)
                    dim_idx = rng.choice(D, size=k, replace=False)
                k = len(dim_idx)
                corr_acc = np.zeros((k,k), dtype=np.float64)
                for p in range(P):
                    X = s_np[:, p, :][:, dim_idx]  # (S,k)
                    if X.std(axis=0).min() < 1e-12:
                        continue
                    C = np.corrcoef(X, rowvar=False)  # (k,k)
                    corr_acc += C
                corr_acc /= max(P,1)
                if struct_sum is None:
                    struct_sum = corr_acc
                else:
                    struct_sum += corr_acc
                struct_count += 1

        done += len(batch)
        i += bw
        if done % (5*bw) == 0 or done == N:
            avg = sum(times) / len(times)
            remain = max(N - done, 0)
            eta = int(remain * avg)
            print(f"[EVAL] progress {done}/{N} ({done/N*100:.1f}%) | "
                  f"last_batch={t1-t0:.2f}s avg/win~{avg:.2f}s | ETA~{eta//60}m{eta%60:02d}s",
                  flush=True)

    # 집계 JSON
    out = {
        "dataset": dataset,
        "D": D_model,
        "pred_len": pred_len,
        "freq": DATASETS[dataset][0],
        "model_role": state.get("role", "teacher" if "cfg" in state else "student"),
        "model_cfg": state.get("cfg", {}),
        "eval_cfg": {
            "num_samples": num_samples,
            "batch_windows": batch_windows,
            "max_windows": max_windows,
            "skip_windows": skip_windows,
            "total_windows_all": total,
            "seed": None,  # main에서 설정
            "start_ts": datetime.datetime.utcnow().isoformat()+"Z",
            "end_ts": None,
            "duration_sec": None
        },
        "env": _env_info(device),
        "source_ckpt": os.path.abspath(ckpt_path),
        "params": params,
        "infer_time_sec_mean": float(np.mean(times)),
        "infer_time_sec_std": float(np.std(times)),
        "infer_time_windows": N,
        "crps_sum_mean": float(np.mean(crps_list)),
        "crps_sum_std": float(np.std(crps_list)),
        "crps_meanD_mean": float(np.mean(crps_md_list)),
        "crps_meanD_std": float(np.std(crps_md_list)),
        "windows": N,
    }

    # per-window 저장
    if save_per_window_crps:
        out["per_window"] = {
            "crps_sum": pw_sum_list,     # 길이 = windows (N)
            "crps_meanD": pw_md_list
        }

    # PIT/coverage
    if save_pit_cov:
        bins, hist, ks_D = _pit_hist_ks(pit_values, n_bins=pit_bins)
        ece = 0.0
        cov_payload = {}
        if cov_total_sum > 0:
            for a in cov_levels:
                p_hat = cov_hits_sum[a] / cov_total_sum
                cov_payload[str(int(a*100))] = {
                    "hits": int(cov_hits_sum[a]),
                    "total": int(cov_total_sum),
                    "emp_cov": float(p_hat)
                }
                ece += abs(p_hat - a)
            ece /= len(cov_levels)

        out["pit"] = {
            "bins": bins.tolist(),
            "hist": [int(x) for x in hist.tolist()],
            "ks_D": float(ks_D),
        }
        out["coverage"] = {
            "levels": [float(a) for a in cov_levels],
            "cov": cov_payload,
            "ECE_cov": float(ece)
        }

    # 구조 요약
    if save_struct and (struct_count > 0) and (struct_sum is not None):
        corr_mean = (struct_sum / struct_count).tolist()
        out["struct"] = {
            "dim_idx": dim_idx.tolist(),
            "corr_mean": corr_mean,     # (k,k) 평균 상관행렬
            "k": int(len(dim_idx)),
            "used_windows": int(struct_count)
        }

    # 종료시간/소요
    t_end = time.time()
    out["eval_cfg"]["end_ts"] = datetime.datetime.utcnow().isoformat()+"Z"
    out["eval_cfg"]["duration_sec"] = float(t_end - t_start)

    return out


# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rolling", action="store_true")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--max_windows", type=int, default=None)
    ap.add_argument("--skip_windows", type=int, default=0,
                    help="앞의 윈도우 N개 스킵(샤딩용)")
    ap.add_argument("--batch_windows", type=int, default=1,
                    help="한 번에 처리할 윈도우 수(B)")
    ap.add_argument("--save_json", default=None)

    # H3-Calibration
    ap.add_argument("--save_pit_cov", action="store_true",
                    help="PIT histogram & Coverage 통계 저장")
    ap.add_argument("--pit_bins", type=int, default=20,
                    help="PIT histogram bin 개수")
    ap.add_argument("--coverage_levels", type=str, default="0.5,0.8,0.9,0.95",
                    help="Coverage 레벨 CSV (예: 0.5,0.8,0.9,0.95)")

    # H2 per-window
    ap.add_argument("--save_per_window_crps", action="store_true",
                    help="윈도우별 CRPS_sum/CRPS_meanD 리스트 저장")

    # H3-Structure
    ap.add_argument("--save_struct", action="store_true",
                    help="다변량 구조 요약(부분차원 평균 상관행렬) 저장")
    ap.add_argument("--struct_sample_dims", type=int, default=64,
                    help="상관계산에 사용할 부분 차원 개수(k)")
    ap.add_argument("--struct_sample_windows", type=int, default=50,
                    help="구조 계산에 사용할 최대 윈도우 수")
    ap.add_argument("--struct_random_seed", type=int, default=2025,
                    help="부분 차원 샘플링 시드")

    args = ap.parse_args()
    _seed_all(args.seed)

    cov_levels = [float(x.strip()) for x in args.coverage_levels.split(",") if x.strip()]

    if args.rolling:
        out = run_eval_rolling(args.ckpt, args.dataset, args.device,
                               num_samples=args.num_samples,
                               max_windows=args.max_windows,
                               skip_windows=args.skip_windows,
                               batch_windows=args.batch_windows,
                               save_pit_cov=args.save_pit_cov,
                               pit_bins=args.pit_bins,
                               cov_levels=cov_levels,
                               save_per_window_crps=args.save_per_window_crps,
                               save_struct=args.save_struct,
                               struct_sample_dims=args.struct_sample_dims,
                               struct_sample_windows=args.struct_sample_windows,
                               struct_random_seed=args.struct_random_seed)
        out["eval_cfg"]["seed"] = args.seed
    else:
        out = run_eval(args.ckpt, args.dataset, args.device, num_samples=args.num_samples)
        out["eval_cfg"]["seed"] = args.seed

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
