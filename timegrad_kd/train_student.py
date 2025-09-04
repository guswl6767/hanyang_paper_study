# train_student.py
import argparse, os, json, math
import numpy as np
import torch
from torch.optim import Adam
from configs import DATASETS
from data import load_multivariate, scale_by_context_mean
from model_timegrad import TimeGrad
from kd_losses import score_matching_kd, one_step_update_kd

def seed_all(s=43):
    import random, numpy as np, torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    seed_all(2025)

    ds_name = args.dataset
    freq, pred_len = DATASETS[ds_name]
    train_mv, test_mv, meta = load_multivariate(ds_name)
    train_item = next(iter(train_mv))
    target = train_item["target"]  # (D,T)
    D, T = target.shape

    # Teacher 로드 (freeze)
    t_state = torch.load(args.teacher_ckpt, map_location=args.device)
    teacher_cfg = t_state["cfg"]
    teacher = TimeGrad(D=D,
                       lstm_hidden=teacher_cfg["lstm_hidden"],
                       lstm_layers=teacher_cfg["lstm_layers"],
                       noise_emb_dim=teacher_cfg["noise_emb_dim"],
                       residual_channels=teacher_cfg["residual_channels"],
                       residual_blocks=teacher_cfg["residual_blocks"],
                       n_steps=teacher_cfg["n_diffusion_steps"],
                       beta_start=teacher_cfg["beta_start"],
                       beta_end=teacher_cfg["beta_end"]).to(args.device)
    teacher.load_state_dict(t_state["model"]); teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    # Student: 경량 + 빠른 샘플링
    student = TimeGrad(D=D,
                       lstm_hidden=24, lstm_layers=2,
                       noise_emb_dim=32,
                       residual_channels=4, residual_blocks=4,
                       n_steps=10, beta_start=1e-4, beta_end=1e-1).to(args.device)

    opt = Adam(student.parameters(), lr=1e-3)
    context_len = pred_len
    win_len = context_len + pred_len
    windows = [(s, s+win_len) for s in range(0, T - win_len + 1)]
    indices = np.arange(len(windows))

    save_dir = args.save_dir or os.path.join("checkpoints", "student", ds_name)
    os.makedirs(save_dir, exist_ok=True)
    ckpt = os.path.join(save_dir, "student.pt")

    N_t = teacher.sched.n
    N_s = student.sched.n

    for epoch in range(1, args.epochs+1):
        np.random.shuffle(indices)
        losses = []
        for i in range(0, len(indices), 64):
            batch_idx = indices[i:i+64]
            if len(batch_idx)==0: break
            ctx_list, fut_list = [], []
            for idx in batch_idx:
                s, e = windows[idx]
                seg = target[:, s:e]
                ctx, fut = seg[:, :context_len], seg[:, context_len:]
                ctx_s, fut_s, _ = scale_by_context_mean(ctx, fut)
                ctx_list.append(ctx_s.T)
                fut_list.append(fut_s.T)

            x_ctx = torch.tensor(np.stack(ctx_list), device=args.device)  # (B,C,D)
            x_fut = torch.tensor(np.stack(fut_list), device=args.device)  # (B,P,D)
            B, C, D_ = x_ctx.shape
            P = x_fut.shape[1]
            dev = x_ctx.device

            # ----- 기본 denoising loss (Student 자기 지도) -----
            loss_data = student.training_loss(x_ctx, x_fut)

            # ----- KD: 노이즈 스텝 정렬 (n_s in [1..N_s], n_t = ceil(n_s * N_t / N_s)) -----
            # 동일 x0_t, eps 사용
            # (1) 히든 상태
            h0_t = torch.zeros(teacher.rnn.num_layers, B, teacher.hdim, device=dev)
            _, (hT, _) = teacher.rnn(x_ctx, (h0_t, torch.zeros_like(h0_t)))
            h_prev_t = hT[-1]
            h0_s = torch.zeros(student.rnn.num_layers, B, student.hdim, device=dev)
            _, (hS, _) = student.rnn(x_ctx, (h0_s, torch.zeros_like(h0_s)))
            h_prev_s = hS[-1]

            # (2) t 선택: 전 시점 중 하나를 통일(배치 동일)해 안정화
            t_idx = torch.randint(low=0, high=P, size=(1,), device=dev).item()
            x0_ttrue = x_fut[:, t_idx]  # (B,D)

            # (3) 노이즈, 스텝 매핑
            n_s = torch.randint(low=1, high=N_s+1, size=(1,), device=dev).item()  # scalar
            n_t = int(np.ceil(n_s * N_t / N_s))
            eps = torch.randn_like(x0_ttrue)

            teacher.sched.to(dev); student.sched.to(dev)
            x_n_t = teacher.sched.sample_noisy(x0_ttrue, torch.full((B,), n_t, device=dev, dtype=torch.long), eps)
            x_n_s = student.sched.sample_noisy(x0_ttrue, torch.full((B,), n_s, device=dev, dtype=torch.long), eps)

            # (4) score-matching KD
            n_emb_t = teacher.noise_emb(torch.full((B,), n_t, device=dev, dtype=torch.long))
            n_emb_s = student.noise_emb(torch.full((B,), n_s, device=dev, dtype=torch.long))
            with torch.no_grad():
                eps_t = teacher.eps_net(x_n_t, h_prev_t, n_emb_t)
            eps_s = student.eps_net(x_n_s, h_prev_s, n_emb_s)
            loss_kd_score = score_matching_kd(eps_s, eps_t, weight=1.0)

            # (5) one-step update KD (각자 스케줄에서 n→n-1)
            z = torch.randn_like(x_n_t)
            with torch.no_grad():
                xn1_t = teacher.sched.ddpm_step(x_n_t, eps_t, n_t, z)
            xn1_s = student.sched.ddpm_step(x_n_s, eps_s, n_s, z)
            loss_kd_step = one_step_update_kd(xn1_s, xn1_t, weight=0.1)

            loss = loss_data + loss_kd_score + loss_kd_step
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        print(f"[{ds_name}] Epoch {epoch:03d} | loss={np.mean(losses):.6f}")
        torch.save({"model": student.state_dict(), "D": D}, ckpt)

    print("Student saved:", ckpt)

if __name__ == "__main__":
    main()
