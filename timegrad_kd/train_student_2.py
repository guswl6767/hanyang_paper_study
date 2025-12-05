# train_student.py
import argparse, os, json, math
import numpy as np
import torch
from torch.optim import Adam
from configs import DATASETS
from data import load_multivariate, scale_by_context_mean
from model_timegrad import TimeGrad
from kd_losses import score_matching_kd, one_step_update_kd
import torch.nn.functional as F   # ★ KD/손실 계산에 필요

# train_student.py top 근처
DATASET_STU_CFG = {
    "exchange_rate":     dict(n_steps_s=20, res_channels=6,  res_blocks=4,  lambda_kd=0.5, mu_kd=0.05, gamma_x0=0.10, kd_warmup=10, grad_clip=1.0),
    "solar_nips":        dict(n_steps_s=30, res_channels=8,  res_blocks=6,  lambda_kd=0.5, mu_kd=0.05, gamma_x0=0.10, kd_warmup=20, grad_clip=1.0),
    "electricity_nips":  dict(n_steps_s=30, res_channels=8,  res_blocks=6,  lambda_kd=0.5, mu_kd=0.05, gamma_x0=0.10, kd_warmup=20, grad_clip=1.0),
    "traffic_nips":      dict(n_steps_s=40, res_channels=12, res_blocks=8,  lambda_kd=0.5, mu_kd=0.05, gamma_x0=0.15, kd_warmup=20, grad_clip=1.0),
    "taxi_30min":        dict(n_steps_s=40, res_channels=12, res_blocks=8,  lambda_kd=0.5, mu_kd=0.05, gamma_x0=0.15, kd_warmup=20, grad_clip=1.0),
    "wiki-rolling_nips": dict(n_steps_s=50, res_channels=16, res_blocks=8,  lambda_kd=0.5, mu_kd=0.05, gamma_x0=0.20, kd_warmup=20, grad_clip=1.0),
}


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
    # argparse 추가
    parser.add_argument("--n_steps_s", type=int, default=20, help="Student diffusion steps (기본 20)")
    parser.add_argument("--res_channels", type=int, default=6, help="Student ε-net residual channels")
    parser.add_argument("--res_blocks", type=int, default=6, help="Student ε-net residual blocks")

    parser.add_argument("--lambda_kd", type=float, default=0.5, help="score KD weight")
    parser.add_argument("--mu_kd",     type=float, default=0.05, help="1-step KD weight")
    parser.add_argument("--gamma_x0",  type=float, default=0.10, help="x0-KD weight (trajectory KD)")
    parser.add_argument("--kd_warmup", type=int,   default=20, help="초반 N epoch은 KD 미적용")

    parser.add_argument("--grad_clip", type=float, default=0.0, help=">0 이면 grad norm clip")

    args = parser.parse_args()
    default_cfg = DATASET_STU_CFG.get(args.dataset, {})
    for k, v in default_cfg.items():
        if getattr(args, k, None) in (None, 0, 0.0):  # CLI에서 안 준 경우만 채우기
            setattr(args, k, v)
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
    # student = TimeGrad(D=D,
    #                     lstm_hidden=24, lstm_layers=2,
    #                     noise_emb_dim=32,
    #                     residual_channels=args.res_channels,    # 6
    #                     residual_blocks=args.res_blocks,        # 6
    #                     n_steps=args.n_steps_s,    
    #                     beta_start=1e-4, beta_end=1e-1).to(args.device)
    student = TimeGrad(D=D,
                    lstm_hidden=24, lstm_layers=2,
                    noise_emb_dim=32,
                    residual_channels=args.res_channels,    # 6
                    residual_blocks=args.res_blocks,        # 6
                    n_steps=args.n_steps_s,    
                    beta_start=1e-4, beta_end=1e-1).to(args.device)

    opt = Adam(student.parameters(), lr=1e-3)
    # ★ EMA 초기화
    ema = {k: v.detach().clone() for k, v in student.state_dict().items()}
    def ema_update(mu: float = 0.999):
        with torch.no_grad():
            for k, v in student.state_dict().items():
                ema[k].mul_(mu).add_(v.detach(), alpha=1 - mu)

    context_len = pred_len
    win_len = context_len + pred_len
    windows = [(s, s+win_len) for s in range(0, T - win_len + 1)]
    indices = np.arange(len(windows))

    save_dir = args.save_dir or os.path.join("/root/Storage/hanyang_paper_study/Models/checkpoints/v2", "student", ds_name)
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
            # h0_t = torch.zeros(teacher.rnn.num_layers, B, teacher.hdim, device=dev)
            # _, (hT, _) = teacher.rnn(x_ctx, (h0_t, torch.zeros_like(h0_t)))
            # h_prev_t = hT[-1]
            # h0_s = torch.zeros(student.rnn.num_layers, B, student.hdim, device=dev)
            # _, (hS, _) = student.rnn(x_ctx, (h0_s, torch.zeros_like(h0_s)))
            # h_prev_s = hS[-1]

            # # (2) t 선택: 전 시점 중 하나를 통일(배치 동일)해 안정화
            # t_idx = torch.randint(low=0, high=P, size=(1,), device=dev).item()
            # x0_ttrue = x_fut[:, t_idx]  # (B,D)

            # # (3) 노이즈, 스텝 매핑
            # n_s = torch.randint(low=1, high=N_s+1, size=(1,), device=dev).item()  # scalar
            # n_t = int(np.ceil(n_s * N_t / N_s))
            # eps = torch.randn_like(x0_ttrue)

            # teacher.sched.to(dev); student.sched.to(dev)
            # x_n_t = teacher.sched.sample_noisy(x0_ttrue, torch.full((B,), n_t, device=dev, dtype=torch.long), eps)
            # x_n_s = student.sched.sample_noisy(x0_ttrue, torch.full((B,), n_s, device=dev, dtype=torch.long), eps)

            # # (4) score-matching KD
            # n_emb_t = teacher.noise_emb(torch.full((B,), n_t, device=dev, dtype=torch.long))
            # n_emb_s = student.noise_emb(torch.full((B,), n_s, device=dev, dtype=torch.long))
            # with torch.no_grad():
            #     eps_t = teacher.eps_net(x_n_t, h_prev_t, n_emb_t)
            # eps_s = student.eps_net(x_n_s, h_prev_s, n_emb_s)
            # loss_kd_score = score_matching_kd(eps_s, eps_t, weight=1.0)

            # # (5) one-step update KD (각자 스케줄에서 n→n-1)
            # z = torch.randn_like(x_n_t)
            # with torch.no_grad():
            #     xn1_t = teacher.sched.ddpm_step(x_n_t, eps_t, n_t, z)
            # xn1_s = student.sched.ddpm_step(x_n_s, eps_s, n_s, z)
            # loss_kd_step = one_step_update_kd(xn1_s, xn1_t, weight=0.1)

            # loss = loss_data + loss_kd_score + loss_kd_step
            # opt.zero_grad(); loss.backward(); opt.step()
            # losses.append(loss.item())
            # ----- KD: 노이즈 스텝 정렬 (n_s in [1..N_s], n_t = ceil(n_s * N_t / N_s)) -----

            # ===교체====
            # 동일 x0_t, eps 사용
            # (1) 히든 상태
            # ----- KD: 멀티-타임스텝 distillation -----
            K = max(1, getattr(args, "kd_multi", 1))
            kd_score = 0.0; kd_step = 0.0; kd_x0 = 0.0; kd_h = 0.0

            # teacher/student 컨텍스트 hidden (한 번만)
            h0_t = torch.zeros(teacher.rnn.num_layers, B, teacher.hdim, device=dev)
            _, (hT, _) = teacher.rnn(x_ctx, (h0_t, torch.zeros_like(h0_t)))
            h_prev_t = hT[-1]

            h0_s = torch.zeros(student.rnn.num_layers, B, student.hdim, device=dev)
            _, (hS, _) = student.rnn(x_ctx, (h0_s, torch.zeros_like(h0_s)))
            h_prev_s = hS[-1]

            for _ in range(K):
                t_pick = torch.randint(low=0, high=P, size=(1,), device=dev).item()
                x0_true = x_fut[:, t_pick]

                n_s = torch.randint(low=1, high=N_s+1, size=(1,), device=dev).item()
                n_t = int(np.ceil(n_s * N_t / N_s))
                eps = torch.randn_like(x0_true)

                teacher.sched.to(dev); student.sched.to(dev)
                x_n_t = teacher.sched.sample_noisy(x0_true, torch.full((B,), n_t, device=dev, dtype=torch.long), eps)
                x_n_s = student.sched.sample_noisy(x0_true, torch.full((B,), n_s, device=dev, dtype=torch.long), eps)

                n_emb_t = teacher.noise_emb(torch.full((B,), n_t, device=dev, dtype=torch.long))
                n_emb_s = student.noise_emb(torch.full((B,), n_s, device=dev, dtype=torch.long))
                with torch.no_grad():
                    eps_t = teacher.eps_net(x_n_t, h_prev_t, n_emb_t)
                eps_s = student.eps_net(x_n_s, h_prev_s, n_emb_s)

                # score-KD
                kd_score = kd_score + F.mse_loss(eps_s, eps_t)

                # 1-step KD
                z = torch.randn_like(x_n_t)
                with torch.no_grad():
                    xn1_t = teacher.sched.ddpm_step(x_n_t, eps_t, n_t, z)
                xn1_s = student.sched.ddpm_step(x_n_s, eps_s, n_s, z)
                kd_step  = kd_step  + F.mse_loss(xn1_s, xn1_t)

                # x0-KD
                ab_t = teacher.sched.alpha_bar[n_t-1].to(dev); ab_s = student.sched.alpha_bar[n_s-1].to(dev)
                sqrt_ab_t, sqrt_ab_s = torch.sqrt(ab_t), torch.sqrt(ab_s)
                sqrt_1mab_t, sqrt_1mab_s = torch.sqrt(1 - ab_t), torch.sqrt(1 - ab_s)
                with torch.no_grad():
                    x0_hat_t = (x_n_t - sqrt_1mab_t * eps_t) / sqrt_ab_t
                x0_hat_s = (x_n_s - sqrt_1mab_s * eps_s) / sqrt_ab_s
                kd_x0 = kd_x0 + F.mse_loss(x0_hat_s, x0_hat_t)
                kd_h = 0.0
                # hidden-KD (컨텍스트 표현 정렬)
                # kd_h = kd_h + F.mse_loss(h_prev_s, h_prev_t)

            # 평균
            kd_score /= K; kd_step /= K; kd_x0 /= K; kd_h /= K

            # ----- total loss (warm-up + weights) -----
            use_kd = (epoch > args.kd_warmup)
            if use_kd:
                kd_term = args.lambda_kd * kd_score + args.mu_kd * kd_step + args.gamma_x0 * kd_x0 + 0.05 * kd_h
            else:
                kd_term = torch.zeros((), device=dev)

            loss = loss_data + kd_term


            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            opt.step()

            ema_update(mu=0.999)  # ★ EMA 갱신

            losses.append(loss.item())
            if (i == 0):
                print(
                    f"[{ds_name}] Epoch {epoch:03d} | total={np.mean(losses):.6f} "
                    f"| data={float(loss_data):.4f} "
                    f"| kd_s={float(kd_score):.4f} kd_step={float(kd_step):.4f} kd_x0={float(kd_x0):.4f} kd_h={float(kd_h):.4f} "
                    f"| use_kd={use_kd}"
                )
        print(f"[{ds_name}] Epoch {epoch:03d} | loss={np.mean(losses):.6f}")
        # torch.save({"model": student.state_dict(), "D": D}, ckpt)
        cpu_ema = {k: v.cpu() for k, v in ema.items()}
        stu_cfg = {
            "lstm_hidden": 24, "lstm_layers": 2, "noise_emb_dim": 32,
            "residual_channels": int(student.eps_net.blocks[0].conv_f.out_channels),
            "residual_blocks":  int(len(student.eps_net.blocks)),
            "n_diffusion_steps": int(student.sched.n),
            "beta_start": 1e-4, "beta_end": 1e-1,
            "lambda_kd": getattr(args, "lambda_kd", None),
            "mu_kd":     getattr(args, "mu_kd", None),
            "gamma_x0":  getattr(args, "gamma_x0", None),
            "kd_warmup": getattr(args, "kd_warmup", None),
            "grad_clip": getattr(args, "grad_clip", None),
        }
        torch.save({"model": cpu_ema, "D": D, "cfg": stu_cfg, "role": "student"}, ckpt)

    print("Student saved:", ckpt)

if __name__ == "__main__":
    main()
