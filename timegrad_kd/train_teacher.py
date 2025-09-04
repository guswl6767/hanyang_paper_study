# train_teacher.py
import argparse, os, json
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from configs import DATASETS, TimeGradPaperCfg
from data import load_multivariate, scale_by_context_mean
from model_timegrad import TimeGrad

def seed_all(s=42):
    import random, numpy as np, torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--epochs", type=int, default=50)
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

    cfg = TimeGradPaperCfg()
    context_len = pred_len
    model = TimeGrad(D=D,
                     lstm_hidden=cfg.lstm_hidden, lstm_layers=cfg.lstm_layers,
                     noise_emb_dim=cfg.noise_emb_dim,
                     residual_channels=cfg.residual_channels, residual_blocks=cfg.residual_blocks,
                     n_steps=cfg.n_diffusion_steps, beta_start=cfg.beta_start, beta_end=cfg.beta_end).to(args.device)
    opt = Adam(model.parameters(), lr=cfg.lr)

    win_len = context_len + pred_len
    windows = [(s, s+win_len) for s in range(0, T - win_len + 1)]
    indices = np.arange(len(windows))

    save_dir = args.save_dir or os.path.join("checkpoints", "teacher", ds_name)
    os.makedirs(save_dir, exist_ok=True)
    ckpt = os.path.join(save_dir, "teacher.pt")

    for epoch in range(1, args.epochs+1):
        np.random.shuffle(indices)
        losses = []
        for i in range(0, len(indices), cfg.batch_size):
            batch_idx = indices[i:i+cfg.batch_size]
            if len(batch_idx)==0: break
            ctx_list, fut_list = [], []
            for idx in batch_idx:
                s, e = windows[idx]
                seg = target[:, s:e]
                ctx, fut = seg[:, :context_len], seg[:, context_len:]
                ctx_s, fut_s, _ = scale_by_context_mean(ctx, fut)
                ctx_list.append(ctx_s.T); fut_list.append(fut_s.T)
            x_ctx = torch.tensor(np.stack(ctx_list), device=args.device)
            x_fut = torch.tensor(np.stack(fut_list), device=args.device)

            loss = model.training_loss(x_ctx, x_fut)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        print(f"[{ds_name}] Epoch {epoch:03d} | loss={np.mean(losses):.6f}")
        torch.save({"model": model.state_dict(), "D": D, "cfg": cfg.__dict__}, ckpt)

    print("Teacher saved:", ckpt)

if __name__ == "__main__":
    main()
