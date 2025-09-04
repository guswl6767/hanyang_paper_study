# model_timegrad.py
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from diffusion import LinearBetaSchedule
from epsilon_net import EpsilonNet, FourierNoiseEmbedding

class TimeGrad(nn.Module):
    def __init__(self, D: int, lstm_hidden=40, lstm_layers=2,
                 noise_emb_dim=32, residual_channels=8, residual_blocks=8,
                 n_steps=100, beta_start=1e-4, beta_end=1e-1):
        super().__init__()
        self.D = D
        self.hdim = lstm_hidden
        self.rnn = nn.LSTM(input_size=D, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.noise_emb = FourierNoiseEmbedding(dim=noise_emb_dim)
        self.eps_net = EpsilonNet(D=D, cond_dim=lstm_hidden, noise_emb_dim=noise_emb_dim,
                                  residual_channels=residual_channels, residual_blocks=residual_blocks)
        self.sched = LinearBetaSchedule(n_steps, beta_start, beta_end)

    def device(self):
        return next(self.parameters()).device

    # ----- training one-step loss (denoising MSE) -----
    def training_loss(self, x_ctx: torch.Tensor, x_fut: torch.Tensor):
        """
        x_ctx: (B, C, D), x_fut: (B, P, D)  -- 정답 x0는 x_fut의 각 시점
        """
        B, C, D = x_ctx.shape
        P = x_fut.shape[1]
        dev = self.device()
        # RNN warm-up with context
        h0 = torch.zeros(self.rnn.num_layers, B, self.hdim, device=dev)
        c0 = torch.zeros_like(h0)
        _, (h_t, c_t) = self.rnn(x_ctx, (h0, c0))  # h_t: (layers,B,H)
        h_prev = h_t[-1]  # (B,H)

        # random pick a future step t and diffusion level n
        t_idx = torch.randint(low=0, high=P, size=(B,), device=dev)
        n_idx = torch.randint(low=1, high=self.sched.n, size=(B,), device=dev)
        x0_t = x_fut[torch.arange(B, device=dev), t_idx]  # (B,D)
        eps = torch.randn_like(x0_t)
        self.sched.to(dev)
        x_n = self.sched.sample_noisy(x0_t, n_idx, eps)

        n_emb = self.noise_emb(n_idx)   # (B,E)
        eps_hat = self.eps_net(x_n, h_prev, n_emb)
        loss = F.mse_loss(eps_hat, eps)
        return loss

    # ----- forecast sampling (Algorithm 2) -----
    @torch.no_grad()
    def forecast(self, x_ctx: torch.Tensor, pred_len: int, num_samples: int = 100):
        """
        x_ctx: (B, C, D) context (스케일링된 값), 반환: samples (B, S, P, D)
        """
        self.eval()
        B, C, D = x_ctx.shape
        dev = self.device()
        self.sched.to(dev)

        # RNN state from context
        h0 = torch.zeros(self.rnn.num_layers, B, self.hdim, device=dev)
        c0 = torch.zeros_like(h0)
        _, (h_t, c_t) = self.rnn(x_ctx, (h0, c0))
        h_prev = h_t[-1]

        S = num_samples
        samples = torch.zeros(B, S, pred_len, D, device=dev)

        for s in range(S):
            x_prev = []
            h = h_prev.clone()
            # iterative autoregressive generation
            for t in range(pred_len):
                # start from white noise x_N
                x_n = torch.randn(B, D, device=dev)
                for n in range(self.sched.n, 0, -1):
                    n_idx = torch.full((B,), n, device=dev, dtype=torch.long)
                    n_emb = self.noise_emb(n_idx)
                    eps_hat = self.eps_net(x_n, h, n_emb)
                    z = torch.randn_like(x_n) if n > 1 else torch.zeros_like(x_n)
                    x_n = self.sched.ddpm_step(x_n, eps_hat, n, z)
                x0 = x_n  # predicted next step
                x_prev.append(x0)
                # feed as input to RNN for next step
                x_in = x0.unsqueeze(1)  # (B,1,D)
                _, (h_t, c_t) = self.rnn(x_in, (h.unsqueeze(0), torch.zeros_like(h).unsqueeze(0)))
                h = h_t[-1]
            samples[:, s] = torch.stack(x_prev, dim=1)  # (B,P,D)

        return samples  # (B,S,P,D)
