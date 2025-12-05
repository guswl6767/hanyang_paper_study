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
        # eps_hat = self.eps_net(x_n, h_prev, n_emb)
        # loss = F.mse_loss(eps_hat, eps)
        eps_hat = self.eps_net(x_n, h_prev, n_emb)     
        raw_mse = F.mse_loss(eps_hat, eps)  # 순수 MSE    # (B,D)
        a_bar = self.sched.alpha_bar[n_idx-1].view(-1,1)   # (B,1)
        snr = a_bar / (1 - a_bar + 1e-8)                   # (B,1)
        w = snr.pow(0.5)                                   # gamma=0.5
        loss = (w * (eps_hat - eps).pow(2)).mean()
        # 디버그용 로그 저장 (모델 속성에)
        self._last_losses = {
            "raw_mse": float(raw_mse.detach().item()),
            "weighted": float(loss.detach().item()),
        }
        return loss

    # ----- forecast sampling (Algorithm 2) -----
    @torch.no_grad()
    def forecast(self, x_ctx: torch.Tensor, pred_len: int, num_samples: int = 100):
        """
        x_ctx: (B, C, D)  -> 반환: samples (B, S, P, D)
        """
        self.eval()
        B, C, D = x_ctx.shape
        dev = self.device()
        self.sched.to(dev)

        # 1) 컨텍스트로 RNN 상태 만들기 (전체 레이어 스택)
        h0 = torch.zeros(self.rnn.num_layers, B, self.hdim, device=dev)
        c0 = torch.zeros_like(h0)
        _, (h_full, c_full) = self.rnn(x_ctx, (h0, c0))   # h_full: (L,B,H), c_full: (L,B,H)
        h_cond = h_full[-1]                                # ε-net 조건으로 쓸 마지막 레이어 (B,H)

        S = num_samples
        samples = torch.zeros(B, S, pred_len, D, device=dev)

        for s in range(S):
            x_prev = []
            # 샘플마다 RNN state 복사
            h_state = h_full.clone()
            c_state = c_full.clone()
            h_last  = h_cond.clone()

            for t in range(pred_len):
                # 2) Diffusion 복원: x_N -> x_0
                x_n = torch.randn(B, D, device=dev)
                for n in range(self.sched.n, 0, -1):
                    n_idx = torch.full((B,), n, device=dev, dtype=torch.long)
                    n_emb = self.noise_emb(n_idx)
                    eps_hat = self.eps_net(x_n, h_last, n_emb)   # 마지막 레이어 hidden만 조건으로
                    z = torch.randn_like(x_n) if n > 1 else torch.zeros_like(x_n)
                    x_n = self.sched.ddpm_step(x_n, eps_hat, n, z)
                x0 = x_n
                x_prev.append(x0)

                # 3) RNN 한 스텝 전진: 전체 레이어 스택 상태를 넘겨서 업데이트
                x_in = x0.unsqueeze(1)  # (B,1,D)
                _, (h_state, c_state) = self.rnn(x_in, (h_state, c_state))  # 유지/업데이트
                h_last = h_state[-1]    # ε-net 조건용 최신 마지막 레이어 hidden

            samples[:, s] = torch.stack(x_prev, dim=1)  # (B,P,D)

        return samples
