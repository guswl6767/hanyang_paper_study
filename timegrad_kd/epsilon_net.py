# epsilon_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FourierNoiseEmbedding(nn.Module):
    # Transformer의 Fourier positional encoding 아이디어로 노이즈 스텝 n 내장. (dim=32) :contentReference[oaicite:10]{index=10}
    def __init__(self, dim: int = 32, max_n: int = 500):
        super().__init__()
        self.dim = dim
        self.max_n = max_n
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, n_idx: torch.Tensor):  # (B,)
        # n_idx ∈ {1..N}
        n = n_idx.float().unsqueeze(1)  # (B,1)
        sinus = n * self.inv_freq  # (B, dim/2)
        emb = torch.cat([torch.sin(sinus), torch.cos(sinus)], dim=-1)  # (B, dim)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv_f = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, padding_mode="circular")
        self.conv_g = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, padding_mode="circular")
        self.res = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        f = torch.tanh(self.conv_f(x))
        g = torch.sigmoid(self.conv_g(x))
        h = f * g
        return self.res(h) + x, self.skip(h)

class EpsilonNet(nn.Module):
    """
    입력: (B, D) noisy x_n, 조건: h_{t-1} (B, H), n-embedding (B, E)
    처리: (B, C, D) with residual dilated Conv1D blocks, gating, skip-sum
    출력: (B, D) 예측 노이즈 ε_theta
    """
    def __init__(self, D: int, cond_dim: int, noise_emb_dim: int = 32, residual_channels: int = 8, residual_blocks: int = 8):
        super().__init__()
        self.D = D
        self.c_in = nn.Conv1d(1, residual_channels, kernel_size=1)
        self.cond_proj = nn.Linear(cond_dim, residual_channels)
        self.noise_proj = nn.Linear(noise_emb_dim, residual_channels)

        blocks = []
        for b in range(residual_blocks):
            dilation = 2 ** (b % 2)  # 1,2 반복 (논문) :contentReference[oaicite:11]{index=11}
            blocks.append(ResidualBlock(residual_channels, dilation))
        self.blocks = nn.ModuleList(blocks)

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(residual_channels, residual_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(residual_channels, 1, kernel_size=1),
        )

    def forward(self, x_nd: torch.Tensor, h: torch.Tensor, n_emb: torch.Tensor):
        # x_nd: (B, D) -> (B, 1, D)
        B, D = x_nd.shape
        x = x_nd.unsqueeze(1)
        x = self.c_in(x)

        # 조건(컨텍스트 RNN h, noise emb)을 채널 바이어스로 주입 (FiLM-like)
        cond = self.cond_proj(h).unsqueeze(-1)  # (B, C, 1)
        nfe = self.noise_proj(n_emb).unsqueeze(-1)  # (B, C, 1)
        x = x + cond + nfe

        skip_sum = 0
        for blk in self.blocks:
            x, s = blk(x)
            skip_sum = skip_sum + s

        out = self.out(skip_sum)  # (B,1,D)
        return out.squeeze(1)     # (B,D)
