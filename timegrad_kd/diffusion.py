# diffusion.py
import torch
import math

class LinearBetaSchedule:
    def __init__(self, n_steps: int, beta_start: float, beta_end: float):
        self.n = n_steps
        self.register_buffers = {}
        beta = torch.linspace(beta_start, beta_end, n_steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.beta = beta
        self.alpha = alpha
        self.alpha_bar = alpha_bar

    def to(self, device):
        for name, tensor in [("beta", self.beta), ("alpha", self.alpha), ("alpha_bar", self.alpha_bar)]:
            setattr(self, name, tensor.to(device))
        return self

    def sample_noisy(self, x0: torch.Tensor, n_idx: torch.Tensor, eps: torch.Tensor):
        # x_n = sqrt(alpha_bar_n)*x0 + sqrt(1-alpha_bar_n)*eps
        a_bar = self.alpha_bar[n_idx-1].view(-1, 1)
        return torch.sqrt(a_bar)*x0 + torch.sqrt(1.0 - a_bar)*eps

    def ddpm_step(self, x_n: torch.Tensor, eps_theta: torch.Tensor, n: int, z: torch.Tensor):
        # Algorithm 2 (논문 식, 표기 동일) :contentReference[oaicite:9]{index=9}
        alpha_n = self.alpha[n-1]
        alpha_bar_n = self.alpha_bar[n-1]
        coef = 1.0 / torch.sqrt(alpha_n)
        mean = coef * (x_n - ( (1 - alpha_n) / torch.sqrt(1 - alpha_bar_n) ) * eps_theta)
        if n > 1:
            sigma = torch.sqrt((1 - alpha_n) * (1 - self.alpha_bar[n-2]) / (1 - alpha_bar_n))
            return mean + sigma * z
        else:
            return mean
