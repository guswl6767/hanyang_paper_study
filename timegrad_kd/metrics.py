# metrics.py
import torch

@torch.no_grad()
def crps_empirical(x_samples: torch.Tensor, x_obs: torch.Tensor):
    """
    CRPS(F,x) = E|X - x| - 0.5 E|X - X'|,  (표본 기반 근사)
    x_samples: (S,) , x_obs: ()
    """
    S = x_samples.shape[0]
    term1 = torch.mean(torch.abs(x_samples - x_obs))
    diffs = torch.abs(x_samples.unsqueeze(0) - x_samples.unsqueeze(1))
    term2 = 0.5 * torch.mean(diffs)
    return term1 - term2

@torch.no_grad()
def crps_sum_over_horizon(samples: torch.Tensor, target: torch.Tensor):
    """
    samples: (S,P,D), target: (P,D)
    1) 각 시점 t에 대해 합산 차원: sum over D
    2) t별 CRPS 계산 후 평균
    """
    S, P, D = samples.shape
    s_sum = samples.sum(dim=2)          # (S,P)
    y_sum = target.sum(dim=1)           # (P,)
    crps_t = []
    for t in range(P):
        crps_t.append(crps_empirical(s_sum[:, t], y_sum[t]))
    return torch.stack(crps_t).mean().item()
