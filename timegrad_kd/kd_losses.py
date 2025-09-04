# kd_losses.py
import torch
import torch.nn.functional as F

def score_matching_kd(eps_s: torch.Tensor, eps_t: torch.Tensor, weight: float = 1.0):
    return weight * F.mse_loss(eps_s, eps_t)

def one_step_update_kd(xn_s_next: torch.Tensor, xn_t_next: torch.Tensor, weight: float = 0.1):
    return weight * F.mse_loss(xn_s_next, xn_t_next)
