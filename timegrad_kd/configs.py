# configs.py
from dataclasses import dataclass

DATASETS = {
    # 논문 Table 1 기준: (freq, pred_len)
    # Exchange: D=8, DAY, pred 30
    "exchange_rate": ("D", 30),
    # Solar: D=137, HOUR, pred 24
    "solar_nips": ("H", 24),
    # Electricity: D=370, HOUR, pred 24
    "electricity_nips": ("H", 24),
    # Traffic: D=963, HOUR, pred 24
    "traffic_nips": ("H", 24),
    # Taxi: D=1214, 30-MIN, pred 24
    "taxi_30min": ("30min", 24),
    # Wikipedia rolling: D=2000, DAY, pred 30
    "wiki-rolling_nips": ("D", 30),
}

@dataclass
class TimeGradPaperCfg:
    # 논문 기본값
    n_diffusion_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 1e-1
    lstm_hidden: int = 40        # h_t ∈ R^40
    lstm_layers: int = 2
    noise_emb_dim: int = 32      # Fourier positional embedding
    residual_channels: int = 8   # ε 네트워크 채널
    residual_blocks: int = 8
    batch_size: int = 64
    lr: float = 1e-3
    num_samples_eval: int = 100  # S=100
    context_equals_pred: bool = True  # context=len(pred)
