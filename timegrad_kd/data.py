# data.py
from typing import Tuple, Iterable, Dict, Any
import numpy as np
import pandas as pd
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.time_feature import time_features_from_frequency_str
from datetime import datetime

def load_multivariate(name: str) -> Tuple[Any, Any, Dict]:
    ds = get_dataset(name)  # train/test/metadata
    # 한 개의 멀티변량 시계열로 묶기
    grouper = MultivariateGrouper(max_target_dim=None)
    train_mv = grouper(ds.train)
    test_mv  = grouper(ds.test)
    return train_mv, test_mv, ds.metadata  # metadata.freq, prediction_length 등

def make_time_features(start: pd.Period, freq: str, length: int):
    # GluonTS 기본 시간피처 사용 (freq에 맞춰 자동 구성)
    feats_fns = time_features_from_frequency_str(freq)
    idx = pd.period_range(start=start, periods=length, freq=freq)
    feats = [fn(idx) for fn in feats_fns]  # list of (length,)
    if len(feats)==0: 
        return np.zeros((0, length), dtype=np.float32)
    return np.stack(feats, axis=0).astype(np.float32)  # (C, length)

def split_context_pred(target: np.ndarray, context_len: int, pred_len: int):
    # target: (D, T)
    D, T = target.shape
    assert T >= context_len + pred_len
    ctx = target[:, T-(context_len+pred_len): T-pred_len]
    fut = target[:, T-pred_len:]
    return ctx.astype(np.float32), fut.astype(np.float32)  # (D, C), (D, P)

def scale_by_context_mean(x_ctx: np.ndarray, x_fut: np.ndarray):
    # 논문 3.3 Scaling: 컨텍스트 평균(0이면 1)으로 나눔 후 추론 시 재곱셈. :contentReference[oaicite:7]{index=7}
    mean = x_ctx.mean(axis=1, keepdims=True)
    mean[mean==0] = 1.0
    return x_ctx/mean, x_fut/mean, mean

def iter_rolling_test(test_mv, pred_len: int):
    # GluonTS test는 롤링 윈도우를 포함. 각 엔트리에 대해 마지막 pred_len를 예측.
    for item in test_mv:
        target = item["target"]  # (D, T_i)
        start: pd.Period = item["start"]
        freq = item["feat_static_cat"] if "feat_static_cat" in item else None
        yield item, target, start
