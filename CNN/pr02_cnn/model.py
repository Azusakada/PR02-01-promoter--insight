from __future__ import annotations

import torch
from torch import nn


class PromoterCNN(nn.Module):
    """A small position-sensitive CNN; two same-length stride-1 convolutions."""
    def __init__(self, cfg):
        super().__init__()
        self.sequence_length = cfg["sequence_length"]
        if type(self.sequence_length) is not int or self.sequence_length < 1 or cfg["channels"] != list("ACGT"):
            raise ValueError("CNN需要正整数序列长度，通道ACGT")
        widths, kernels = cfg["conv_channels"], cfg["kernel_sizes"]
        if len(widths) != 2 or len(kernels) != 2 or any(k % 2 != 1 or k < 1 for k in kernels):
            raise ValueError("需要两个正奇数卷积核")
        if any(type(x) is not int or x <= 0 for x in [*widths, cfg["hidden_dim"]]):
            raise ValueError("模型维度必须是正整数")
        if not 0 <= cfg["dropout"] < 1:
            raise ValueError("dropout范围为[0,1)")
        pooling = cfg.get("pooling", "flatten")
        pooled_length = cfg.get("pooled_length", self.sequence_length)
        if type(pooled_length) is not int or not 1 <= pooled_length <= self.sequence_length:
            raise ValueError("pooled_length必须在1到序列长度之间")
        if pooling == "flatten":
            if pooled_length != self.sequence_length:
                raise ValueError("flatten必须保留全部序列位置")
            self.pool = nn.Identity()
        elif pooling == "adaptive_avg":
            self.pool = nn.AdaptiveAvgPool1d(pooled_length)
        elif pooling == "adaptive_max":
            self.pool = nn.AdaptiveMaxPool1d(pooled_length)
        else:
            raise ValueError("pooling必须为flatten/adaptive_avg/adaptive_max")
        self.features = nn.Sequential(
            nn.Conv1d(4, widths[0], kernels[0], padding=kernels[0] // 2), nn.ReLU(),
            nn.Conv1d(widths[0], widths[1], kernels[1], padding=kernels[1] // 2), nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(widths[1] * pooled_length, cfg["hidden_dim"]), nn.ReLU(),
            nn.Dropout(cfg["dropout"]), nn.Linear(cfg["hidden_dim"], 1),
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (4, self.sequence_length):
            raise ValueError(f"输入shape错误: {tuple(x.shape)}，要求(N,4,{self.sequence_length})")
        # No sigmoid: validation/test targets may lie outside the train range.
        return self.regressor(self.pool(self.features(x))).squeeze(-1)


class Log10Predictor(nn.Module):
    """Differentiable wrapper for attribution in the declared sequence coordinates."""
    def __init__(self, model: PromoterCNN, transform: dict):
        super().__init__()
        self.model = model
        self.a = float(transform["min_log10"])
        self.span = float(transform["max_log10"] - transform["min_log10"])

    def forward(self, x):
        return self.a + self.span * self.model(x)
