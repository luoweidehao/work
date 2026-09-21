"""事件向量 + 距 Echo 时间嵌入 + 单 Query attention 的纵向 ECG 基线。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class LongitudinalECGEncoder(nn.Module):
    """将 [B, N, D] 历史 ECG 表征汇总为 [B, D]。

    days_to_echo 为 [B, N] 非负天数，支持小数；event_mask 为同形状
    bool 张量，True 表示有效事件。每个样本至少需要一个有效事件。
    Query 是共享可学习参数，不使用 Echo 标签或 Echo 检查结果。
    """

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        if not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        self.output_dim = embedding_dim
        self.time_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.echo_query = nn.Parameter(torch.empty(1, embedding_dim))
        nn.init.normal_(self.echo_query, std=0.02)
        self.query_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward_features(
        self,
        event_vectors: Tensor,
        days_to_echo: Tensor,
        event_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """返回 embedding [B,D]、attention_weights [B,N] 和 time_tokens [B,N,D]。"""
        if event_vectors.ndim != 3 or event_vectors.shape[-1] != self.output_dim:
            raise ValueError(f"Expected event_vectors [B, N, {self.output_dim}]")
        if not event_vectors.is_floating_point():
            raise TypeError("event_vectors must be floating point")
        if event_vectors.shape[0] == 0 or event_vectors.shape[1] == 0:
            raise ValueError("Batch and event dimensions must be nonempty")
        if days_to_echo.shape != event_vectors.shape[:2]:
            raise ValueError("days_to_echo must have shape [B, N]")
        if days_to_echo.device != event_vectors.device:
            raise ValueError("days_to_echo and event_vectors must be on the same device")
        if days_to_echo.is_complex() or days_to_echo.dtype == torch.bool:
            raise TypeError("days_to_echo must contain real numeric days")
        if event_mask is None:
            event_mask = torch.ones_like(days_to_echo, dtype=torch.bool)
        if event_mask.shape != days_to_echo.shape or event_mask.dtype != torch.bool:
            raise ValueError("event_mask must be bool with shape [B, N]")
        if event_mask.device != event_vectors.device:
            raise ValueError("event_mask and event_vectors must be on the same device")
        if not event_mask.any(dim=1).all():
            raise ValueError("Each sample must contain at least one valid ECG event")
        valid_days = days_to_echo[event_mask]
        if not torch.isfinite(valid_days).all() or (valid_days < 0).any():
            raise ValueError("Valid days_to_echo must be finite and nonnegative")
        if not torch.isfinite(event_vectors[event_mask]).all():
            raise ValueError("Valid event_vectors must be finite")
        safe_vectors = event_vectors.masked_fill(~event_mask.unsqueeze(-1), 0)
        safe_days = days_to_echo.masked_fill(~event_mask, 0).to(dtype=torch.float32)
        time_features = torch.log1p(safe_days).unsqueeze(-1).to(dtype=event_vectors.dtype)
        time_tokens = safe_vectors + self.time_embedding(time_features)
        time_tokens = time_tokens.masked_fill(~event_mask.unsqueeze(-1), 0)
        query = self.query_projection(self.echo_query)
        keys = self.key_projection(time_tokens)
        values = self.value_projection(time_tokens)
        scores = (keys.float() * query.float()).sum(dim=-1) / math.sqrt(self.output_dim)
        scores = scores.masked_fill(~event_mask, float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1).to(dtype=values.dtype)
        embedding = torch.sum(attention_weights.unsqueeze(-1) * values, dim=1)
        return {
            "embedding": embedding,
            "attention_weights": attention_weights,
            "time_tokens": time_tokens,
        }

    def forward(
        self,
        event_vectors: Tensor,
        days_to_echo: Tensor,
        event_mask: Tensor | None = None,
    ) -> Tensor:
        return self.forward_features(event_vectors, days_to_echo, event_mask)["embedding"]
