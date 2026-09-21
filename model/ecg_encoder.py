"""共享多阶段 1D ResNet + temporal pyramid + 单次 ECG Transformer。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ResidualBlock1D(nn.Module):
    """使用 GroupNorm，避免依赖训练时的 batch 大小。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        groups = math.gcd(8, out_channels)
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 7, stride=stride, padding=3, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 7, padding=3, bias=False),
            nn.GroupNorm(groups, out_channels),
        )
        self.shortcut = (
            nn.Identity() if in_channels == out_channels and stride == 1 else
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_channels),
            )
        )
        self.activation = nn.GELU()

    def forward(self, waveform: Tensor) -> Tensor:
        return self.activation(self.main(waveform) + self.shortcut(waveform))


class SingleECGEncoder(nn.Module):
    """将一次完整 ECG 编码为事件向量，不包含纵向建模或任务预测头。

    输入为物理量波形 [B, 12, sample_rate * duration_seconds]；导联顺序
    应与 utils.read_ecg 一致。默认 500 Hz、10 秒。每个 stage 对特征图
    划分 4/2/1 个连续区间并均值池化，共 21 个 token，加 CLS 后为 22。
    模块不执行滤波、重采样或缺失值填补。
    """

    def __init__(
        self,
        *,
        sample_rate: int = 500,
        duration_seconds: float = 10.0,
        stage_channels: tuple[int, int, int] = (32, 64, 128),
        embedding_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if sample_rate <= 0 or not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError("sample_rate and duration_seconds must be positive")
        sample_count = sample_rate * duration_seconds
        if not float(sample_count).is_integer() or sample_count < 64:
            raise ValueError("Expected an integer input length of at least 64 samples")
        if len(stage_channels) != 3 or any(channel <= 0 for channel in stage_channels):
            raise ValueError("stage_channels must contain three positive channel counts")
        if num_heads <= 0 or embedding_dim <= 0 or embedding_dim % num_heads:
            raise ValueError("embedding_dim must be positive and divisible by num_heads")
        if num_layers < 1 or not 0 <= dropout < 1:
            raise ValueError("num_layers must be positive and dropout must be in [0, 1)")
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.num_samples = int(sample_count)
        self.output_dim = embedding_dim
        self.bins = (4, 2, 1)
        self.num_tokens = 3 * sum(self.bins)
        self.stem = nn.Sequential(
            nn.Conv1d(12, stage_channels[0], 15, stride=2, padding=7, bias=False),
            nn.GroupNorm(math.gcd(8, stage_channels[0]), stage_channels[0]),
            nn.GELU(),
        )
        self.stages = nn.ModuleList()
        self.projections = nn.ModuleList()
        in_channels = stage_channels[0]
        for channels in stage_channels:
            self.stages.append(nn.Sequential(
                ResidualBlock1D(in_channels, channels, stride=2),
                ResidualBlock1D(channels, channels),
            ))
            self.projections.append(nn.Sequential(
                nn.Linear(channels, embedding_dim), nn.LayerNorm(embedding_dim),
            ))
            in_channels = channels
        stage_ids, scale_ids, intervals = [], [], []
        for stage_index in range(3):
            for scale_index, count in enumerate(self.bins):
                for segment_index in range(count):
                    stage_ids.append(stage_index)
                    scale_ids.append(scale_index)
                    intervals.append((segment_index / count, (segment_index + 1) / count))
        self.register_buffer("stage_ids", torch.tensor(stage_ids, dtype=torch.long))
        self.register_buffer("scale_ids", torch.tensor(scale_ids, dtype=torch.long))
        self.register_buffer("intervals", torch.tensor(intervals, dtype=torch.float32))
        self.stage_embedding = nn.Embedding(3, embedding_dim)
        self.scale_embedding = nn.Embedding(3, embedding_dim)
        self.position_embedding = nn.Sequential(
            nn.Linear(2, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, embedding_dim),
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, embedding_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.input_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(embedding_dim),
            enable_nested_tensor=False,
        )
        for encoder_layer in self.transformer.layers:
            nn.init.xavier_uniform_(encoder_layer.self_attn.in_proj_weight)
            for module in encoder_layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward_features(self, waveform: Tensor) -> dict[str, Tensor]:
        """返回事件向量、21 个聚合前/后 token 及其 stage/scale/时间区间。

        tokens 为投影后的内容特征；context_tokens 为 Transformer 输出。
        intervals_seconds 是 [21, 2] 的名义窗口边界，并非严格感受野边界。
        """
        if waveform.ndim != 3 or waveform.shape[1:] != (12, self.num_samples):
            raise ValueError(f"Expected [B, 12, {self.num_samples}], got {tuple(waveform.shape)}")
        if not waveform.is_floating_point():
            raise TypeError("waveform must be a floating-point tensor in mV")
        features = self.stem(waveform)
        stage_tokens = []
        for stage, projection in zip(self.stages, self.projections):
            features = stage(features)
            pooled = [
                segment.mean(dim=-1)
                for count in self.bins
                for segment in torch.tensor_split(features, count, dim=-1)
            ]
            stage_tokens.append(projection(torch.stack(pooled, dim=1)))
        tokens = torch.cat(stage_tokens, dim=1)
        embeddings = (
            self.stage_embedding(self.stage_ids)
            + self.scale_embedding(self.scale_ids)
            + self.position_embedding(self.intervals)
        )
        encoded_input = tokens + embeddings.unsqueeze(0)
        encoded_input = torch.cat((self.cls_token.expand(waveform.shape[0], -1, -1), encoded_input), dim=1)
        encoded = self.transformer(self.input_dropout(self.input_norm(encoded_input)))
        return {
            "embedding": encoded[:, 0],
            "tokens": tokens,
            "context_tokens": encoded[:, 1:],
            "stage_ids": self.stage_ids,
            "scale_ids": self.scale_ids,
            "intervals_seconds": self.intervals * self.duration_seconds,
        }

    def forward(self, waveform: Tensor) -> Tensor:
        return self.forward_features(waveform)["embedding"]
