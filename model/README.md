# 单次 ECG 编码器

`SingleECGEncoder` 是后续 ECG 纵向模型的组件，不包含 Echo 预测头、跨模态融合或跨检查 Transformer。
默认输入 `[B, 12, 5000]`，输出 `[B, 256]`。

```python
import torch
from model import SingleECGEncoder
from utils import iter_samples, read_ecg

sample = next(iter_samples())
record = read_ecg(sample["ecg"][0]["path"])
waveform = torch.from_numpy(record["signal"]).unsqueeze(0)
encoder = SingleECGEncoder()
encoder.eval()
with torch.no_grad():
    event_embedding = encoder(waveform)
    details = encoder.forward_features(waveform)
```

完整波形只通过一次 backbone：stride=2 的卷积 stem，随后三个各含两个残差块的 stage。
三个 stage 分别输出 `[B, 32, 1250]`、`[B, 64, 625]`、`[B, 128, 313]`。
使用 GroupNorm，不依赖 batch 统计量。

每个 stage 将时间轴分别分为 4、2、1 个不重叠区间，在各区间均值池化，再投影到统一维度。
不能整除时使用长度最多相差一个特征点的区间。因此 2.5/5/10 秒是名义时间范围；卷积和
GroupNorm 会引入窗口之外的上下文，不能将 token 解释为严格隔离片段或指定的生理概念。

21 个 token 按 stage → scale（4/2/1 段）→ 时间顺序排列，加入 stage、scale 的可学习嵌入，
以及根据归一化起止位置生成的时间嵌入。CLS 加入后共 22 个 token，经两层、四头 Transformer
输出事件向量。默认维度 256，前馈维度 1024，dropout 0.1。

`forward_features` 返回：

- `embedding`：`[B, D]`，与 `forward` 一致。
- `tokens`：`[B, 21, D]`，加入标识嵌入之前的内容特征。
- `context_tokens`：`[B, 21, D]`，Transformer 聚合之后的 token。
- `stage_ids`、`scale_ids`：`[21]`，从 0 开始的层级与尺度索引。
- `intervals_seconds`：`[21, 2]`，对应 token 的名义起止秒数。

后续纵向模块可先将有效 ECG 事件合并成一个 batch 调用本模块，再恢复 `[B, N, D]`，
加入距预测时点的时间差和事件掩码。单次模块接收完整、已处理缺失值的浮点波形，不接收补齐的假事件。
导联遵循 `utils.read_ecg` 的标准顺序，单位 mV。模块不自动滤波、重采样或标准化。
如果调整 `duration_seconds`，窗口时长随之变为总长度的 1/4、1/2 和全部。
当前从零初始化，不含预训练权重，也暂不加入 change tokens。

## 纵向 ECG 编码器（第一版）

`LongitudinalECGEncoder` 位于 `ecg_long_encoder.py`，接收单次编码器的事件向量，
默认输出 `[B, 256]`。它独立于单次 backbone，可以读取缓存向量，也可以联合训练。

```text
[B, N, 256] 事件向量 + MLP(log1p(days_to_echo))
                        ↓
               [B, N, 256] time_tokens
                        ↓
         一个可学习 Echo Query 的单头 Cross Attention
                        ↓
                [B, 256] 纵向表征
```

计算为 `x_i = e_i + TimeMLP(log1p(days_i))`，然后
`Q = Wq(q_echo)`、`K_i = Wk(x_i)`、`V_i = Wv(x_i)`；
`alpha = softmax(Q · K_i / sqrt(D))`，输出 `sum_i(alpha_i * V_i)`。
时间 MLP 为 `Linear(1,D) → GELU → Linear(D,D)`，第一版采用单头注意力，
没有纵向 self-attention、change/recent 分支、额外 FFN 或 inter-event interval。
Echo Query 是所有样本共享的可学习参数，不读取真实 Echo 标签，也不是每个任务独立的 Query。

```python
import torch
from model import LongitudinalECGEncoder

long_encoder = LongitudinalECGEncoder(embedding_dim=256)
events = torch.randn(2, 3, 256)
days = torch.tensor([[120.0, 35.0, 2.0], [8.5, 0.25, 0.0]])
mask = torch.tensor([[True, True, True], [True, True, False]])
representation = long_encoder(events, days, mask)  # [2, 256]
details = long_encoder.forward_features(events, days, mask)
weights = details["attention_weights"]  # [2, 3]，补齐位置权重为 0
```

- `event_vectors`：浮点 `[B,N,D]`，D 与构造参数一致。
- `days_to_echo`：`[B,N]`，表示 `(echo_time - ecg_time).total_seconds() / 86400`，
  输入是非负天数（例如 120、35、2），而非 -120、-35、-2；保留小数天。
  模块允许 0 天，严格“Echo 之前”的筛选仍由数据层负责。
- `event_mask`：可选 bool `[B,N]`，True 为真实事件，False 为补齐位置。
  不传表示全部有效。输入张量必须位于相同设备。
- `forward` 返回 `[B,D]`；`forward_features` 额外返回 `[B,N]` 权重及 `[B,N,D]`
  带时间信息的 token。补齐位置 token 为零；有效事件的权重之和为 1。
- 补齐位置的值会先被清除，即使是 NaN 也不会污染结果。有效位置的 NaN/Inf、负时间
  或全空序列会报错；缺失整个 ECG 模态应由未来的多模态层单独处理。
- `log1p` 压缩远期时间差，但不强制模型偏好近期 ECG；注意力权重不是临床可靠性概率。
  没有事件序号嵌入，同步置换事件、时间和掩码后结果保持不变。

### 连接单次与纵向模块

训练时仅将真实事件送入单次编码器，再还原带补齐的 batch，梯度可贯穿两个模块：

```python
from model import SingleECGEncoder, LongitudinalECGEncoder

single_encoder = SingleECGEncoder()
long_encoder = LongitudinalECGEncoder(single_encoder.output_dim)
# waveforms: [B,N,12,5000]，mask: [B,N]，days: [B,N]
valid_embeddings = single_encoder(waveforms[mask])
event_vectors = valid_embeddings.new_zeros(*mask.shape, single_encoder.output_dim)
event_vectors[mask] = valid_embeddings
history_embedding = long_encoder(event_vectors, days, mask)
```

这里只处理 ECG。`history_embedding` 后续可接 CXR 融合模块或任务头。
缓存事件向量适用于冻结单次编码器；联合微调时应重新前向计算以保留梯度。
