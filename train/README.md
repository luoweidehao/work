# ECG 历史预测 LVEF：第一版试验

`lvef.py` 将共享 `SingleECGEncoder`、`LongitudinalECGEncoder` 和一个小型回归头连接，
端到端学习 `历史 ECG → 当前 Echo 的 lvef`。回归头为 LayerNorm → Linear(256,64)
→ GELU → Linear(64,1)。当前不使用 CXR，不输入 Echo 的其他标签。

## 标签与划分

- 读取 `echo.lvef.result`，目标单位为百分比；缺失、无法解析、非有限或不在 [0,100] 的值排除并计数。
  不把 `lvef_upper` 当第二个标签，也不把它与 lvef 平均；这是对原始 lvef 字段的基线回归。
- 当前 JSONL 有 9,136 条 lvef 记录，单位均为 `%`。实际纳入数量由每次运行的审计输出决定。
- 按有合格标签的患者随机划分 70%/15%/15% 训练、验证、测试，同一患者的全部 Echo 归入一个集合。
  默认 seed=42，保存具体患者 ID，可复现划分；不是按时间或外部医院验证。
- 目标用训练集均值和标准差标准化，使用 SmoothL1 损失、AdamW、梯度裁剪。
  输出还原为百分比后报告 MAE/RMSE，单位为百分点；同时报告训练集均值预测基线。
- 每个 Echo 样本等权。同患者多条记录并非独立，正式报告还需患者层面的统计评估。

## 波形及显存

读取 ECG mV 波形，固定 `[12,5000]`、500 Hz，沿用读取工具的导联顺序。
时间输入为 `(echo_time - ecg_time)` 的正数天数，只接受过去 180 天的事件。
变长序列补零并附事件掩码，仅真实事件进入单次编码器。
缺文件和异常格式会明确报错。非有限波形点默认在同一导联时间轴上做线性插值，原始有效掩码仍被保留；
整份 ECG 缺失比例超过 1% 或某导联完全无有效值时，跳过该次 ECG，并继续向更早的历史记录寻找，
直到达到 `max-events`。进度和 epoch 指标记录跳过数量；一个样本完全没有可用 ECG 时才停止训练。

默认每个样本保留最近 8 次 ECG（`--max-events 8`），控制端到端反向传播显存；
这是试验的截断策略，不代表已利用完整历史。`--max-events 0` 保留全部，可能显著增加显存。
默认 batch size=2，未冻结任何编码器，从零初始化。CUDA 默认启用 bfloat16 自动混合精度，
其数值范围比 float16 更适合当前从零训练的网络；可用 `--no-amp` 关闭，或显式传
`--amp-dtype fp16` 做对照。FP16 在当前模型的实测首批梯度中出现过非有限值，因此不作为默认值。
DataLoader 在 CUDA 下使用锁页内存和异步传输。

## GPU 与运行

2026-09-21 检查时，物理 GPU 1 有其他 Python 计算任务，GPU 0 无计算任务。
运行前可再次用 `nvidia-smi` 检查；以下命令只暴露物理 GPU 0，在进程内仍叫 `cuda:0`。

在项目根目录执行：

```bash
conda activate echo_multimodal
CUDA_VISIBLE_DEVICES=0 python -m train.lvef \
  --epochs 1 --batch-size 1 --max-events 2 --max-batches 2 \
  --output train/runs/lvef_smoke_new
```

`--max-batches` 大于 0 表示冒烟试跑，每轮只处理指定数量的训练/验证批次，不评估测试集。
试跑只检查流程，其指标不能代表泛化性能。正式训练可用：

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.lvef \
  --epochs 10 --batch-size 2 --max-events 8 \
  --output train/runs/lvef_full
```

输出目录必须不存在，避免覆盖旧试验。写出 `config.json`（参数、标签筛选统计、患者划分、
训练标签统计量）、`metrics.jsonl`、按验证 MAE 选择的 `best.pt`。
完整训练结束后仅用最佳权重评估测试集一次，写出 `test.json`（指标和逐样本预测）。
检查点包含模型和目标缩放信息，不包含优化器状态，当前不提供断点续训。

训练、验证和测试均用 tqdm 展示批次进度、当前标准化 SmoothL1 loss、累计 MAE（百分点）和
跳过的低质量 ECG 数量。
日志重定向或不需要动态进度时可加 `--no-progress`。如果上一次运行在正式训练前失败，输出目录只有
`config.json`，同一命令可以直接重试；一旦产生指标或权重文件，必须指定新的输出目录以免覆盖结果。

推理时先载入 `LVEFRegressor` 权重，再将输出乘 `target_std` 并加 `target_mean`。
输出没有硬裁剪到 [0,100]，评估按原始预测计算。
