# ECG / CXR 读取与预处理

依赖：`numpy`、`wfdb`、`Pillow`。当前 `echo_multimodal` conda 环境已有这些依赖。
从项目根目录运行：

```bash
conda activate echo_multimodal
python -m utils.inspect_sample --index 0
```

## 在训练代码中读取

```python
from utils import iter_samples, read_ecg, read_cxr

sample = next(iter_samples())
ecg = read_ecg(sample["ecg"][0]["path"])
cxr = read_cxr(sample["cxr"][0]["path"], mode="RGB", size=(224, 224))

waveform = ecg["signal"]
image = cxr["image"]
```

- ECG 返回 `float32 [12, 5000]`（具体长度依文件而定）、`fs`、导联名、mV 单位、时长和逐点有效掩码。
- CXR 返回 `float32 [1, H, W]` 或 `[3, H, W]`，范围 `[0, 1]`，附原始尺寸 `(H, W)` 和有效图像区域掩码。RGB 将灰度图转换为三通道。
- 默认以项目根目录解析相对路径，不依赖运行目录；绝对路径直接使用。
- 如果输入是原始表中的 `files/...` 路径，传 `root="/absolute/path/to/dataset/mimic-ecg"` 或 CXR 根目录。
- ECG 接受无后缀前缀、`.hea` 或 `.dat`，由 WFDB 读取配套文件。缺文件直接报错。
- ECG 默认遇到 NaN/Inf 报错；检查时可用 `missing="keep"`，也支持 `zero` 和逐导联时间插值
  `interpolate`。所有方式都返回处理前的 `valid_mask`；插值默认拒绝缺失比例超过 1% 的记录。
  超过阈值会抛出 `ECGQualityError`，训练数据层可据此跳过整次低质量 ECG，而不会隐藏路径、格式或导联错误。
- CXR 的 `size=(高度, 宽度)` 使用等比例缩放、居中黑色补边；不做自动裁剪、直方图均衡或骨干网络特定标准化。这里读取 JPG/PNG，不负责 DICOM 解码。

## 本地抽样结果

2026-09-21，取 JSONL 前 30 个样本中的前 40 个去重 ECG 路径进行实际解码：
40 个均为 500 Hz、5000 点、12 导联、单位 mV，未发现 NaN/Inf。这是抽样结果，不是全量质量保证。
首个记录为 `42790963`，头文件格式为 16 位、增益 200 ADC/mV、基线 0；实际物理幅度范围为 -0.805 至 1.08 mV。
默认使用 WFDB 的物理量转换，不对已转换的数据再次除以增益。

抽样原始导联顺序为 `I, II, III, aVR, aVF, aVL, V1, ..., V6`；
工具默认按名字重排为 `I, II, III, aVR, aVL, aVF, V1, ..., V6`。
使用预训练模型时必须与其导联约定核对；`standard_leads=False` 可保留源顺序。
首个样本的三张图均为 JPEG、灰度 L、宽 2544 × 高 3056。

官方数据说明：https://physionet.org/content/mimic-iv-ecg/1.0/

## ECG 处理建议

1. **单次检查的波形维度**：先按头文件转换到 mV、固定导联顺序，检查缺失、平直导联和异常幅值。
   第一版保留完整 10 秒和 500 Hz；若使用预训练骨干，采样率、输入长度及标准化均遵循该骨干约定。
   需要降采样时使用带抗混叠处理的重采样，不能直接隔点抽取。
2. **滤波与标准化**：读取层保留原信号。是否去基线、带通或陷波应结合实际噪声和预训练要求验证，
   不预设所有记录都需要相同滤波。逐条、逐导联 z-score 会改变幅度信息；若采用训练集统计量，
   只能在训练患者上拟合，并原样应用于验证/测试集。
3. **跨检查的纵向维度**：一次 ECG 是 `[12, 5000]`，同一 Echo 前的 N 次 ECG 是 N 个独立事件。
   单次编码器分别产生 `[N, D]` 表征，再结合距预测时点的真实时间差做时间池化或轻量时序编码。
   不把相隔数天的波形沿采样轴拼成一条连续信号，也不把事件序号当真实时间。
4. **批处理**：固定单次长度后可组织为 `[B, N_max, 12, 5000]`，并另外提供 `[B, N_max]` 事件掩码
   与时间差；波形的 `valid_mask` 和事件补齐掩码含义不同。可先缓存按 study 去重的单次表征，减少重复解码。
5. **时间和拆分**：ECG/Echo 当前时间戳无时区，CXR 含偏移；时间差计算之前需确认并统一时间语义，
   不凭空转换成 UTC。患者级划分训练/验证/测试，预测时点之后的检查不可作为输入。

额外观察：首个样本的部分 AP 图像检查描述为 `PORTABLE ABDOMEN`。
仅按 AP 视位筛选不能保证每张都是胸片，后续训练前需要单独审核图像类型；读取工具不改变队列。
