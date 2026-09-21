"""读取 process.py 生成的路径，返回通道在前的 NumPy 数组。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STANDARD_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


class ECGQualityError(ValueError):
    """ECG 文件可读取，但缺失值超过允许的质量范围。"""


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    """相对路径默认以项目根目录解析；files/... 路径需传对应数据集根目录。"""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path(root) if root is not None else PROJECT_ROOT) / resolved
    return resolved.resolve()


def iter_samples(
    path: str | Path = "dataset/processed/samples.jsonl",
    *,
    root: str | Path | None = None,
) -> Iterator[dict]:
    """逐行读取样本，避免一次性载入全部 JSONL。"""
    with resolve_path(path, root).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at line {line_number}: {path}") from error
            if not isinstance(sample, dict):
                raise ValueError(f"Expected an object at line {line_number}: {path}")
            yield sample


def read_ecg(
    path: str | Path,
    *,
    root: str | Path | None = None,
    standard_leads: bool = True,
    missing: str = "raise",
    max_missing_fraction: float = 0.01,
) -> dict:
    """读取 WFDB 前缀或 .hea/.dat 路径，返回 float32 [导联, 采样点]。

    使用头文件增益和基线转换为物理量，再统一到 mV。
    missing 可选 raise、keep、zero、interpolate；处理后仍保留原始 valid_mask。
    interpolate 在每个导联的时间轴上做线性插值，并限制最大缺失比例。
    默认不滤波、不标准化、不重采样，避免隐式改变原始波形。
    """
    import wfdb

    if missing not in {"raise", "keep", "zero", "interpolate"}:
        raise ValueError("missing must be 'raise', 'keep', 'zero' or 'interpolate'")
    if not 0 <= max_missing_fraction <= 1:
        raise ValueError("max_missing_fraction must be in [0, 1]")
    record_path = resolve_path(path, root)
    if record_path.suffix.lower() in {".hea", ".dat"}:
        record_path = record_path.with_suffix("")
    if not record_path.with_suffix(".hea").is_file():
        raise FileNotFoundError(f"ECG header not found: {record_path}.hea")
    record = wfdb.rdrecord(str(record_path), physical=True)
    leads = list(record.sig_name)
    if len(set(leads)) != len(leads):
        raise ValueError(f"Duplicate lead names: {leads}")
    if standard_leads:
        absent = set(STANDARD_LEADS) - set(leads)
        if absent:
            raise ValueError(f"Missing ECG leads: {sorted(absent)}")
        indices = [leads.index(lead) for lead in STANDARD_LEADS]
    else:
        indices = list(range(len(leads)))
    signal = record.p_signal[:, indices].T.astype(np.float32, copy=True)
    scales = {"mV": 1.0, "uV": 0.001, "µV": 0.001, "μV": 0.001, "V": 1000.0}
    for channel, source in enumerate(indices):
        unit = record.units[source]
        if unit not in scales:
            raise ValueError(f"Unsupported ECG unit: {unit!r}")
        signal[channel] *= scales[unit]
    valid_mask = np.isfinite(signal)
    if missing == "raise" and not valid_mask.all():
        raise ValueError(
            f"Non-finite ECG values in {record_path}; "
            "use missing='keep', 'zero' or 'interpolate'"
        )
    if missing == "zero":
        signal[~valid_mask] = 0.0
    if missing == "interpolate" and not valid_mask.all():
        invalid_fraction = float((~valid_mask).mean())
        if invalid_fraction > max_missing_fraction:
            raise ECGQualityError(
                f"ECG missing fraction {invalid_fraction:.6f} exceeds "
                f"{max_missing_fraction:.6f}: {record_path}"
            )
        positions = np.arange(signal.shape[1])
        for lead_index, lead_valid in enumerate(valid_mask):
            if lead_valid.all():
                continue
            if not lead_valid.any():
                raise ECGQualityError(f"ECG lead contains no valid values: {record_path}")
            signal[lead_index, ~lead_valid] = np.interp(
                positions[~lead_valid],
                positions[lead_valid],
                signal[lead_index, lead_valid],
            )
    return {
        "signal": np.ascontiguousarray(signal),
        "valid_mask": valid_mask,
        "fs": float(record.fs),
        "duration_seconds": signal.shape[1] / float(record.fs),
        "lead_names": [leads[index] for index in indices],
        "units": ["mV"] * len(indices),
        "path": str(record_path),
    }


def read_cxr(
    path: str | Path,
    *,
    root: str | Path | None = None,
    mode: str = "L",
    size: tuple[int, int] | None = None,
) -> dict:
    """读取 JPG/PNG，返回 float32 [通道, 高, 宽]，数值范围 [0, 1]。

    mode 为 L 或 RGB；size 为 (高, 宽)。缩放保持比例并居中补黑边，
    valid_mask 为 [1, 高, 宽]，区分图像区域和补边。此函数不读取 DICOM。
    """
    from PIL import Image, ImageOps

    if mode not in {"L", "RGB"}:
        raise ValueError("mode must be 'L' or 'RGB'")
    if size is not None and (
        len(size) != 2 or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in size)
    ):
        raise ValueError("size must be two positive integers: (height, width)")
    image_path = resolve_path(path, root)
    with Image.open(image_path) as source:
        picture = source.convert(mode)
        original_size = (source.height, source.width)
    if size is None:
        mask = np.ones((1, picture.height, picture.width), dtype=bool)
    else:
        height, width = size
        picture = ImageOps.contain(picture, (width, height), Image.Resampling.BILINEAR)
        left = (width - picture.width) // 2
        top = (height - picture.height) // 2
        mask = np.zeros((1, height, width), dtype=bool)
        mask[:, top:top + picture.height, left:left + picture.width] = True
        canvas = Image.new(mode, (width, height), color=0)
        canvas.paste(picture, (left, top))
        picture = canvas
    pixels = np.asarray(picture, dtype=np.float32) / 255.0
    pixels = pixels[None, ...] if mode == "L" else pixels.transpose(2, 0, 1)
    return {
        "image": np.ascontiguousarray(pixels),
        "valid_mask": mask,
        "original_size": original_size,
        "mode": mode,
        "path": str(image_path),
    }
