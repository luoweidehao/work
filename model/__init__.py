"""多模态模型的独立编码组件。"""

from .ecg_encoder import SingleECGEncoder
from .ecg_long_encoder import LongitudinalECGEncoder

__all__ = ["SingleECGEncoder", "LongitudinalECGEncoder"]
