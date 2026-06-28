# models/dual_stream_xception/config.py
from dataclasses import dataclass, field
from config_global import GlobalConfig
from typing import List, Tuple

@dataclass
class ModelConfig(GlobalConfig):
    """Two-Stream XceptionNet (Spatial + Frequency) 모델 설정"""
    
    MODEL_NAME: str = "DualStream_XceptionNet"
    
    IMG_SIZE: int = 299 
    IN_CHANNELS: int = 3
    LR: float = 0.0001
    BETAS: Tuple[float, float] = (0.9, 0.999)

    MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    CHECKPOINT_DIR: str = "./checkpoints"
    PRETRAINED_WEIGHTS: str = None