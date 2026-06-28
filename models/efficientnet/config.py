# models/efficientnet_b4/config.py
from dataclasses import dataclass, field
from config_global import GlobalConfig  #
from typing import List, Tuple

@dataclass
class ModelConfig(GlobalConfig): # <-- GlobalConfig 상속
    """EfficientNet-B4 모델을 위한 고유 설정"""
    
    # 이 모델의 고유 이름
    MODEL_NAME: str = "EfficientNet_B4"
    
    # EfficientNet-B4의 표준 입력 크기
    IMG_SIZE: int = 380 

    # --- 모델 파라미터 (고유값) ---
    BATCH_SIZE: int = 24  # B4는 Xception보다 무거우므로 배치 크기를 약간 줄입니다.
    ACCUMULATION_STEPS: int = 2 
    IN_CHANNELS: int = 3
    LR: float = 0.0001
    BETAS: Tuple[float, float] = (0.9, 0.999)

    # --- 정규화 (ImageNet 통계치) ---
    MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    
    CHECKPOINT_DIR: str = "./checkpoints"
    PRETRAINED_WEIGHTS: str = None