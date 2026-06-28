# models/cnn_discriminator/config.py
from dataclasses import dataclass, field
from config_global import GlobalConfig # <-- 글로벌 설정 상속
from typing import List, Tuple

@dataclass
class ModelConfig(GlobalConfig): # <-- 상속
    """CNN Discriminator 모델을 위한 고유 설정"""
    
    # 이 모델의 고유 이름
    MODEL_NAME: str = "CNN_Discriminator"
    
    # --- 모델 파라미터 (고유값) ---
    IMG_SIZE: int = 256
    IN_CHANNELS: int = 3
    LR: float = 0.0002
    BETAS: Tuple[float, float] = (0.5, 0.999) # GAN 스타일 베타
    
    # --- 정규화 (고유값) ---
    # [-1, 1] 스케일링
    MEAN: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    STD: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])

    CHECKPOINT_DIR: str = "./checkpoints"
    
    # 불러올 사전 학습된 가중치 파일 경로 (기본값: None)
    # 이 값을 모델별 config.py에서 덮어쓰면 됨
    PRETRAINED_WEIGHTS: str = None