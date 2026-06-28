# models/xceptionnet/config.py
from dataclasses import dataclass, field
from config_global import GlobalConfig
from typing import List, Tuple


@dataclass
class ModelConfig(GlobalConfig):
    """
    XceptionNet 모델을 위한 설정 클래스
    GlobalConfig를 상속받아 모델별 고유 설정을 정의합니다.
    """
    
    MODEL_NAME: str = "XceptionNet_Phase2"
    
    BATCH_SIZE: int = 32

    ACCUMULATION_STEPS: int = 2 

    LR: float = 0.0001  # 학습률

    IMG_SIZE: int = 299  # Xception 입력 이미지 크기
    IN_CHANNELS: int = 3
    BETAS: Tuple[float, float] = (0.9, 0.999)  # Adam 옵티마이저 beta 파라미터

    # ImageNet 통계치 (정규화용)
    MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    CHECKPOINT_DIR: str = "./checkpoints"
    # PRETRAINED_WEIGHTS: str = "./checkpoints/xceptionnet/xceptionnet_best.pth"

    POS_WEIGHT: float = 1.05  # 클래스 불균형을 위한 positive class 가중치

    BEST_WEIGHT_LOAD: bool = False  # 최고 성능 체크포인트 로드 여부