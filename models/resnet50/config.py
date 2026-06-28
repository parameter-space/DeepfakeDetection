# models/resnet50/config.py
from dataclasses import dataclass, field
from config_global import GlobalConfig # <-- 글로벌 설정 상속
from typing import List, Tuple

@dataclass
class ModelConfig(GlobalConfig):
    """
    ResNet50 (구조 직접 구현) 모델을 위한 고유 설정
    """
    
    # 이 모델의 고유 이름
    MODEL_NAME: str = "ResNet50_Custom"
    
    # --- 모델 파라미터 (고유값) ---
    # ResNet 계열은 224x224를 표준으로 사용
    IMG_SIZE: int = 224 
    IN_CHANNELS: int = 3
    LR: float = 0.0001 # 전이 학습이므로 LR을 낮게 설정
    BETAS: Tuple[float, float] = (0.9, 0.999) # Adam 기본값
    
    # ImageNet 표준 정규화 값
    MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    CHECKPOINT_DIR: str = "./checkpoints"
    
    # main.py가 이 파일을 로드하지 않도록 None으로 둡니다.
    # 모델 파일 내에서 직접 torchvision의 가중치만 불러올 것입니다.
    PRETRAINED_WEIGHTS: str = None
    
    # 모델 파일이 torchvision 가중치를 로드할지 여부
    LOAD_TORCHVISION_WEIGHTS: bool = True