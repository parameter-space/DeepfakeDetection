# models/dinov2_vitb14/config.py
from dataclasses import dataclass, field
from config_global import GlobalConfig
from typing import List, Tuple


@dataclass
class ModelConfig(GlobalConfig):
    """
    [신규] DINOv2 (ViT-Small, Patch 14) 모델을 위한 설정
    - ViT-Small (vit_small_patch14_dinov2) 백본 사용
    - 네이티브 해상도 518x518로 학습
    """
    
    # [수정] 모델 이름 변경 (Base -> Small)
    MODEL_NAME: str = "DINOv2_ViT_Small14"
    
    # --- 모델 파라미터 ---
    # DINOv2-Small14의 네이티브 학습 해상도 (14 * 37 = 518)
    IMG_SIZE: int = 518 
    IN_CHANNELS: int = 3
    LR: float = 1e-05 
    BETAS: Tuple[float, float] = (0.9, 0.999)

    # [수정] ViT-Small은 가벼우므로, BATCH_SIZE 8 / G-A 4로 복구
    BATCH_SIZE: int = 8
    ACCUMULATION_STEPS: int = 4
    
    # --- 정규화 (ImageNet) ---
    MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    CHECKPOINT_DIR: str = "./checkpoints"
    
    # --- [핵심] timm 모델 로드 설정 ---
    
    # [수정] 님의 요청대로 ViT-Small 모델로 변경
    TIMM_MODEL_NAME: str = "vit_small_patch14_dinov2.lvd142m"
    
    LOAD_TIMM_PRETRAINED: bool = True 
    PRETRAINED_WEIGHTS: str = None