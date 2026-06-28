# models/xceptionnet/model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from .config import ModelConfig
import timm


from torchvision.transforms import v2, InterpolationMode
import random
import io
from PIL import Image

class CustomRandomJPEG:
    """
    랜덤 JPEG 압축을 적용하는 데이터 증강 클래스.
    PIL 이미지를 받아 랜덤 품질로 JPEG 압축을 적용합니다.
    """
    def __init__(self, quality, p=0.5):
        self.quality_min, self.quality_max = quality
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        img_jpeg = Image.open(buffer)
        return img_jpeg

class Model(nn.Module):
    """
    XceptionNet 기반 딥페이크 탐지 모델
    timm의 legacy_xception을 백본으로 사용합니다.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Xception 백본 로드 (사전 학습된 가중치 사용)
        self.backbone = timm.create_model(
            'legacy_xception', 
            pretrained=True, 
            num_classes=0, 
            global_pool=''
        )
        
        in_features = self.backbone.num_features

        # 분류 헤드
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        # Loss 함수 (클래스 불균형을 위한 pos_weight 사용)
        pos_weight_val = getattr(config, 'POS_WEIGHT', 1.0)
        pos_weight = torch.tensor([pos_weight_val]).to(config.DEVICE)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # 옵티마이저 및 스케줄러
        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=config.LR, 
            betas=config.BETAS,
            weight_decay=1e-5
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=1e-6
        )

        print(f"[Model Initialized] {config.MODEL_NAME} (timm Backbone) - Augmentation Version")
        print("-> Pretrained backbone weights loaded via timm.")
        print(f"-> Dropout(0.5) 및 WeightDecay(1e-5) 적용.")
        print(f"-> 비율 무시 Resize 및 강력한 증강을 훈련에 적용합니다.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x) 
        return x

    def step(self, images, labels):
        """학습 스텝: forward pass 및 loss 계산"""
        preds = self.forward(images) 
        loss = self.criterion(preds, labels)
        return preds, loss 

    @torch.no_grad()
    def predict(self, x, threshold: float = 0.5):
        """예측 수행"""
        probs = torch.sigmoid(self.forward(x))
        return (probs >= threshold).long()

    @staticmethod
    def get_transforms(config: ModelConfig, is_train: bool = True) -> v2.Compose:
        """
        이미지 변환 파이프라인을 생성합니다.
        
        Args:
            config: 모델 설정
            is_train: 학습 모드 여부 (True면 증강 적용)
        
        Returns:
            변환 파이프라인
        """
        transforms_list = []
        img_size_tuple = (config.IMG_SIZE, config.IMG_SIZE)

        if is_train:
            # 비율 무시 리사이즈
            transforms_list.append(v2.Resize(img_size_tuple, interpolation=InterpolationMode.BICUBIC, antialias=True))
            
            # 기하학적 증강
            transforms_list.append(v2.RandomHorizontalFlip(p=0.7))
            transforms_list.append(v2.RandomApply([
                v2.RandomAffine(
                    degrees=15,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                )
            ], p=0.5))

            # 색상 증강
            transforms_list.append(v2.RandomApply([
                v2.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            )
            ], p=0.5))
            
            # 품질/노이즈 증강
            transforms_list.append(CustomRandomJPEG(quality=(10, 30), p=0.7))
            transforms_list.append(v2.RandomApply([
                v2.GaussianBlur(kernel_size=(5, 9))
            ], p=0.6))

            # 텐서 변환 및 정규화
            transforms_list.extend([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=config.MEAN, std=config.STD),
                v2.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            ])
        
        else:
            # 검증 모드: 증강 없이 리사이즈 및 정규화만
            transforms_list.append(v2.Resize(img_size_tuple, interpolation=InterpolationMode.BICUBIC, antialias=True))

            transforms_list.extend([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=config.MEAN, std=config.STD),
            ])
        
        return v2.Compose(transforms_list)