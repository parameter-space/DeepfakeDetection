# models/efficientnet_b4/model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from .config import ModelConfig  # <-- efficientnet_b4의 config를 임포트
import timm

from torchvision.transforms import v2, InterpolationMode
import random
import io
from PIL import Image

# [신규] 'efficientnet' 모델에서 CustomRandomJPEG 클래스를 가져옴
# (v2.JPEG는 PIL 이미지를 받지 못하므로, PIL용 커스텀 클래스 필요)
class CustomRandomJPEG:
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
# --- [클래스 종료] ---


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 1. [핵심 수정] timm에서 EfficientNet-B4 백본 로드 (pretrained=True)
        self.backbone = timm.create_model(
            'efficientnet_b4', # <-- 모델 이름 변경
            pretrained=True, 
            num_classes=0, 
            global_pool=''
        )
        
        # [핵심 수정] EfficientNet-B4의 출력 피처 수는 1792
        in_features = self.backbone.num_features # 1792

        # 3. Head (분류기) - (1792->512->1)
        # (XceptionNet과 동일한 헤드 구조 사용)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512), # <-- 2048에서 1792로 변경
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        # 4. criterion, optimizer, scheduler 정의 (XceptionNet과 동일)
        self.criterion = nn.BCEWithLogitsLoss()
        
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
        print(f"-> [규제] Dropout(0.5) 및 WeightDecay(1e-5) 적용.")
        print(f"-> [핵심] '비율 무시 Resize (Squash)' 및 '강력한 증강'을 훈련에 적용합니다.")

    # forward, step, predict는 XceptionNet과 완벽히 동일
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x) 
        return x

    def step(self, images, labels):
        preds = self.forward(images) 
        loss = self.criterion(preds, labels)
        return preds, loss 

    @torch.no_grad()
    def predict(self, x, threshold: float = 0.5):
        probs = torch.sigmoid(self.forward(x))
        return (probs >= threshold).long()

    @staticmethod
    def get_transforms(config: ModelConfig, is_train: bool = True) -> v2.Compose:
        """
        [수정 없음]
        XceptionNet의 증강 로직을 그대로 사용합니다.
        config.IMG_SIZE (380)를 자동으로 참조하므로 코드를 수정할 필요가 없습니다.
        """
        transforms_list = []
        # config.IMG_SIZE가 380이므로 (380, 380)이 됩니다.
        img_size_tuple = (config.IMG_SIZE, config.IMG_SIZE) 

        if is_train:
            # --- 훈련용 증강 스택 ---
            transforms_list.append(v2.Resize(img_size_tuple, interpolation=InterpolationMode.BICUBIC, antialias=True))
            transforms_list.append(v2.RandomHorizontalFlip(p=0.5))
            transforms_list.append(v2.RandomApply([
                v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ], p=0.8))
            transforms_list.append(v2.RandomApply([
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=0.8))
            transforms_list.append(CustomRandomJPEG(quality=(40, 90), p=0.5))
            transforms_list.append(v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.3))
            transforms_list.extend([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=config.MEAN, std=config.STD),
                v2.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            ])
        
        else:
            # --- 검증용 (깨끗한) 전처리 ---
            transforms_list.append(v2.Resize(img_size_tuple, interpolation=InterpolationMode.BICUBIC, antialias=True))
            transforms_list.extend([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=config.MEAN, std=config.STD),
            ])
        
        return v2.Compose(transforms_list)