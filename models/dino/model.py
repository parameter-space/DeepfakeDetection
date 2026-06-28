# models/dinov2_vitb14/model.py
import torch
import torch.nn as nn
import torch.optim as optim
from .config import ModelConfig
from typing import Tuple
from torchvision.transforms import v2
import timm 
from itertools import chain
from torch.optim.lr_scheduler import CosineAnnealingLR
import io
import random
from PIL import Image

# --- CustomRandomJPEG (이전과 동일) ---
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
        
        # [수정] Mixup은 DINOv2가 충분히 강하므로 일단 False로 시작
        self.use_mixup = False
        self.mixup_alpha = 0.3

        # 1. [핵심] DINOv2 백본 로드
        # num_classes=0 : timm에서 분류기 헤드를 제거하고 특징만 반환
        # global_pool='token' : ViT의 [CLS] 토큰 특징을 사용
        self.backbone = timm.create_model(
            config.TIMM_MODEL_NAME, 
            pretrained=config.LOAD_TIMM_PRETRAINED,
            num_classes=0,       # 분류기 헤드 제거
            global_pool='token'  # [CLS] 토큰 특징만 출력
        )
        
        # DINOv2 ViT-Base의 특징 벡터 크기는 768
        num_features = self.backbone.num_features
        
        # 2. [신규] 단순한 분류기 헤드
        # (이전의 복잡한 ViT-Hybrid-Head가 필요 없어짐)
        self.head_dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(num_features, 1)

        # 3. [유지] 옵티마이저 + 스케줄러 (차등 LR 적용)
        self.criterion = nn.BCEWithLogitsLoss()
        
        backbone_params = self.backbone.parameters()
        head_params = chain(
            self.head_dropout.parameters(),
            self.classifier.parameters(),
        )
        
        param_groups = [
            {'params': backbone_params, 'lr': config.LR}, 
            {'params': head_params, 'lr': config.LR * 3} # 새 헤드는 3배 빠른 학습
        ]

        self.optimizer = optim.Adam(
            param_groups,
            betas=config.BETAS,
            weight_decay=1e-4
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=config.LR * 0.01
        )

        print(f"[Model Initialized] {config.MODEL_NAME}")
        if config.LOAD_TIMM_PRETRAINED:
            print(f"-> Pretrained DINOv2 weights loaded via timm.")
        print(f"-> Optimizer Setup: Backbone LR={config.LR}, New Head LR={config.LR * 3}")
        print(f"-> [AUGMENTATION] Mixup={self.use_mixup}, CustomJPEG/Blur (in loader).")

    # ( ... mixup_data, mixup_criterion 함수는 이전과 동일 ...)
    def mixup_data(self, x, y):
        # (이전 코드와 동일)
        import numpy as np 
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        # (이전 코드와 동일)
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        (수정) DINOv2 백본 + 단순 헤드
        """
        # x_features의 shape: (batch_size, 768)
        x_features = self.backbone(x) 
        
        x = self.head_dropout(x_features)
        x_out = self.classifier(x)
        return x_out

    def step(self, images, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (수정) Mixup을 적용하는 step 함수
        """
        if self.training and self.use_mixup:
            images, labels_a, labels_b, lam = self.mixup_data(images, labels)
            
            logits = self.forward(images)
            loss = self.mixup_criterion(self.criterion, logits, labels_a, labels_b, lam)
        else:
            logits = self.forward(images)
            loss = self.criterion(logits, labels)
            
        return logits, loss

    @torch.no_grad()
    def predict(self, x, threshold: float = 0.5):
        # (이전 코드와 동일)
        probs = torch.sigmoid(self.forward(x))
        return (probs >= threshold).long()


    @staticmethod
    def get_transforms(config: ModelConfig, is_train: bool) -> v2.Compose:
        """
        [수정]
        - IMG_SIZE를 518로 변경
        - 안정성을 위해 Augmentation 강도 하향 (TrivialAugment 제거)
        """
        img_size = config.IMG_SIZE # 518
        mean = config.MEAN
        std = config.STD

        if is_train:
            return v2.Compose([
                v2.Resize(size=img_size, antialias=True),
                v2.RandomCrop(size=img_size),
                v2.RandomHorizontalFlip(p=0.5),
                
                # [수정] TrivialAugmentWide 제거 (DINOv2는 이미 강하므로)
                
                # [수정] 확률을 0.5 -> 0.2로 낮춤
                CustomRandomJPEG(quality=(50, 90), p=0.2),
                
                # [수정] 확률을 0.3 -> 0.1로 낮춤
                v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 3))], p=0.1),
                
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ])
        else:
            # (유지) 테스트 시에는 518로 Resize/CenterCrop
            return v2.Compose([
                v2.Resize(size=img_size, antialias=True),
                v2.CenterCrop(size=img_size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ])