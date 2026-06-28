# models/dual_stream_xception/model.py
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from .config import ModelConfig

# (1) transforms 임포트 (정적 메서드용)
from torchvision.transforms import v2, InterpolationMode

# --- (2) 주파수 변환 (FFT) 헬퍼 ---
def high_frequency_fft(images: torch.Tensor, radius_ratio: float = 0.6) -> torch.Tensor:
    """
    FFT를 수행하고, 저주파 영역(가운데)을 마스킹하여
    고주파 아티팩트만 남깁니다.
    """
    # images: [B, C, H, W]
    
    # FFT 수행 (rfft2는 실수 입력에 대한 2D FFT)
    fft_images = torch.fft.rfft2(images, dim=(-2, -1), norm='ortho')
    
    # 텐서의 중심(저주파)을 마스킹하기 위한 마스크 생성
    b, c, h, w = fft_images.shape
    center_h, center_w = h // 2, w // 2
    radius_h, radius_w = int(h * radius_ratio / 2), int(w * radius_ratio / 2)
    
    mask = torch.ones_like(fft_images, dtype=torch.bool)
    
    # 중심부(저주파)를 False로 마스킹
    mask[..., center_h - radius_h : center_h + radius_h, 
             center_w - radius_w : center_w + radius_w] = False
    
    # 마스킹 적용 (고주파만 남김)
    fft_images_hf = fft_images * mask
    
    # 역 FFT를 통해 다시 이미지 공간으로 변환
    hf_images = torch.fft.irfft2(fft_images_hf, s=images.shape[-2:], dim=(-2, -1), norm='ortho')
    
    return hf_images

# --- (3) Two-Stream 모델 정의 ---
class Model(nn.Module):
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # --- (A) 인코더 정의 ---
        
        # 스트림 1: 공간(Spatial) 인코더
        # XceptionNet의 특징 추출기(Head 제외)만 로드
        self.spatial_encoder = timm.create_model(
            'xception', 
            pretrained=True, 
            num_classes=0 # Head를 제외하고 특징맵만 반환
        )
        
        # 스트림 2: 주파수(Frequency) 인코더
        # (동일한 구조, 가중치 공유 가능하나 별도 학습이 유리)
        self.frequency_encoder = timm.create_model(
            'xception', 
            pretrained=True, 
            num_classes=0
        )
        
        # --- (B) "다른 처리" (교수님이 말한 강력한 Head) ---
        
        # XceptionNet의 특징 출력은 2048
        # 두 스트림을 합치므로 2048 * 2 = 4096
        num_combined_features = 2048 * 2
        
        self.classifier_head = nn.Sequential(
            nn.Linear(num_combined_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1) # 최종 1개 logit 출력
        )
        
        # --- (C) 엔진 호환부 ---
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=config.LR, 
            betas=config.BETAS
        )
        print(f"[Model Initialized] {config.MODEL_NAME} (Spatial + Frequency)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x는 [B, 3, 299, 299]
        
        # 1. 주파수 이미지 생성 (FFT 수행)
        x_freq = high_frequency_fft(x)
        
        # 2. 스트림 1: 공간 특징 추출
        feat_spatial = self.spatial_encoder(x) # [B, 2048]
        
        # 3. 스트림 2: 주파수 특징 추출
        feat_freq = self.frequency_encoder(x_freq) # [B, 2048]
        
        # 4. 특징 결합 (Concatenate)
        combined_features = torch.cat([feat_spatial, feat_freq], dim=1) # [B, 4096]
        
        # 5. 강력한 Head 통과
        logits = self.classifier_head(combined_features) # [B, 1]
        
        return logits

    def step(self, images, labels):
        self.optimizer.zero_grad()
        preds = self.forward(images) # Two-Stream forward 호출
        loss = self.criterion(preds, labels)
        loss.backward()
        self.optimizer.step()
        return preds, loss.item()

    @torch.no_grad()
    def predict(self, x, threshold: float = 0.5):
        probs = torch.sigmoid(self.forward(x))
        return (probs >= threshold).long()

    # (4) 전처리 메서드 (기존 XceptionNet과 동일)
    @staticmethod
    def get_transforms(config: ModelConfig, is_train: bool = True) -> v2.Compose:
        """
        XceptionNet을 위한 커스텀 Augmentation 파이프라인
        (RandomResizedCrop + JPEG)
        """
        transforms_list = []

        if is_train:
            transforms_list.extend([
                v2.RandomResizedCrop(
                    (config.IMG_SIZE, config.IMG_SIZE), 
                    scale=(0.85, 1.0),
                    ratio=(0.9, 1.1),
                    interpolation=InterpolationMode.BICUBIC
                ),
                v2.RandomApply([
                    v2.JPEG(quality=(40, 90))
                ], p=0.5),
                v2.RandomHorizontalFlip(p=0.5),
            ])
        else:
            transforms_list.append(
                 v2.Resize(
                    (config.IMG_SIZE, config.IMG_SIZE), 
                    interpolation=InterpolationMode.BICUBIC
                )
            )

        transforms_list.extend([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=config.MEAN, std=config.STD),
        ])
        
        return v2.Compose(transforms_list)