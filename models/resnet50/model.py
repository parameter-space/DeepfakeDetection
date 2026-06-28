# models/resnet50/model.py
import torch
import torch.nn as nn
import torch.optim as optim
from .config import ModelConfig
from typing import Type, List, Optional
# torchvision.models에서 '모델 구조'가 아닌 '가중치'만 불러오기 위해 import
from torchvision.models import resnet50, ResNet50_Weights 

# --- ResNet50의 핵심 빌딩 블록 정의 ---

class Bottleneck(nn.Module):
    """ResNet50에서 사용되는 Bottleneck 블록"""
    expansion: int = 4 # 출력 채널이 입력 채널의 4배

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        # 1x1 Conv
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3x3 Conv
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1x1 Conv (출력 채널 확장)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# --- ResNet 구조를 감싸는 우리 프레임워크의 Model 클래스 ---

class Model(nn.Module):
    """
    ResNet50 구조를 직접 정의하고, 
    engine.py와 호환되도록 래핑(wrapping)하는 메인 클래스
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.inplanes = 64
        
        # === ResNet 구조 정의 시작 ===
        # 1. Stem (초기 레이어)
        self.conv1 = nn.Conv2d(config.IN_CHANNELS, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 2. ResNet 블록 (ResNet50 = [3, 4, 6, 3] 구조)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        
        # 3. Head (분류기)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # (★중요) 이진 분류를 위해 마지막 'fc' 레이어 교체
        self.fc = nn.Linear(512 * Bottleneck.expansion, 1) # 1개 logit 출력
        
        # === ResNet 구조 정의 끝 ===

        # 4. 가중치 초기화 및 사전 학습된 가중치 로드
        self._initialize_weights()
        if config.LOAD_TORCHVISION_WEIGHTS:
            self._load_pretrained_weights()

        # 5. engine.py가 요구하는 criterion과 optimizer 정의
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=config.LR, 
            betas=config.BETAS
        )

        print(f"[Model Initialized] {config.MODEL_NAME} - Structure custom-built.")
        if config.LOAD_TORCHVISION_WEIGHTS:
            print("-> Successfully loaded pretrained weights from torchvision.")

    def _make_layer(
        self,
        block: Type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """ResNet의 'stage'를 생성하는 헬퍼 함수"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 해상도나 채널 수가 바뀔 때 identity 연결을 위한 1x1 Conv
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """기본 가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _load_pretrained_weights(self):
        """
        Torchvision에서 사전 학습된 '가중치'만 가져와서
        우리가 직접 정의한 '구조'에 매핑합니다.
        """
        print("Loading pretrained weights from torchvision...")
        # 1. 가중치만 불러오기
        weights = ResNet50_Weights.IMAGENET1K_V2
        pretrained_state_dict = weights.get_state_dict(progress=True)
        
        # 2. 현재 모델의 state_dict 가져오기
        model_state_dict = self.state_dict()
        
        # 3. 가중치 매핑
        # (Classifier(fc)를 제외한 모든 가중치를 복사)
        for k in pretrained_state_dict:
            if k in model_state_dict and model_state_dict[k].size() == pretrained_state_dict[k].size():
                model_state_dict[k] = pretrained_state_dict[k]
            elif k == "fc.weight" or k == "fc.bias":
                 print(f"Skipping mismatched layer: {k}") # fc 레이어는 크기가 달라 스킵
            else:
                 print(f"Warning: Layer {k} not found or mismatched.")

        # 4. 매핑된 가중치를 모델에 로드
        # strict=False는 'fc' 레이어가 달라도 오류를 내지 않음
        self.load_state_dict(model_state_dict, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ResNet의 forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x) # [B, 1] logit 출력
        
        return x

    def step(self, images, labels):
        self.optimizer.zero_grad()
        preds = self.forward(images)
        loss = self.criterion(preds, labels)
        loss.backward()
        self.optimizer.step()
        return preds, loss.item()

    @torch.no_grad()
    def predict(self, x, threshold: float = 0.5):
        probs = torch.sigmoid(self.forward(x))
        return (probs >= threshold).long()