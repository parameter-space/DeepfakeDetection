# config_global.py
from dataclasses import dataclass, field
from typing import List, Tuple
import torch


@dataclass
class GlobalConfig:
    """
    전역 설정 클래스
    모든 모델에서 공통으로 사용하는 기본 설정값들을 정의합니다.
    """
    
    # 시스템 설정
    DEVICE: torch.device = field(default_factory=lambda: 
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # 학습 하이퍼파라미터
    NUM_EPOCHS: int = 80
    BATCH_SIZE: int = 32
    ACCUMULATION_STEPS: int = 2  # gradient accumulation 스텝 수
    
    # 데이터로더 설정
    SHUFFLE_SIZE: int = 1000  # 데이터셋 내부 셔플 크기
    
    # 데이터 청크 크기 제한 (GB)
    MAX_CHUNK_SIZE_GB: float = 35.0

    USE_NAS_DIRECTLY: bool = False  # NAS에서 직접 데이터 로드 여부
    NUM_WORKERS: int = 4  # 데이터로더 워커 수
    
    # 평가/로깅 설정
    LOG_INTERVAL: int = 500  # 로그 출력 간격 (스텝 수)
    USE_EARLY_STOPPING: bool = True  # 조기 종료 사용 여부
    EARLY_STOPPING_PATIENCE: int = 7  # 조기 종료 patience 

    # 학습 데이터셋 경로 및 레이블 (경로, 레이블) 튜플 리스트
    # 레이블: 1=Fake, 0=Real
    TRAIN_SHARDS: List[Tuple[str, int]] = field(default_factory=lambda: [
        # Fake 데이터셋 (레이블=1)
        # ("./data/diffusion_face/ADM-{000..004}.tar", 1),
        # ("./data/diffusion_face/DDIM-{000..004}.tar", 1),
        # ("./data/diffusion_face/DDPM-{000..004}.tar", 1),
        # ("./data/diffusion_face/DiffSwap-{000..005}.tar", 1), # 0-5 (6개)
        # ("./data/diffusion_face/Inpaint-{000..004}.tar", 1),
        # ("./data/diffusion_face/LDM-{000..004}.tar", 1),
        # ("./data/diffusion_face/PNDM-{000..004}.tar", 1),
        # ("./data/diffusion_face/SDv15_DS0.3-{000..004}.tar", 1),
        # ("./data/diffusion_face/SDv15_DS0.5-{000..004}.tar", 1),
        # ("./data/diffusion_face/SDv15_DS0.7-{000..004}.tar", 1),
        # ("./data/diffusion_face/SDv21_DS0.3-{000..004}.tar", 1),
        # ("./data/diffusion_face/SDv21_DS0.5-{000..004}.tar", 1),
        # ("./data/diffusion_face/SDv21_DS0.7-{000..004}.tar", 1),
        # ("./data/diffusion_face/stable_diffusion_v_1_5_text2img_p3g7-{000..004}.tar", 1),
        # ("./data/diffusion_face/stable_diffusion_v_1_5_text2img_p4g5-{000..004}.tar", 1),
        # ("./data/diffusion_face/stable_diffusion_v_1_5_text2img_p5g3-{000..004}.tar", 1),
        # ("./data/diffusion_face/stable_diffusion_v_2_1_text2img_p0g5-{000..004}.tar", 1),
        # ("./data/diffusion_face/stable_diffusion_v_2_1_text2img_p1g7-{000..004}.tar", 1),
        # ("./data/diffusion_face/stable_diffusion_v_2_1_text2img_p2g3-{000..004}.tar", 1),
        
        # 2. AIFaceDataset3000 (샘플링)
        # ("./data/AIFaceDataset3000/AIFaceDataset3000-{000000..000026}.tar", 1), # 0-26 (27개)
        
        # 3. SFHQ (샘플링)
        ("./data/SFHQ/SFHQ_1A/SFHQ_1A-{000000..000223}.tar", 1),
        ("./data/SFHQ/SFHQ_1B/SFHQ_1B-{000000..000223}.tar", 1),
        ("./data/SFHQ/SFHQ_1C/SFHQ_1C-{000000..000223}.tar", 1),
        ("./data/SFHQ/SFHQ_1D/SFHQ_1D-{000000..000223}.tar", 1),
        ("./data/SFHQ/SFHQ_2A/SFHQ_2A-{000000..000227}.tar", 1),
        ("./data/SFHQ/SFHQ_2B/SFHQ_2B-{000000..000227}.tar", 1),
        ("./data/SFHQ/SFHQ_2C/SFHQ_2C-{000000..000227}.tar", 1),
        ("./data/SFHQ/SFHQ_2D/SFHQ_2D-{000000..000227}.tar", 1),
        ("./data/SFHQ/SFHQ_3A/SFHQ_3A-{000000..000294}.tar", 1),
        ("./data/SFHQ/SFHQ_3B/SFHQ_3B-{000000..000294}.tar", 1),
        ("./data/SFHQ/SFHQ_3C/SFHQ_3C-{000000..000294}.tar", 1),
        ("./data/SFHQ/SFHQ_3D/SFHQ_3D-{000000..000294}.tar", 1),
        ("./data/SFHQ/SFHQ_4A/SFHQ_4A-{000000..000313}.tar", 1),
        ("./data/SFHQ/SFHQ_4B/SFHQ_4B-{000000..000313}.tar", 1),
        ("./data/SFHQ/SFHQ_4C/SFHQ_4C-{000000..000313}.tar", 1),
        ("./data/SFHQ/SFHQ_4D/SFHQ_4D-{000000..000313}.tar", 1),
        
        
        # 4. faceswap (train 스플릿 사용)
        # ("./data/faceswap/train/fake_shards/FaceSwap_Fake_Train-{000000..000072}.tar", 1),
        
        ("./data/HI-SFHQ/HI-SFHQ_1/HI-SFHQ_1-{000000..000305}.tar", 1),
        ("./data/HI-SFHQ/HI-SFHQ_2/HI-SFHQ_2-{000000..000305}.tar", 1),
        ("./data/HI-SFHQ/HI-SFHQ_3/HI-SFHQ_3-{000000..000305}.tar", 1),
        ("./data/HI-SFHQ/HI-SFHQ_4/HI-SFHQ_4-{000000..000305}.tar", 1),

        ("./data/SFHQ-T2I/DALLE3/shard-000000.tar", 1),
        ("./data/SFHQ-T2I/DALLE3/shard-000000.tar", 1),
        ("./data/SFHQ-T2I/FLUX1_dev/shard-{000000..000006}.tar", 1),
        ("./data/SFHQ-T2I/FLUX1_pro/shard-{000000..000002}.tar", 1),
        ("./data/SFHQ-T2I/FLUX1_schnell/shard-{000000..000057}.tar", 1),
        ("./data/SFHQ-T2I/SDXL/shard-{000000..000052}.tar", 1),
        
        # Real 데이터셋 (레이블=0)
        ("./data/FFHQ/FFHQ_1/FFHQ_1-{000000..000048}.tar", 0),
        ("./data/FFHQ/FFHQ_2/FFHQ_2-{000000..000048}.tar", 0),
        ("./data/FFHQ/FFHQ_3/FFHQ_3-{000000..000048}.tar", 0),
        ("./data/FFHQ/FFHQ_4/FFHQ_4-{000000..000048}.tar", 0),
        ("./data/FFHQ/FFHQ_5/FFHQ_5-{000000..000048}.tar", 0),
        ("./data/FFHQ/FFHQ_6/FFHQ_6-{000000..000048}.tar", 0),
        ("./data/FFHQ/FFHQ_7/FFHQ_7-{000000..000048}.tar", 0),
        ("./data/FFHQ/FFHQ_8/FFHQ_8-{000000..000048}.tar", 0),
        ("./data/FFHQ/FFHQ_9/FFHQ_9-{000000..000048}.tar", 0),
        ("./data/FFHQ/FFHQ_10/FFHQ_10-{000000..000048}.tar", 0),
        ("./data/FFHQ/FFHQ_11/FFHQ_11-{000000..000048}.tar", 0),
        ("./data/FFHQ/FFHQ_12/FFHQ_12-{000000..000048}.tar", 0),
        ("./data/FFHQ/FFHQ_13/FFHQ_13-{000000..000048}.tar", 0),
        ("./data/FFHQ/FFHQ_14/FFHQ_14-{000000..000048}.tar", 0),
        
        # 3. faceswap (train 스플릿 사용)
        # ("./data/faceswap/train/real_shards/FaceSwap_Real_Train-{000000..000072}.tar", 0),
        
        # ("./data/CelebA/CelebA-{000000..001999}.tar", 0),
        # ("./data/VGGFACE/VGGFACE/img_files-{000000..001777}.tar", 0),
        ("./data/celeba_hq/train/shard-{000000..000027}.tar", 0),

        ("./data/generated/generated/generated-{000000..000033}.tar", 1),

        # # ==========================================
        # # 2. OLD FAKE (조연) - 대폭 축소 (기존 지식 유지용)
        # # ==========================================
        # ("./data/SFHQ/SFHQ_1A/SFHQ_1A-{000000..000001}.tar", 1),
        # ("./data/SFHQ/SFHQ_1B/SFHQ_1B-{000000..000001}.tar", 1),
        # ("./data/SFHQ/SFHQ_1C/SFHQ_1C-{000000..000001}.tar", 1),
        # ("./data/SFHQ/SFHQ_1D/SFHQ_1D-{000000..000001}.tar", 1),
        # ("./data/SFHQ/SFHQ_2A/SFHQ_2A-{000000..000001}.tar", 1),
        # ("./data/SFHQ/SFHQ_2B/SFHQ_2B-{000000..000001}.tar", 1),
        # ("./data/SFHQ/SFHQ_2C/SFHQ_2C-{000000..000001}.tar", 1),
        # ("./data/SFHQ/SFHQ_2D/SFHQ_2D-{000000..000001}.tar", 1),
        # ("./data/SFHQ/SFHQ_3A/SFHQ_3A-{000000..000001}.tar", 1),
        # ("./data/SFHQ/SFHQ_3B/SFHQ_3B-{000000..000001}.tar", 1),
        # ("./data/SFHQ/SFHQ_3C/SFHQ_3C-{000000..000001}.tar", 1),
        # ("./data/SFHQ/SFHQ_3D/SFHQ_3D-{000000..000001}.tar", 1),
        # ("./data/SFHQ/SFHQ_4A/SFHQ_4A-{000000..000001}.tar", 1),
        # ("./data/SFHQ/SFHQ_4B/SFHQ_4B-{000000..000001}.tar", 1),
        # ("./data/SFHQ/SFHQ_4C/SFHQ_4C-{000000..000001}.tar", 1),
        # ("./data/SFHQ/SFHQ_4D/SFHQ_4D-{000000..000001}.tar", 1),

        # ("./data/AIFaceDataset3000/AIFaceDataset3000-{000000..000001}.tar", 1),
        
        # ("./data/HI-SFHQ/HI-SFHQ_1/HI-SFHQ_1-{000000..000003}.tar", 1),
        # ("./data/HI-SFHQ/HI-SFHQ_2/HI-SFHQ_2-{000000..000003}.tar", 1),
        # ("./data/HI-SFHQ/HI-SFHQ_3/HI-SFHQ_3-{000000..000003}.tar", 1),
        # ("./data/HI-SFHQ/HI-SFHQ_4/HI-SFHQ_4-{000000..000003}.tar", 1),
        
        # ("./data/SFHQ-T2I/DALLE3/shard-000000.tar", 1),
        # ("./data/SFHQ-T2I/FLUX1_dev/shard-{000000..000002}.tar", 1),
        # ("./data/SFHQ-T2I/FLUX1_pro/shard-{000000..000005}.tar", 1),
        # ("./data/SFHQ-T2I/FLUX1_schnell/shard-{000000..000002}.tar", 1),
        # ("./data/SFHQ-T2I/SDXL/shard-{000000..000005}.tar", 1),

        # # ==========================================
        # # 3. REAL (균형추) - Fake 합계(약 80개)와 비슷하게 맞춤
        # # ==========================================
        # # FFHQ 1번 폴더 전체 (49개)
        # ("./data/FFHQ/FFHQ_1/FFHQ_1-{000000..000004}.tar", 0), # 0-48 (45개)
        # ("./data/FFHQ/FFHQ_2/FFHQ_2-{000000..000004}.tar", 0),
        # ("./data/FFHQ/FFHQ_3/FFHQ_3-{000000..000004}.tar", 0),
        # ("./data/FFHQ/FFHQ_4/FFHQ_4-{000000..000004}.tar", 0),
        # ("./data/FFHQ/FFHQ_5/FFHQ_5-{000000..000004}.tar", 0),
        # ("./data/FFHQ/FFHQ_6/FFHQ_6-{000000..000004}.tar", 0),
        # ("./data/FFHQ/FFHQ_7/FFHQ_7-{000000..000004}.tar", 0),
        # ("./data/FFHQ/FFHQ_8/FFHQ_8-{000000..000004}.tar", 0),
        # ("./data/FFHQ/FFHQ_9/FFHQ_9-{000000..000004}.tar", 0),
        # ("./data/FFHQ/FFHQ_10/FFHQ_10-{000000..000004}.tar", 0),
        # ("./data/FFHQ/FFHQ_11/FFHQ_11-{000000..000004}.tar", 0),
        # ("./data/FFHQ/FFHQ_12/FFHQ_12-{000000..000004}.tar", 0),
        # ("./data/FFHQ/FFHQ_13/FFHQ_13-{000000..000004}.tar", 0),
        # ("./data/FFHQ/FFHQ_14/FFHQ_14-{000000..000004}.tar", 0),
        
        # # CelebA-HQ Train 전체 (28개) - 고화질 Real 유지
        # ("./data/celeba_hq/train/shard-{000000..000020}.tar", 0),
        

    ])

    # 테스트/검증 데이터셋 경로 및 레이블
    TEST_SHARDS: List[Tuple[str, int]] = field(default_factory=lambda: [
        # Fake 데이터셋 (레이블=1)
        # ("./data/diffusion_face/ADM-005.tar", 1),
        # ("./data/diffusion_face/DDIM-005.tar", 1),
        # ("./data/diffusion_face/DDPM-005.tar", 1),
        # ("./data/diffusion_face/DiffSwap-006.tar", 1), # 마지막 파일
        # ("./data/diffusion_face/Inpaint-005.tar", 1),
        # ("./data/diffusion_face/LDM-005.tar", 1),
        # ("./data/diffusion_face/PNDM-005.tar", 1),
        # ("./data/diffusion_face/SDv15_DS0.3-005.tar", 1),
        # ("./data/diffusion_face/SDv15_DS0.5-005.tar", 1),
        # ("./data/diffusion_face/SDv15_DS0.7-005.tar", 1),
        # ("./data/diffusion_face/SDv21_DS0.3-005.tar", 1),
        # ("./data/diffusion_face/SDv21_DS0.5-005.tar", 1),
        # ("./data/diffusion_face/SDv21_DS0.7-005.tar", 1),
        # ("./data/diffusion_face/stable_diffusion_v_1_5_text2img_p3g7-005.tar", 1),
        # ("./data/diffusion_face/stable_diffusion_v_1_5_text2img_p4g5-005.tar", 1),
        # ("./data/diffusion_face/stable_diffusion_v_1_5_text2img_p5g3-005.tar", 1),
        # ("./data/diffusion_face/stable_diffusion_v_2_1_text2img_p0g5-005.tar", 1),
        # ("./data/diffusion_face/stable_diffusion_v_2_1_text2img_p1g7-005.tar", 1),
        # ("./data/diffusion_face/stable_diffusion_v_2_1_text2img_p2g3-005.tar", 1),
        
        # 2. AIFaceDataset3000 (샘플링)
        ("./data/AIFaceDataset3000/AIFaceDataset3000-{000027..000029}.tar", 1), # 27-29 (3개)
        
        # 3. SFHQ (샘플링)
        ("./data/SFHQ/SFHQ_1A/SFHQ_1A-000224.tar", 1), # 200-224 (25개)
        ("./data/SFHQ/SFHQ_1B/SFHQ_1B-000224.tar", 1),
        ("./data/SFHQ/SFHQ_1C/SFHQ_1C-000224.tar", 1),
        ("./data/SFHQ/SFHQ_1D/SFHQ_1D-000224.tar", 1),
        ("./data/SFHQ/SFHQ_2A/SFHQ_2A-000228.tar", 1), # 200-228 (29개)
        ("./data/SFHQ/SFHQ_2B/SFHQ_2B-000228.tar", 1),
        ("./data/SFHQ/SFHQ_2C/SFHQ_2C-000228.tar", 1),
        ("./data/SFHQ/SFHQ_2D/SFHQ_2D-000228.tar", 1),
        ("./data/SFHQ/SFHQ_3A/SFHQ_3A-000295.tar", 1), # 270-295 (26개)
        ("./data/SFHQ/SFHQ_3B/SFHQ_3B-000295.tar", 1),
        ("./data/SFHQ/SFHQ_3C/SFHQ_3C-000295.tar", 1),
        ("./data/SFHQ/SFHQ_3D/SFHQ_3D-000295.tar", 1),
        ("./data/SFHQ/SFHQ_4A/SFHQ_4A-000314.tar", 1), # 280-314 (35개)
        ("./data/SFHQ/SFHQ_4B/SFHQ_4B-000314.tar", 1),
        ("./data/SFHQ/SFHQ_4C/SFHQ_4C-000314.tar", 1),
        ("./data/SFHQ/SFHQ_4D/SFHQ_4D-000314.tar", 1),
        
        # 4. faceswap (test/val 스플릿 사용)
        # ("./data/faceswap/test/fake_shards/FaceSwap_Fake_Test-{000000..000009}.tar", 1),
        # ("./data/faceswap/val/fake_shards/FaceSwap_Fake_Val-{000000..000009}.tar", 1),

        ("./data/HI-SFHQ/HI-SFHQ_1/HI-SFHQ_1-000306.tar", 1),
        ("./data/HI-SFHQ/HI-SFHQ_2/HI-SFHQ_2-000306.tar", 1),
        ("./data/HI-SFHQ/HI-SFHQ_3/HI-SFHQ_3-000306.tar", 1),
        ("./data/HI-SFHQ/HI-SFHQ_4/HI-SFHQ_4-000306.tar", 1),

        ("./data/SFHQ-T2I/DALLE3/shard-000001.tar", 1),
        ("./data/SFHQ-T2I/DALLE3/shard-000001.tar", 1),
        ("./data/SFHQ-T2I/FLUX1_dev/shard-000007.tar", 1),
        ("./data/SFHQ-T2I/FLUX1_pro/shard-000003.tar", 1),
        ("./data/SFHQ-T2I/FLUX1_schnell/shard-000058.tar", 1),
        ("./data/SFHQ-T2I/SDXL/shard-000053.tar", 1),


        ("./data/generated/generated/generated-000034.tar", 1),

        # Real 데이터셋 (레이블=0)
        ("./data/FFHQ/FFHQ_1/FFHQ_1-000049.tar", 0),
        ("./data/FFHQ/FFHQ_2/FFHQ_2-000049.tar", 0),
        ("./data/FFHQ/FFHQ_3/FFHQ_3-000049.tar", 0),
        ("./data/FFHQ/FFHQ_4/FFHQ_4-000049.tar", 0),
        ("./data/FFHQ/FFHQ_5/FFHQ_5-000049.tar", 0),
        ("./data/FFHQ/FFHQ_6/FFHQ_6-000049.tar", 0),
        ("./data/FFHQ/FFHQ_7/FFHQ_7-000049.tar", 0),
        ("./data/FFHQ/FFHQ_8/FFHQ_8-000049.tar", 0),
        ("./data/FFHQ/FFHQ_9/FFHQ_9-000049.tar", 0),
        ("./data/FFHQ/FFHQ_10/FFHQ_10-000049.tar", 0),
        ("./data/FFHQ/FFHQ_11/FFHQ_11-000049.tar", 0),
        ("./data/FFHQ/FFHQ_12/FFHQ_12-000049.tar", 0),
        ("./data/FFHQ/FFHQ_13/FFHQ_13-000049.tar", 0),
        ("./data/FFHQ/FFHQ_14/FFHQ_14-000049.tar", 0),
        
        # 3. faceswap (test/val 스플릿 사용)
        # ("./data/faceswap/test/real_shards/FaceSwap_Real_Test-{000000..000009}.tar", 0),
        # ("./data/faceswap/val/real_shards/FaceSwap_Real_Val-{000000..000009}.tar", 0),

        # ("./data/CelebA/CelebA-{002000..002025}.tar", 0),
        # ("./data/VGGFACE/VGGFACE/img_files-{001778..001976}.tar", 0),
        ("./data/celeba_hq/val/shard-{000000..000001}.tar", 0),
    ])

    # 모델 기본값 (모델별 config에서 오버라이드 가능)
    IMG_SIZE: int = 256  # 입력 이미지 크기
    MEAN: List[float] = None  # 정규화 평균값
    STD: List[float] = None  # 정규화 표준편차
    MODEL_NAME: str = "BaseModel"  # 모델 이름
    LR: float = 0.001  # 학습률
    BETAS: Tuple[float, float] = (0.9, 0.999)  # Adam 옵티마이저 beta 파라미터
    BEST_WEIGHT_LOAD = False  # 최고 성능 가중치 로드 여부