# final_tester.py
import torch
import torch.nn as nn
from torchvision.transforms import v2
from PIL import Image
import numpy as np
import os
import glob
import tarfile
import shutil
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import dlib
import cv2
from pathlib import Path
from typing import Tuple

# dlib 얼굴 탐지기 초기화
try:
    face_detector = dlib.get_frontal_face_detector()
    print("[FinalTester] dlib 얼굴 탐지기 로드 성공.")
except Exception as e:
    print(f"[FinalTester] dlib 로드 실패: {e}.")
    face_detector = None

def get_boundingbox(face, width, height, scale=1.3):
    """
    얼굴 바운딩 박스로부터 크롭 영역을 계산합니다.
    
    Args:
        face: dlib 얼굴 감지 결과
        width: 이미지 너비
        height: 이미지 높이
        scale: 바운딩 박스 확대 비율
    
    Returns:
        (x, y, size) 튜플
    """
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb_w = min(width - x1, size_bb)
    size_bb_h = min(height - y1, size_bb)
    size_bb = min(size_bb_w, size_bb_h)
    return x1, y1, size_bb

def preprocess_image_dlib(
    img_np: np.ndarray,
    target_size_tuple: Tuple[int, int],
) -> torch.Tensor:
    """
    이미지를 전처리하여 모델 입력 텐서로 변환합니다.
    dlib을 사용하여 얼굴을 감지하고 크롭한 후 정규화합니다.
    
    Args:
        img_np: numpy 배열 이미지
        target_size_tuple: 목표 이미지 크기 (width, height)
    
    Returns:
        전처리된 텐서 또는 None (실패 시)
    """
    global face_detector
    if face_detector is None:
        raise RuntimeError("dlib face_detector가 로드되지 않았습니다.")
        
    try:
        # 이미지 타입 변환: uint8이 아니면 변환
        if img_np.dtype != np.uint8:
            if np.issubdtype(img_np.dtype, np.floating):
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255)
            
            elif np.issubdtype(img_np.dtype, np.integer) and img_np.max() > 255:
                img_np = (img_np >> 8)
            
            img_np = img_np.astype(np.uint8)

        # 채널 수 변환: 1채널 또는 4채널을 3채널로 변환
        if img_np.ndim == 2 or img_np.shape[2] == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        
        # BGR을 RGB로 변환 (dlib은 RGB를 기대)
        img_rgb_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        original_h, original_w, _ = img_rgb_np.shape

        # 큰 이미지는 리사이즈 (dlib 성능 향상)
        scale = 1.0
        if original_w > 640:
            scale = 640.0 / original_w
            resized_h = int(original_h * scale)
            img_resized_np = cv2.resize(img_rgb_np, (640, resized_h), interpolation=cv2.INTER_AREA)
        else:
            img_resized_np = img_rgb_np
            
        # dlib 요구사항: uint8 및 contiguous 배열
        if img_resized_np.dtype != np.uint8:
             img_resized_np = img_resized_np.astype(np.uint8)
             
        img_to_detect = np.ascontiguousarray(img_resized_np, dtype=np.uint8)
        
        # 얼굴 감지
        faces = face_detector(img_to_detect, 1)

        # 얼굴이 없으면 중앙 크롭, 있으면 얼굴 영역 크롭
        if not faces:
            size = min(original_w, original_h)
            x1 = (original_w - size) // 2
            y1 = (original_h - size) // 2
            cropped_np = img_rgb_np[y1:y1 + size, x1:x1 + size]
        else:
            # 가장 큰 얼굴 선택
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            scaled_face_rect = dlib.rectangle(
                left=int(face.left() / scale), top=int(face.top() / scale),
                right=int(face.right() / scale), bottom=int(face.bottom() / scale)
            )
            x, y, size = get_boundingbox(scaled_face_rect, original_w, original_h)
            cropped_np = img_rgb_np[y:y + size, x:x + size]
        
        # 목표 크기로 리사이즈 및 텐서 변환
        face_img_resized_np = cv2.resize(cropped_np, target_size_tuple, interpolation=cv2.INTER_AREA)
        
        face_tensor = torch.from_numpy(face_img_resized_np.transpose((2, 0, 1))).float()
        face_tensor = face_tensor / 255.0
        
        # ImageNet 통계치로 정규화
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        face_tensor = (face_tensor - mean) / std
        
        return face_tensor

    except Exception as e:
        print(f"dlib/cv2 처리 중 치명적 오류: {e}. None 반환.")
        return None

def predict_final_test(model: nn.Module, config, test_dir_path: str) -> Tuple[float, float, dict]:
    """
    최종 테스트 데이터에 대해 모델을 평가합니다.
    
    Args:
        model: 평가할 모델
        config: 설정 객체
        test_dir_path: 테스트 데이터 디렉토리 경로
    
    Returns:
        (F1 스코어, 정확도, confusion matrix) 튜플
    """
    DEVICE = config.DEVICE
    IMG_SIZE = config.IMG_SIZE
    y_true, y_pred = [], []
    
    # 이미지 파일 수집
    image_files = glob.glob(os.path.join(test_dir_path, '**', '*.jpg'), recursive=True)
    image_files.extend(glob.glob(os.path.join(test_dir_path, '**', '*.png'), recursive=True))
    
    print(f"[FinalTester] 이미지 {len(image_files)}개 추론 중...")
    for img_path in tqdm(image_files, desc="Final Test (Images)"):
        try:
            label = 1
            img_np = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) 
            if img_np is None: 
                print(f"Warning: cv2.imread 실패 (파일 손상?): {img_path}")
                continue
            
            input_tensor = preprocess_image_dlib(img_np, (IMG_SIZE, IMG_SIZE))
            if input_tensor is None: 
                print(f"Warning: dlib/preprocess 실패: {img_path}")
                continue
            
            input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = model(input_tensor)
                prob = torch.sigmoid(logits).item()
                
            pred = 1 if prob > 0.5 else 0
            y_true.append(label)
            y_pred.append(pred)
        except Exception as e:
            print(f"이미지 처리 오류 {img_path}: {e}")
            continue

    # 비디오 파일 처리
    video_files = glob.glob(os.path.join(test_dir_path, '**', '*.mp4'), recursive=True)
    print(f"[FinalTester] 비디오 {len(video_files)}개 추론 중...")
    
    for vid_path in tqdm(video_files, desc="Final Test (Videos)"):
        try:
            label = 1
            cap = cv2.VideoCapture(vid_path)
            frame_preds = []
            frame_idx = 0
            
            # 비디오에서 프레임 추출 및 추론
            while cap.isOpened():
                ret, frame_np = cap.read()
                if not ret: break
                
                # 10프레임마다 추론 수행
                if frame_idx % 10 == 0:
                    input_tensor = preprocess_image_dlib(frame_np, (IMG_SIZE, IMG_SIZE))
                    if input_tensor is None: continue
                    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        logits = model(input_tensor)
                        prob = torch.sigmoid(logits).item()
                    frame_preds.append(prob)
                frame_idx += 1
            cap.release()
            
            if frame_preds:
                video_prob = np.mean(frame_preds)
                pred = 1 if video_prob > 0.5 else 0
                y_true.append(label)
                y_pred.append(pred)
        except Exception as e:
            print(f"비디오 처리 오류 {vid_path}: {e}")
            continue

    # 최종 메트릭 계산
    if not y_true:
        print("[FinalTester] 경고: 유효한 테스트 샘플이 0개입니다.")
        return 0.0, 0.0, None
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return f1, acc, cm

def run_final_test(model, config, local_data_path):
    """
    최종 테스트를 실행합니다. tar 파일을 압축 해제하고 모델을 평가합니다.
    
    Args:
        model: 평가할 모델
        config: 설정 객체
        local_data_path: 로컬 데이터 경로
    
    Returns:
        F1 스코어
    """
    TAR_PATH = "/data/leetj3610/repo/deepfake-detection/data/final_test/sample_data.tar"
    UNTAR_PATH = Path(local_data_path) / "final_test_unpacked"

    if not os.path.exists(TAR_PATH):
        print(f"[FinalTester] 오류: {TAR_PATH}에 sample_data.tar 파일이 없습니다.")
        return 0.0 

    # tar 파일 압축 해제
    print(f"[FinalTester] '{TAR_PATH}' 파일의 압축을 해제합니다...")
    print(f"            -> 대상: {UNTAR_PATH}")
    if UNTAR_PATH.exists(): shutil.rmtree(UNTAR_PATH)
    UNTAR_PATH.mkdir(parents=True)
    try:
        with tarfile.open(TAR_PATH, "r") as tar:
            tar.extractall(path=UNTAR_PATH)
        print("[FinalTester] 압축 해제 완료.")
    except Exception as e:
        print(f"[FinalTester] .tar 파일 압축 해제 실패: {e}")
        return 0.0

    # 모델 평가
    model.eval()
    f1, acc, cm = predict_final_test(model=model, config=config, test_dir_path=str(UNTAR_PATH))

    print("\n" + "="*40)
    print(f"       [ {config.MODEL_NAME} - 최종 테스트 결과 ]")
    print("="*40)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print("-" * 20)
    print(f"Confusion Matrix (행: 실제, 열: 예측):")
    print(f"       [REAL (예측)] [FAKE (예측)]")
    tn = cm[0][0] if cm is not None and cm.shape[0] > 0 and cm.shape[1] > 0 else 0
    fp = cm[0][1] if cm is not None and cm.shape[0] > 0 and cm.shape[1] > 1 else 0
    fn = cm[1][0] if cm is not None and cm.shape[0] > 1 and cm.shape[1] > 0 else 0
    tp = cm[1][1] if cm is not None and cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    print(f"[REAL] [[TN: {tn:<6}]   [FP: {fp:<6}]]")
    print(f"[FAKE] [[FN: {fn:<6}]   [TP: {tp:<6}]]")
    print("="*40)

    shutil.rmtree(UNTAR_PATH)
    print(f"[FinalTester] 임시 폴더({UNTAR_PATH}) 정리 완료.")
    return f1