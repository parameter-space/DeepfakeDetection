# engine.py
import torch
import time
from torch.utils.data import DataLoader
from config_global import GlobalConfig
from typing import Tuple, Generator
from torch.amp import autocast, GradScaler 

def _calculate_metrics(tp, fp, fn, tn):
    """
    Confusion matrix 값으로부터 정확도와 F1 스코어를 계산합니다.
    
    Args:
        tp: True Positive
        fp: False Positive
        fn: False Negative
        tn: True Negative
    
    Returns:
        (accuracy, f1_score) 튜플
    """
    # F1 스코어 계산
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # 정확도 계산
    total_samples = tp + fp + fn + tn
    accuracy = (tp + tn) / (total_samples + 1e-6)
    
    return accuracy, f1

def train_one_epoch(model: torch.nn.Module, 
                      dataloader: DataLoader, 
                      config: GlobalConfig, 
                      epoch: int) -> Tuple[float, float, float]:
    """
    한 에포크 동안 모델을 학습합니다.
    AMP(Automatic Mixed Precision)와 Gradient Accumulation을 사용합니다.
    
    Args:
        model: 학습할 모델
        dataloader: 학습 데이터로더
        config: 전역 설정
        epoch: 현재 에포크 번호
    
    Returns:
        (평균 loss, 정확도, F1 스코어) 튜플
    """
    model.train()
    
    running_loss = 0.0
    
    # F1 스코어 계산을 위한 confusion matrix 카운터
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    start_time = time.time()
    device = config.DEVICE

    # Gradient accumulation 설정
    acc_steps = getattr(config, "ACCUMULATION_STEPS", 1)
    scaler = GradScaler()
    model.optimizer.zero_grad()
    
    i = 0 
    images: torch.Tensor
    labels: torch.Tensor
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels_float = labels.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)
        labels_long = labels.to(device, dtype=torch.long, non_blocking=True)

        # Forward pass with mixed precision
        with autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model.step(images, labels_float)
            loss = loss / acc_steps
        
        loss_value = loss.item() * acc_steps
        scaler.scale(loss).backward()

        # Gradient accumulation: acc_steps마다 업데이트
        if (i + 1) % acc_steps == 0:
            scaler.unscale_(model.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(model.optimizer)
            scaler.update()
            model.optimizer.zero_grad()

        # 예측 및 메트릭 계산
        preds = (torch.sigmoid(logits.view(-1)) > 0.5).long()
        labels_view = labels_long.view(-1)
        
        total_tp += ((preds == 1) & (labels_view == 1)).sum().item()
        total_fp += ((preds == 1) & (labels_view == 0)).sum().item()
        total_fn += ((preds == 0) & (labels_view == 1)).sum().item()
        total_tn += ((preds == 0) & (labels_view == 0)).sum().item()
        
        running_loss += loss_value * images.size(0)

        # 주기적 로그 출력
        if (i + 1) % config.LOG_INTERVAL == 0:
            total_samples_so_far = total_tp + total_fp + total_fn + total_tn
            
            acc, f1 = _calculate_metrics(total_tp, total_fp, total_fn, total_tn)
            avg_loss = running_loss / total_samples_so_far
            elapsed = time.time() - start_time
            
            print(f"[epoch {epoch}/{config.NUM_EPOCHS} | step {i + 1}] "
                  f"loss={avg_loss:.4f} acc={acc:.4f} f1={f1:.4f} elapsed={elapsed:.1f}s")
            
    # 에포크 끝에 남은 gradient 처리
    if (i + 1) % acc_steps != 0:
        print(f"Applying remaining gradients at end of epoch (step {i+1})...")
        scaler.step(model.optimizer)
        scaler.update()
        model.optimizer.zero_grad()

    # 최종 메트릭 계산
    total_samples = total_tp + total_fp + total_fn + total_tn
    
    if total_samples == 0:
        print(f"Epoch {epoch} done: No samples in train loader.")
        return 0.0, 0.0, 0.0
        
    avg_loss = running_loss / total_samples
    acc, f1 = _calculate_metrics(total_tp, total_fp, total_fn, total_tn)
    
    print(f"Epoch {epoch} done: Train loss={avg_loss:.4f} Train acc={acc:.4f} Train F1={f1:.4f}")
    
    return avg_loss, acc, f1

def evaluate(model: torch.nn.Module, 
             dataloader_generator: Generator[DataLoader, None, None],
             config: GlobalConfig) -> Tuple[float, float, float]: 
    """
    모델을 평가합니다. DataLoader 생성기를 받아 청크별로 순회하며 평가를 수행합니다.
    
    Args:
        model: 평가할 모델
        dataloader_generator: 테스트 데이터로더 생성기
        config: 전역 설정
    
    Returns:
        (평균 loss, 정확도, F1 스코어) 튜플
    """
    model.eval()
    
    running_loss = 0.0
    
    # F1 스코어 계산을 위한 confusion matrix 카운터
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    device = config.DEVICE

    with torch.no_grad():
        # 각 청크별로 평가 수행
        for dataloader_chunk in dataloader_generator:
            images: torch.Tensor
            labels: torch.Tensor
            for images, labels in dataloader_chunk:
                images = images.to(device, non_blocking=True)
                labels_float = labels.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)
                labels_long = labels.to(device, dtype=torch.long, non_blocking=True)

                # Forward pass with mixed precision
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(images)
                    loss = model.criterion(logits, labels_float)

                # 예측 및 메트릭 계산
                preds = (torch.sigmoid(logits.view(-1)) > 0.5).long()
                labels_view = labels_long.view(-1)

                total_tp += ((preds == 1) & (labels_view == 1)).sum().item()
                total_fp += ((preds == 1) & (labels_view == 0)).sum().item()
                total_fn += ((preds == 0) & (labels_view == 1)).sum().item()
                total_tn += ((preds == 0) & (labels_view == 0)).sum().item()

                running_loss += loss.item() * images.size(0)

    # 최종 메트릭 계산
    total_samples = total_tp + total_fp + total_fn + total_tn
    
    if total_samples == 0:
        print(f"--- Evaluation ---")
        print(f"Warning: No samples found in test/validation set.")
        print(f"------------------")
        return 0.0, 0.0, 0.0

    avg_loss = running_loss / total_samples
    acc, f1 = _calculate_metrics(total_tp, total_fp, total_fn, total_tn)

    print(f"--- Evaluation ---")
    print(f"Test/Validation loss: {avg_loss:.4f}, Test/Validation acc: {acc:.4f}, Test/Validation F1: {f1:.4f}")
    print(f"------------------")
    
    return avg_loss, acc, f1