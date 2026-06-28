# main.py
import os
import importlib
import data_loader 
import engine
import copy
import subprocess
import shutil
from pathlib import Path
import time

import torch
import argparse
import final_tester

def _sync_cleanup_dir(cleanup_target_dir: str, required_gb: float):
    """
    지정된 폴더 내부를 청소하고 NFS가 공간을 확보할 때까지 대기합니다.
    
    Args:
        cleanup_target_dir: 청소할 디렉토리 경로
        required_gb: 필요한 공간 (GB)
    """
    local_cleanup_path = Path(cleanup_target_dir) 
    
    if not local_cleanup_path.exists():
        print(f"Cleanup not needed. {local_cleanup_path} does not exist.")
        return 

    print(f"Starting SYNC cleanup of contents in {local_cleanup_path}...")
    try:
        # 폴더 내용물 삭제
        for item in local_cleanup_path.glob('*'):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print("Folder contents deleted.")

        # NFS 공간 확보 대기 (최대 30초)
        required_bytes = required_gb * 1024 * 1024 * 1024
        for _ in range(30):
            df_output = subprocess.check_output(
                ["df", "-P", str(local_cleanup_path)]
            ).decode("utf-8").splitlines()[-1]
            
            available_kb = int(df_output.split()[3])
            available_bytes = available_kb * 1024
            
            if available_bytes >= required_bytes:
                print(f"SYNC cleanup complete. {available_bytes / (1024**3):.1f}GB available (Required: {required_gb}GB).")
                return
            
            print(f"Waiting for NFS... (Avail: {available_bytes / (1024**3):.1f}GB / Need: {required_gb}GB)")
            time.sleep(1)
            
        raise Exception(f"NFS failed to free space after 30 seconds.")
        
    except Exception as e:
        print(f"CRITICAL SYNC CLEANUP FAILURE: {e}")
        raise e

def discover_experiments() -> list:
    """
    models 디렉토리에서 사용 가능한 실험(모델) 목록을 찾습니다.
    
    Returns:
        실험 이름 리스트
    """
    models_dir = "models"
    experiment_names = []
    for name in os.listdir(models_dir):
        path = os.path.join(models_dir, name)
        if os.path.isdir(path) and not name.startswith("__"):
            experiment_names.append(name)
    print(f"발견된 실험: {experiment_names}")
    return experiment_names


def run_experiment(exp_name: str, local_data_path: str): 
    """
    단일 실험(모델)을 실행합니다.
    
    Args:
        exp_name: 실험 이름 (models 디렉토리 내 모델 폴더명)
        local_data_path: 로컬 데이터 저장 경로
    
    Returns:
        실험 결과 딕셔너리
    """
    data_loader.reset_caches()
    print(f"\n========================================")
    print(f"  [실험 시작] 모델: {exp_name}")
    print(f"========================================")
    
    # 모델 설정 및 모듈 로드
    config_module = importlib.import_module(f"models.{exp_name}.config")
    config = config_module.ModelConfig()
    model_module = importlib.import_module(f"models.{exp_name}.model")

    use_nas_directly = getattr(config, "USE_NAS_DIRECTLY", False)
    try:
        transform_func = model_module.Model.get_transforms
    except Exception as e:
        print(f"!!! [{exp_name}] model.py에 get_transforms @staticmethod가 없습니다: {e}")
        raise e

    # 체크포인트 저장 디렉토리 설정
    save_dir = os.path.join(config.CHECKPOINT_DIR, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"체크포인트 저장 경로: {save_dir}")

    # 데이터 변환 함수 준비
    print("Loading data with model-specific transforms...")
    train_transform = transform_func(config, is_train=True)
    test_transform = transform_func(config, is_train=False)

    # 모델 초기화
    print(f"Building model: {config.MODEL_NAME}...")
    model = model_module.Model(config).to(config.DEVICE)

    # 학습 상태 초기화
    start_epoch = 1
    best_test_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_path = os.path.join(save_dir, f"{exp_name}_best.pth")

    # 사전 학습된 가중치 또는 체크포인트 로드
    pretrained_path = getattr(config, "PRETRAINED_WEIGHTS", None)
    load_resume = getattr(config, "BEST_WEIGHT_LOAD", False)
    
    # 전이 학습: 사전 학습된 가중치만 로드
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"\n[Transfer Learning] Loading weights from Phase 1: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location=config.DEVICE)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print("-> Phase 1 weights loaded successfully.")
            print("-> Starting FRESH training (Epoch 1, New Optimizer/Scheduler).")
        except Exception as e:
            print(f"!!! Error loading pretrained weights: {e}")
            print("!!! Starting from scratch.")
    elif load_resume and os.path.exists(best_model_path):
        print(f"\n[Resume] Loading checkpoint: {best_model_path}")
        try:
            checkpoint = torch.load(best_model_path, map_location=config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            if 'optimizer_state_dict' in checkpoint and hasattr(model, 'optimizer'):
                model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and hasattr(model, 'scheduler'):
                model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            best_test_f1 = checkpoint.get('test_f1', 0.0)
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_model_wts = model.state_dict()
            print(f"-> Resumed from Epoch {start_epoch-1}. Next Epoch: {start_epoch}")
        except Exception as e:
            print(f"Error resuming: {e}. Starting from Epoch 1.")

    else:
        print("\n[Fresh Start] Starting training from Epoch 1.")

    # 학습 루프 시작
    print(f"Starting training on {config.DEVICE}...")
    history = []
    epochs_no_improve = 0
    final_epoch_results = {}
    
    # 디렉토리 경로 설정
    required_space_gb = config.MAX_CHUNK_SIZE_GB + 5.0 
    local_train_dir = os.path.join(local_data_path, "train")
    local_test_dir = os.path.join(local_data_path, "test")

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):

        print(f"\n--- Epoch {epoch}/{config.NUM_EPOCHS} ---")

        # 로컬 모드일 경우 학습 데이터 디렉토리 정리
        if not use_nas_directly:
            print(f"[Local Mode] Cleaning /train directory before sync...")
            _sync_cleanup_dir(local_train_dir, required_space_gb)
        
        train_loader = data_loader.get_train_loader(config, train_transform, epoch=epoch, local_data_path=local_data_path)
        train_loss, train_acc, train_f1 = engine.train_one_epoch(
            model, train_loader, config, epoch)

        if not use_nas_directly:
            print(f"[Local Mode] Cleaning /train directory after run...")
            _sync_cleanup_dir(local_train_dir, required_space_gb)
        
        if hasattr(model, 'scheduler'):
            model.scheduler.step()
            current_lr = model.optimizer.param_groups[0]['lr']
            print(f"LR scheduler stepped. Current Base LR: {current_lr:.7f}")

        if not use_nas_directly:
            print(f"[Local Mode] Cleaning /test directory before evaluation...")
            _sync_cleanup_dir(local_test_dir, required_space_gb)
        
        test_loader_generator = data_loader.get_test_loaders_generator(
            config, test_transform, local_data_path
        )
        test_loss, test_acc, test_f1 = engine.evaluate(
            model, test_loader_generator, config
        )

        if not use_nas_directly:
            print(f"[Local Mode] Cleaning /test directory after evaluation...")
            _sync_cleanup_dir(local_test_dir, required_space_gb)
        
        # 학습 히스토리 기록
        history.append({ "epoch": epoch, "test_acc": test_acc, "test_f1": test_f1 })
        final_epoch_results = {'epoch': epoch, 'test_loss': test_loss, 'test_acc': test_acc, 'test_f1': test_f1}

        # 최고 성능 모델 저장
        if test_f1 > best_test_f1:
            print(f"*** New Best F1: {test_f1:.4f} (기존: {best_test_f1:.4f}).")
            best_test_f1 = test_f1
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict()) 
            
            optimizer_state = model.optimizer.state_dict() if hasattr(model, 'optimizer') else None
            scheduler_state = model.scheduler.state_dict() if hasattr(model, 'scheduler') else None

            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer_state,
                'scheduler_state_dict': scheduler_state,
                'test_f1': best_test_f1,
                'test_acc': test_acc
            }, best_model_path)
            print(f"Saved best model checkpoint to {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in F1 for {epochs_no_improve} epoch(s). (Best: {best_test_f1:.4f})")

        # 조기 종료 체크
        if config.USE_EARLY_STOPPING and epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nTest(Val) F1-Score has not improved for {config.EARLY_STOPPING_PATIENCE} epochs.")
            print(f"Early stopping triggered at epoch {epoch}.")
            break 

    # 최고 성능 모델 가중치 복원
    print(f"\nTraining finished. Restoring best model weights (F1: {best_test_f1:.4f}).")
    model.load_state_dict(best_model_wts)

    # 최종 테스트 실행
    print("\n--- [Final Test Phase] ---")
    print(f"Running final test on {exp_name} (best F1: {best_test_f1:.4f}) model...")
    try:
        final_f1_score = final_tester.run_final_test(
            model=model, 
            config=config, 
            local_data_path=local_data_path
        )
    except Exception as e:
        print(f"!!! [Final Test] 치명적인 오류 발생: {e}")
        final_f1_score = 0.0

    # 최종 정리
    if not use_nas_directly:
        print("Final cleanup of /train and /test directories...")
        _sync_cleanup_dir(local_train_dir, required_space_gb)
        _sync_cleanup_dir(local_test_dir, required_space_gb)
    
    print(f"--- [실험 종료] {exp_name} ---")
    
    return {
        "model_name": config.MODEL_NAME,
        "best_test_f1": best_test_f1,
        "final_test_f1": final_f1_score
    }

def main():
    """
    메인 함수: 모든 실험을 순차적으로 실행하고 결과를 요약합니다.
    """
    parser = argparse.ArgumentParser(description="Deepfake Detection Training")
    parser.add_argument(
        "--local-data-path", 
        type=str, 
        required=True, 
        help="Path to local disk storage (/local_datasets/...). This is ignored if USE_NAS_DIRECTLY=True."
    )
    args = parser.parse_args()
    print(f"Using local data path: {args.local_data_path}")
    
    # 사용 가능한 실험 목록 찾기
    experiment_names = discover_experiments()
    all_results = []
    if not experiment_names:
        print("실행할 실험을 'models' 폴더에서 찾지 못했습니다.")
        return
    
    # 각 실험 실행
    for exp_name in experiment_names:
        try:
            result = run_experiment(exp_name, args.local_data_path)
            all_results.append(result)
        except Exception as e:
            print(f"!!! [{exp_name}] 실험 중 오류 발생: {e}")
            all_results.append({
                "model_name": exp_name,
                "best_test_f1": 0.0, 
                "final_test_f1": 0.0,
                "error": str(e)
            })
    print("\n========================================")
    print("          [최종 실험 결과 요약]")
    print("========================================")
    if not all_results:
        print("실행된 실험 결과가 없습니다.")
        return
    all_results.sort(key=lambda x: x.get('final_test_f1', 0.0), reverse=True)
    
    print(f"{'Rank':<5} | {'Model':<25} | {'Val F1 (Best)':<15} | {'Final Test F1':<15}")
    print("-" * 65)
    
    for i, res in enumerate(all_results):
        rank = f"{i+1}."
        val_f1 = res['best_test_f1']
        final_f1 = res.get('final_test_f1', 0.0)
        print(f"{rank:<5} | {res['model_name']:<25} | {val_f1:<15.4f} | {final_f1:<15.4f}")
    print("========================================")

if __name__ == "__main__":
    main()