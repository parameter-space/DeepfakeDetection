# data_loader.py
import subprocess
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import re
from typing import List, Tuple, Generator 
import random
import copy
import braceexpand 
import os 

from config_global import GlobalConfig
from datasets.diffface import DiffusionFaceDataset

def _expand_and_label_shards(
    shard_list_with_labels: List[Tuple[str, int]]
) -> List[Tuple[str, int, int]]:
    """
    브레이스 확장 패턴을 사용하여 샤드 경로를 확장하고 파일 크기를 가져옵니다.
    
    Args:
        shard_list_with_labels: (경로패턴, 레이블) 튜플 리스트
    
    Returns:
        (경로, 레이블, 파일크기) 튜플 리스트
    """
    expanded_list = []
    print("Expanding shards and getting file sizes... (This may take a moment)")
    for shard_str, label in shard_list_with_labels:
        expanded_paths = list(braceexpand.braceexpand(shard_str))
        for path in expanded_paths:
            try:
                file_size_bytes = os.path.getsize(path) 
                expanded_list.append((path, label, file_size_bytes)) 
            except FileNotFoundError:
                print(f"  > NAS Warning: {path} not found. Skipping.")
                continue
            except Exception as e:
                print(f"  > NAS Error (skipping {path}): {e}")
                continue
    print(f"Found {len(expanded_list)} total shard files.")
    return expanded_list


def _sync_shards_to_local(
    nas_shard_tuples: List[Tuple], 
    local_target_dir: Path 
) -> List[Tuple[str, int]]:
    """
    NAS에서 로컬 디렉토리로 샤드 파일들을 동기화합니다.
    
    Args:
        nas_shard_tuples: NAS 샤드 (경로, 레이블) 튜플 리스트
        local_target_dir: 로컬 타겟 디렉토리
    
    Returns:
        로컬 샤드 (경로, 레이블) 튜플 리스트
    """
    local_target_dir.mkdir(parents=True, exist_ok=True)
    local_shard_tuples = []
    if not nas_shard_tuples:
        return []
    print(f"Syncing {len(nas_shard_tuples)} shards from NAS to {local_target_dir}...")
    for nas_tuple in nas_shard_tuples:
        nas_path_str = nas_tuple[0]
        label = nas_tuple[1]
        nas_path = Path(nas_path_str)
        if not nas_path.exists():
            print(f"  > NAS Warning: {nas_path_str} not found. Skipping.")
            continue
        local_path = local_target_dir / nas_path.name
        subprocess.run(
            ["rsync", "-ah", str(nas_path_str), str(local_path)],
            check=True
        )
        local_shard_tuples.append((str(local_path), label))
    print(f"Sync complete. {len(local_shard_tuples)} shards are now local.")
    return local_shard_tuples

def _chunk_list_by_size(
    flat_list_with_size: List[Tuple[str, int, int]], 
    max_size_bytes: int, 
    dataset_name: str
) -> List[List[Tuple[str, int]]]:
    """
    파일 크기를 기준으로 샤드 리스트를 청크로 나눕니다.
    
    Args:
        flat_list_with_size: (경로, 레이블, 파일크기) 튜플 리스트
        max_size_bytes: 최대 청크 크기 (바이트)
        dataset_name: 데이터셋 이름 (로깅용)
    
    Returns:
        청크 리스트 (각 청크는 (경로, 레이블) 튜플 리스트)
    """
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    for (path, label, size_bytes) in flat_list_with_size:
        if current_chunk_size + size_bytes > max_size_bytes and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [(path, label)]
            current_chunk_size = size_bytes
        else:
            current_chunk.append((path, label))
            current_chunk_size += size_bytes
    if current_chunk:
        chunks.append(current_chunk)
    print(f"Loaded {dataset_name} shards: {len(flat_list_with_size)} files -> {len(chunks)} chunks (size_limit={max_size_bytes / (1024**3):.1f}GB)")
    return chunks


# 전역 캐시 변수
_TRAIN_SHARD_CHUNKS = None 
_TEST_SHARDS_CHUNKS = None
_TEST_SHARDS_FLAT_NAS_WITH_SIZE = None

def reset_caches():
    """데이터로더 캐시를 초기화합니다."""
    global _TRAIN_SHARD_CHUNKS, _TEST_SHARDS_CHUNKS, _TEST_SHARDS_FLAT_NAS_WITH_SIZE
    print("Resetting data_loader caches...")
    _TRAIN_SHARD_CHUNKS = None
    _TEST_SHARDS_CHUNKS = None
    _TEST_SHARDS_FLAT_NAS_WITH_SIZE = None

def get_train_loader(
    config: GlobalConfig, 
    transform: v2.Compose, 
    epoch: int, 
    local_data_path: str
) -> DataLoader:
    """
    학습용 데이터로더를 생성합니다.
    
    Args:
        config: 전역 설정
        transform: 이미지 변환 함수
        epoch: 현재 에포크 번호
        local_data_path: 로컬 데이터 경로
    
    Returns:
        학습용 DataLoader
    """
    global _TRAIN_SHARD_CHUNKS

    # 첫 호출 시 학습 샤드 초기화
    if _TRAIN_SHARD_CHUNKS is None:
        print("Initializing 'Train' Shards... (This happens only once)")
        flat_shards_with_size = _expand_and_label_shards(config.TRAIN_SHARDS)
        print(f"Shuffling {len(flat_shards_with_size)} (Fake+Real) train shards...")
        random.shuffle(flat_shards_with_size)
        max_bytes = int(config.MAX_CHUNK_SIZE_GB * (1024**3))
        _TRAIN_SHARD_CHUNKS = _chunk_list_by_size(
            flat_shards_with_size, 
            max_bytes, 
            "Train (Combined)"
        )

    # 현재 에포크에 사용할 청크 선택
    chunk_idx = (epoch - 1) % len(_TRAIN_SHARD_CHUNKS)
    nas_shards_for_this_epoch = _TRAIN_SHARD_CHUNKS[chunk_idx]

    # NAS 직접 모드 또는 로컬 복사 모드
    if getattr(config, "USE_NAS_DIRECTLY", False):
        print(f"  > Epoch {epoch} [Train]: Loading {len(nas_shards_for_this_epoch)} shards directly from NAS.")
        shard_tuples = nas_shards_for_this_epoch
    else:
        local_train_dir = Path(local_data_path) / "train"
        shard_tuples = _sync_shards_to_local(
            nas_shards_for_this_epoch, 
            local_train_dir 
        )
        print(f"  > Epoch {epoch} [Train]: Loading {len(shard_tuples)} (Fake+Real) shards (from {local_train_dir}).")

    dataset = DiffusionFaceDataset(
        shard_tuples=shard_tuples,
        shuffle_size=config.SHUFFLE_SIZE, 
        batch_size=config.BATCH_SIZE,
        transform=transform,
    )

    return DataLoader(dataset, batch_size=None, batch_sampler=None, num_workers=config.NUM_WORKERS)


def get_test_loaders_generator(
    config: GlobalConfig, 
    transform: v2.Compose, 
    local_data_path: str
) -> Generator[DataLoader, None, None]: 
    """
    테스트/검증용 데이터로더 생성기를 반환합니다.
    청크별로 순차적으로 DataLoader를 생성합니다.
    
    Args:
        config: 전역 설정
        transform: 이미지 변환 함수
        local_data_path: 로컬 데이터 경로
    
    Yields:
        테스트용 DataLoader (청크별)
    """
    global _TEST_SHARDS_CHUNKS, _TEST_SHARDS_FLAT_NAS_WITH_SIZE
    
    # 첫 호출 시 테스트 샤드 초기화
    if _TEST_SHARDS_FLAT_NAS_WITH_SIZE is None:
        print("Initializing 'Test(Val)' FULL (Fake+Real) Shards (NAS Paths & Sizes)...")
        _TEST_SHARDS_FLAT_NAS_WITH_SIZE = _expand_and_label_shards(config.TEST_SHARDS)

    if _TEST_SHARDS_CHUNKS is None:
        print(f"Chunking 'Test(Val)' shards by size limit (Max {config.MAX_CHUNK_SIZE_GB}GB)...")
        max_bytes = int(config.MAX_CHUNK_SIZE_GB * (1024**3))
        _TEST_SHARDS_CHUNKS = _chunk_list_by_size(
            _TEST_SHARDS_FLAT_NAS_WITH_SIZE,
            max_bytes,
            "Test (Combined)"
        )

    total_chunks = len(_TEST_SHARDS_CHUNKS)
    
    # 각 청크에 대해 DataLoader 생성
    if getattr(config, "USE_NAS_DIRECTLY", False):
        for i, nas_chunk_with_labels in enumerate(_TEST_SHARDS_CHUNKS):
            print(f"\n--- Loading Test Chunk {i+1}/{total_chunks} directly from NAS ---")
            if not nas_chunk_with_labels:
                print(f"  > Test Chunk {i+1} is empty. Skipping.")
                continue
            
            shard_tuples = nas_chunk_with_labels
            print(f"  > [Test Chunk {i+1}]: Loading {len(shard_tuples)} shards (from NAS).")

            dataset = DiffusionFaceDataset(
                shard_tuples=shard_tuples,
                shuffle_size=0,
                batch_size=config.BATCH_SIZE,
                transform=transform,
            )
            yield DataLoader(dataset, batch_size=None, batch_sampler=None, num_workers=config.NUM_WORKERS)

    else:
        local_test_dir = Path(local_data_path) / "test"
        for i, nas_chunk_with_labels in enumerate(_TEST_SHARDS_CHUNKS):
            print(f"\n--- Loading Test Chunk {i+1}/{total_chunks} (Syncing to Local) ---")
            
            shard_tuples = _sync_shards_to_local(
                nas_chunk_with_labels,
                local_test_dir
            )
            
            if not shard_tuples:
                print(f"  > Test Chunk {i+1} is empty. Skipping.")
                continue
                
            print(f"  > [Test Chunk {i+1}]: Loading {len(shard_tuples)} shards (from {local_test_dir}).")

            dataset = DiffusionFaceDataset(
                shard_tuples=shard_tuples,
                shuffle_size=0,
                batch_size=config.BATCH_SIZE,
                transform=transform,
            )
            yield DataLoader(dataset, batch_size=None, batch_sampler=None, num_workers=config.NUM_WORKERS)
        
    print("--- All Test Chunks Processed ---")