# Deepfake Detection

This repository provides a highly scalable and robust framework for training image classification models to detect deepfakes. Built with memory efficiency and massive dataset handling in mind, it leverages **WebDataset** to stream terabytes of data directly from network-attached storage (NAS) or sync them dynamically to local fast storage.

## Key Architecture Highlights

- **WebDataset-based Pipeline**: Processes huge amounts of Real (Label 0) and Fake (Label 1) face images stored as `.tar` shards. Supports chunking (e.g., 35GB per epoch) to control local disk footprint.
- **Dynamic Data Syncing**: Can stream directly from NAS (`USE_NAS_DIRECTLY = True`) or intelligently sync `.tar` shards via `rsync` to local NVMe storage on a per-epoch basis.
- **Optimized Training Engine**: 
  - Implements **Automatic Mixed Precision (AMP)** and **Gradient Accumulation** to train large models with high batch sizes on limited GPU memory.
  - Built-in LR scheduling and model checkpointing based on best **F1-Score**.
- **Modular Model Ecosystem**: Models are stored in the `models/` directory. Each model defines its own network architecture, optimizer, and image transformations, extending a global configuration.

## Directory Structure

```text
deepfake-detection/
├── config_global.py       # Global settings (datasets, hyperparams, thresholds)
├── data_loader.py         # WebDataset logic, chunking, and local syncing
├── engine.py              # Core AMP train/eval loops (train_one_epoch, evaluate)
├── main.py                # Entry point: discovers models, runs training & validation
├── final_tester.py        # Final evaluation logic for the best saved weights
└── models/                # Directory for experiment/model architectures
    ├── resnet50/
    │   ├── config.py      # Inherits GlobalConfig
    │   └── model.py       # PyTorch model definition & get_transforms()
    └── ...
```

## Installation

```bash
pip install -r requirements.txt
```
*Core dependencies include: `torch`, `torchvision`, `webdataset`, `timm`, `tensorboard`, `mlflow`.*

## Dataset Configuration

The datasets are defined in `config_global.py` via the `TRAIN_SHARDS` and `TEST_SHARDS` variables. Data is expected to be in WebDataset `.tar` format.
- **Fake Images (Deepfakes)** are mapped to label **1**.
- **Real Images** are mapped to label **0**.

Brace expansion is supported for easy shard definitions:
```python
("./data/FFHQ/FFHQ_1/FFHQ_1-{000000..000048}.tar", 0) # Real
("./data/SFHQ/SFHQ_1A/SFHQ_1A-{000000..000223}.tar", 1) # Fake
```

## How to Add a Custom Model

The framework automatically discovers models placed inside the `models/` directory. To add a new model (e.g., `my_custom_cnn`), create a folder `models/my_custom_cnn/` containing:

### 1. `config.py`
Define a `ModelConfig` class that inherits from `GlobalConfig` to override specific hyperparameters:
```python
from config_global import GlobalConfig
from typing import List

class ModelConfig(GlobalConfig):
    MODEL_NAME: str = "MyCustomCNN"
    IMG_SIZE: int = 256 
    LR: float = 0.0005
    MEAN: List[float] = [0.485, 0.456, 0.406]
    STD: List[float] = [0.229, 0.224, 0.225]
```

### 2. `model.py`
Create a `Model` class that initializes your network, criterion, and optimizer. It must also implement `step()`, `predict()`, and a static method `get_transforms()`:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.network = ... # Define your PyTorch network here
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=config.LR)
    
    @staticmethod
    def get_transforms(config, is_train: bool):
        # Must return torchvision transforms
        if is_train:
            return v2.Compose([
                v2.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=config.MEAN, std=config.STD)
            ])
        else:
            return v2.Compose([...]) # Eval transforms

    def forward(self, x):
        return self.network(x)

    def step(self, images, labels):
        # Used by engine.py for training
        self.optimizer.zero_grad()
        preds = self.forward(images)
        loss = self.criterion(preds, labels)
        loss.backward()
        self.optimizer.step()
        return preds, loss.item()
```

## Usage

To train all the models discovered in the `models/` directory:

```bash
python main.py --local-data-path /mnt/fast_nvme/local_datasets
```

- `--local-data-path`: The local directory where data shards will be synced. Ensure the path has sufficient space for `MAX_CHUNK_SIZE_GB` (set in `config_global.py`).
- If `USE_NAS_DIRECTLY = True` in `config_global.py`, the pipeline bypasses local syncing and streams directly from the defined network paths.

## Evaluation & Checkpointing

During training, the framework logs validation accuracy and F1-Scores. 
- The model with the **best validation F1-Score** is automatically saved as a checkpoint in `CHECKPOINT_DIR`.
- If the F1-Score does not improve for `EARLY_STOPPING_PATIENCE` epochs, training is halted.
- A final test is triggered automatically at the end of training using the best saved weights.
