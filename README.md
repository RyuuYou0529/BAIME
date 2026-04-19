# <img src="assets/baime_logo.png" width="30"> BAIME: Build AI Models with Ease.

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-black.json)](https://github.com/copier-org/copier)

A Concise Blueprint for Deep Learning Projects.

## Features

### 1. Modular Design

BAIME guides users in structuring their deep learning projects into modules within the **lib/** directory:

- **arch/**: Model architecture definitions.
- **dataset/**: Data reading and processing.
- **loss/**: Loss functions.
- **optimizer/**: Weight optimization algorithms.
- **scheduler/**: Learning rate scheduling.
- **trainer/**: Training logic — orchestrates the modules above. Runtime-unaware.
- **inferencer/**: Inference logic.
- **runtime/**: Deployment layer — handles device placement and distributed training.
- **utils/**: Helper functions.

Each module has a base class and a registration mechanism. You implement your own by inheriting the base class and registering it with a decorator:

```python
@ARCH_REGISTER.register('MyModel')
class MyModel(BaseModel):
    ...
```

### 2. Runtime

BAIME decouples **training logic** from **deployment**. The trainer doesn't know or care how many GPUs are running — it just trains. The runtime handles the rest.

| Runtime | Description | Use Case |
|---------|-------------|----------|
| `standalone` | Single GPU, runs in the main process | Debugging, quick tests, IDE breakpoints |
| `ddp` | Multi-GPU via PyTorch DistributedDataParallel | Serious training |

Furthermore, BAIME offers a launch wrapper for **Slurm** systems.

### 3. Trial Management

Each experiment is configured via a YAML file:

```yaml
# === GENERAL ===
model: BAIME_Example        # trial/experiment name
out_path: ${your_output_path}
reset: true                  # clear trial folder on start

# === RUNTIME ===
runtime: standalone          # standalone / ddp
devices: [0]                 # GPU device IDs

# === Trainer ===
trainer: BaseTrainer
epochs: 10
save_every: 1

# === Architecture ===
arch: ExampleMLP
input_dim: 1024
output_dim: 32

# === Dataset ===
train_data_path: ${your_data_path}
val_data_path: null
train_dataset: ExampleRandomDataset
val_dataset: null
batch_size_per_worker: 8
shuffle: true
workers: 2

# === Loss ===
loss: ExampleLoss

# === Optimizer & Scheduler ===
optimizer: adam
scheduler: warmup
lr_start: 0.0005
lr_end: 0.000001
lr_warmup: 10
```

Output directory structure:

```
${trial_name}/
├── checkpoints/
│   ├── Epoch_0001/
│   │   └── ${trial_name}_Epoch0001.pt
│   └── Epoch_0010/
├── logs/
│   ├── worker_0.out
│   └── worker_0.err
├── tensorboard/
│   └── events.out.tfevents.xxxxxx
├── slurm/                          # only when using Slurm
│   ├── ${job_name}.sh
│   ├── ${job_name}_out.log
│   └── ${job_name}_err.log
└── ${trial_name}.yaml              # config backup
```

## Quick Start

### 1. Create a new project from the template

```bash
copier copy --trust https://github.com/RyuuYou0529/BAIME ${your_project_workspace}
```

### 2. Run training

Single GPU (debuggable — set breakpoints in your IDE):

```bash
python -m lib.train --cfg config/BAIME.yaml
```

This defaults to `standalone` runtime on device `0`. Override via CLI:

```bash
python -m lib.train --cfg config/BAIME.yaml --runtime standalone --devices 0
```

Multi-GPU DDP:

```bash
python -m lib.train --cfg config/BAIME.yaml --runtime ddp --devices 0,1,2,3
```

Or set it in the YAML:

```yaml
runtime: ddp
devices: [0, 1, 2, 3]
```

CLI arguments override YAML values.

### 3. Customize modules

1. Create your module file (e.g. `lib/arch/my_arch.py`).
2. Inherit the base class and register it:

```python
from ..core.register import ARCH_REGISTER
from .base_arch import BaseModel

@ARCH_REGISTER.register('MyModel')
class MyModel(BaseModel):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = ...

    @classmethod
    def init_from_args(cls, args):
        return cls(args.hidden_dim)

    def forward(self, x):
        return self.net(x)
```

3. Reference it in your config YAML:

```yaml
arch: MyModel
hidden_dim: 256
```

The same pattern applies to `dataset`, `loss`, `optimizer`, `scheduler`, and `trainer`.

## Acknowledgments

This project is inspired by **[ramyamounir/Template](https://github.com/ramyamounir/Template)** and **[BasicSR](https://github.com/XPixelGroup/BasicSR)**.
