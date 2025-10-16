# <img src="assets/baime_logo.png" width="30"> BAIME: Build AI Models with Ease.

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-black.json)](https://github.com/copier-org/copier)

A Concise Blueprint for Deep Learing Project

## Features

### 1. Scaling

BAIME levrages **PyTorch** and **Ray** frameworks to facilitate the scaling of AI and python applications. 

Furthermore, BAIME offers lanuch wrapper for **Slurm** system.

### 2. Guidance
BAIME guides users in structuring their deep learning projects into serval modules, primarily located within **lib/** directory:

- **arch/**: Contains model architecture definitions.
- **dataset/**: Manages data reading and processing.
- **loss/**: Defines loss functions.
- **optimizer/**: Implements weight optimization algorithms.
- **scheduler/**: Handles learning rate scheduling.
- **trainer/**: Encapsulates the training process logic, orchestrating the use of the aforementioned modules.
- **inferencer/**: Contains the logic for the inference process.
- **utils/**: Provides various helper functions.

BAIME includes example implementations for these modules and offers a registration mechanism to streamline their invocation.

### 3. Trial Management

BAIME facilitates the management of each experiment/trial/task/job through its configuration system. Each trial's parameters are defined within a YAML file.

```yaml
# === GENERAL ===
model: BAIME_Example # your trial/experiment/task name
out_path: ${your_output_path} # absolute path
reset: true # whether clear trial folder or not

# === RAY DISTRIBUTED ===
ray_address: null
num_workers: 1
gpus_per_worker: 1.0
cpus_per_worker: 2

# === Trainer ===
trainer: BaseTrainer # registered trainer module
epochs: 10
save_every: 1

# === Architecture ===
arch: ExampleMLP # registered arch module
input_dim: 1024
output_dim: 32

# === Dataset ===
train_data_path: ${your_data_path}
val_data_path: null
train_dataset: ExampleRandomDataset # registered dataset module
val_dataset: null
batch_size_per_worker: 8
shuffle: true
workers: 2

# === Loss ===
loss: ExampleLoss # egistered loss module

# === Optimizer & Scheduler ===
optimizer: adam # registered optimizer module
scheduler: warmup # registered scheduler module
lr_start: 0.0005
lr_end: 0.000001
lr_warmup: 10
```

```bash
python -m lib.train -cfg ${CONFIG_FILE_PATH}
```

The corresponding results are formatted as follows:

```bash
${trial_name}/
|
|--- checkpoints/ #model weights
|   |--- Epoch_0100/
|   |--- ${trial_name}_Epoch0100.pt
|   |--- Epoch_0200/
|
|--- logs/ #runtime log for each worker (GPU)
|   |--- worker_0.out # stdout
|   |--- worker_0.err # stderr
|   |--- workser_1.out
|   |--- workser_1.err
|
|--- ray/ #ray framework related files
|
|--- tensorboard/ #tensorboard log file
|   |--- events.out.tfevents.xxxxxx
|
|--- slurm/ # slurm related files
|   |--- ${trial_on_slurm_name}_${timestamp}.sh
|   |--- ${trial_on_slurm_name}_${timestamp}_out.log
|   |--- ${trial_on_slurm_name}_${timestamp}_err.log
|   
|--- ${trial_name}.yaml #config yaml (backup)
```

## Usage

```bash
copier copy --trust https://github.com/RyuuYou0529/BAIME ${your_project_workspace}
```

## Acknowledgments
This project is inspired by **[ramyamounir/Template](https://github.com/ramyamounir/Template)** and **[BasicSR](https://github.com/XPixelGroup/BasicSR)**.