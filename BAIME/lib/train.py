import os
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
import argparse
import pprint

import ray
from ray import train as ray_train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

from .trainer import get_trainer
from .utils.file import read_cfg, parse_trial_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Deep Learning with Ray')

    # === GENERAL === #
    parser.add_argument('-model', type=str, default="BAIME_Example",
                        help='Model/Task/Trial name')
    parser.add_argument('-reset', action='store_true',
                        help='Reset saved model logs and weights')
    parser.add_argument('-cfg', type=str,
                        help='Configuration file')
    parser.add_argument('-out_path', type=str, default="out",
                        help='path to out directory')

    # === RAY DISTRIBUTED === #
    parser.add_argument('-ray_address', type=str, default=None,
                        help='Ray cluster address (None for local)')
    parser.add_argument('-num_workers', type=int, default=1,
                        help='Number of distributed training workers')
    parser.add_argument('-gpus_per_worker', type=float, default=1.0,
                        help='GPUs per worker (can be fractional)')
    parser.add_argument('-cpus_per_worker', type=int, default=2,
                        help='CPUs per worker')
    
    # === Trainer === #
    parser.add_argument('-trainer', type=str, default='BaseTrainer',
                        help='Trainer to choose')
    parser.add_argument('-epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('-save_every', type=int, default=10,
                        help='Save frequency')

    # === Architecture === #
    parser.add_argument('-arch', type=str, default='ExampleMLP',
                        help='Architecture to choose')
    
    # === Dataset === #
    parser.add_argument('-train_data_path', type=str, default="data",
                        help='path to dataset directory')
    parser.add_argument('-val_data_path', type=str, default=None,
                        help='path to validation dataset directory (if different from training data)')
    parser.add_argument('-train_dataset', type=str, default='ExampleRandomDataset',
                        help='Dataset to choose')
    parser.add_argument('-val_dataset', type=str, default=None,
                        help='Validation dataset to choose (if different from training dataset)')
    parser.add_argument('-batch_size_per_worker', type=int, default=8,
                        help='batch size per gpu')
    parser.add_argument('-shuffle', type=bool, default=True,
                        help='Shuffle dataset')
    parser.add_argument('-workers', type=int, default=2,
                        help='number of dataloader workers')

    # === Loss === #
    parser.add_argument('-loss', type=str, default='ExampleLoss',
                        help='Loss function to choose')

    # === Optimizer & Scheduler === #
    parser.add_argument('-optimizer', type=str, default='adam',
                        help='Optimizer function to choose')
    parser.add_argument('-scheduler', type=str, default='warmup',
                        help='Scheduler function to choose')
    parser.add_argument('-lr_start', type=float, default=5e-4,
                        help='Initial Learning Rate')
    parser.add_argument('-lr_end', type=float, default=1e-6,
                        help='Final Learning Rate')
    parser.add_argument('-lr_warmup', type=int, default=0,
                        help='warmup epochs for learning rate')
    
    # === Misc === #
    parser.add_argument('-slurm', action='store_true', default=False,
                        help='Use this flag if running on a SLURM cluster')

    args = parser.parse_args()
    # === Read CFG File === #
    args = read_cfg(args)
    return args


def train():
    args = parse_args()
    pprint.pprint(vars(args))
    # === Prepare Output Directories For the Trial === #
    # if running on SLURM, the directories are prepared in the slurm launch script
    args = parse_trial_dir(args, check=not args.slurm)

    # === Backup Config File === #
    if args.cfg is not None:
        os.system(f'cp {args.cfg} {args.out_trial_path}/{args.model}.yaml')

    # === Initialize Ray === #
    if args.ray_address:
        # Connect to existing Ray cluster
        ray.init(address=args.ray_address)
        print(f"Connected to Ray cluster at {args.ray_address}")
    else:
        # Start local Ray instance
        ray.init()
        print("Started local Ray instance")
    # Print cluster info
    print(f"Ray cluster resources: {ray.cluster_resources()}")
    
    # === Prepare Scaling Config === #
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=args.gpus_per_worker > 0,
        resources_per_worker={
            "CPU": args.cpus_per_worker,
            "GPU": args.gpus_per_worker
        }
    )

    # === Prepare Trainer Config === #
    train_loop_config = {"args": args}

    # === Prepare Runtime Config === #
    run_config = ray_train.RunConfig(
        name='ray',
        storage_path=args.out_trial_path,
        checkpoint_config=ray_train.CheckpointConfig(
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min",
            num_to_keep=3,
        ),
    )

    # === Create a Ray Trainer === #
    trainer_per_worker = get_trainer(args)
    trainer = TorchTrainer(
        train_loop_per_worker=trainer_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config
    )

    # === Start Training === #
    print(f"Starting distributed training with {args.num_workers} workers...")
    result = trainer.fit()
    
    print("Training completed!")
    print(f"Results: {result}")
    
    # === Shutdown Ray === #
    ray.shutdown()

if __name__ == "__main__":
    train()
