import os
import argparse
import pprint

from .trainer import get_trainer
from .runtime import get_runtime
from .utils.file import read_cfg, parse_trial_dir


def parse_args():
    parser = argparse.ArgumentParser(description='BAIME: Build AI Models with Ease')

    # === GENERAL === #
    parser.add_argument('--model', type=str, default="BAIME_Example",
                        help='Model/Task/Trial name')
    parser.add_argument('--reset', action='store_true',
                        help='Reset saved model logs and weights')
    parser.add_argument('--cfg', type=str,
                        help='Configuration file')
    parser.add_argument('--out_path', type=str, default="out",
                        help='path to out directory')

    # === RUNTIME === #
    parser.add_argument('--runtime', type=str, default='standalone',
                        help='Runtime to use (standalone / ddp)')
    parser.add_argument('--devices', type=str, default='0',
                        help='Device IDs, e.g. "0" or "0,1,2,3"')

    # === Trainer === #
    parser.add_argument('--trainer', type=str, default='BaseTrainer',
                        help='Trainer to choose')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save frequency')

    # === Architecture === #
    parser.add_argument('--arch', type=str, default='ExampleMLP',
                        help='Architecture to choose')

    # === Dataset === #
    parser.add_argument('--train_data_path', type=str, default="data",
                        help='path to dataset directory')
    parser.add_argument('--val_data_path', type=str, default=None,
                        help='path to validation dataset directory')
    parser.add_argument('--train_dataset', type=str, default='ExampleRandomDataset',
                        help='Dataset to choose')
    parser.add_argument('--val_dataset', type=str, default=None,
                        help='Validation dataset to choose')
    parser.add_argument('--batch_size_per_worker', type=int, default=8,
                        help='batch size per gpu')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Shuffle dataset')
    parser.add_argument('--workers', type=int, default=2,
                        help='number of dataloader workers')

    # === Loss === #
    parser.add_argument('--loss', type=str, default='ExampleLoss',
                        help='Loss function to choose')

    # === Optimizer & Scheduler === #
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer function to choose')
    parser.add_argument('--scheduler', type=str, default='warmup',
                        help='Scheduler function to choose')
    parser.add_argument('--lr_start', type=float, default=5e-4,
                        help='Initial Learning Rate')
    parser.add_argument('--lr_end', type=float, default=1e-6,
                        help='Final Learning Rate')
    parser.add_argument('--lr_warmup', type=int, default=0,
                        help='warmup epochs for learning rate')

    # === Misc === #
    parser.add_argument('--slurm', action='store_true', default=False,
                        help='Use this flag if running on a SLURM cluster')

    args = parser.parse_args()
    # === Read CFG File === #
    args = read_cfg(args)
    # === Parse devices string into list of ints === #
    if isinstance(args.devices, list):
        args.devices = [int(d) for d in args.devices]
    else:
        args.devices = [int(d) for d in str(args.devices).split(',')]
    return args


def train():
    args = parse_args()
    pprint.pprint(vars(args))
    # === Prepare Output Directories For the Trial === #
    args = parse_trial_dir(args, check=not args.slurm)

    # === Backup Config File === #
    if args.cfg is not None:
        os.system(f'cp {args.cfg} {args.out_trial_path}/{args.model}.yaml')

    # === Get Trainer and Runtime === #
    trainer = get_trainer(args)
    runtime = get_runtime(args)

    # === Start Training === #
    print(f"Starting training with runtime={args.runtime}, devices={args.devices}")
    runtime.launch(trainer, args)
    print("Training completed!")


if __name__ == "__main__":
    train()
