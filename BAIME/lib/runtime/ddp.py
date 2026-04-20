import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .base_runtime import BaseRuntime, RuntimeContext
from ..core.register import RUNTIME_REGISTER


@RUNTIME_REGISTER.register('ddp')
class DDPRuntime(BaseRuntime):
    """Multi-GPU runtime using native PyTorch DistributedDataParallel.

    Two modes:
    - Local: spawns processes via ``mp.spawn`` (default).
    - Slurm: detected automatically via SLURM_PROCID — each srun task
      is already a separate process, so we just init the process group.
    """

    def launch(self, trainer, args):
        if 'SLURM_PROCID' in os.environ:
            self._launch_slurm(trainer, args)
        else:
            self._launch_local(trainer, args)

    def _launch_local(self, trainer, args):
        devices = args.devices
        if not isinstance(devices, list):
            devices = list(range(devices))
        world_size = len(devices)

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(d) for d in devices)
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

        mp.spawn(
            _worker_fn,
            args=(world_size, trainer, args),
            nprocs=world_size,
            join=True,
        )

    def _launch_slurm(self, trainer, args):
        """Slurm already spawned one process per GPU via srun."""
        local_rank = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])

        os.environ.setdefault('MASTER_ADDR',
            os.environ.get('SLURM_LAUNCH_NODE_IPADDR', 'localhost'))
        os.environ.setdefault('MASTER_PORT', '29500')

        _worker_fn(local_rank, world_size, trainer, args, global_rank=global_rank)


def _prepare_trainer_ddp(trainer):
    """Post-process trainer modules for DDP: wrap model, rebuild dataloaders
    with DistributedSampler."""
    ctx = trainer.ctx
    args = trainer.args

    trainer.MODEL = DDP(trainer.MODEL, device_ids=[ctx.device])

    train_sampler = DistributedSampler(
        trainer.TrainDataloader.dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
    )
    trainer.TrainDataloader = DataLoader(
        trainer.TrainDataloader.dataset,
        batch_size=args.batch_size_per_worker,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.workers,
    )

    if hasattr(trainer, 'ValDataloader'):
        val_sampler = DistributedSampler(
            trainer.ValDataloader.dataset,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=False,
        )
        trainer.ValDataloader = DataLoader(
            trainer.ValDataloader.dataset,
            batch_size=args.batch_size_per_worker,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.workers,
        )


def _worker_fn(local_rank, world_size, trainer, args, global_rank=None):
    """Entry point for each DDP process (spawned or srun-launched)."""
    if global_rank is None:
        global_rank = local_rank

    dist.init_process_group(
        backend='nccl',
        rank=global_rank,
        world_size=world_size,
    )
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    ctx = RuntimeContext(
        rank=global_rank,
        world_size=world_size,
        is_main=(global_rank == 0),
        device=device,
        prepare_trainer=_prepare_trainer_ddp,
    )

    try:
        trainer.train_func(args, ctx)
    finally:
        dist.destroy_process_group()
