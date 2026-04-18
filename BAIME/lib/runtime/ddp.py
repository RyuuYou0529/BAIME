import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .base_runtime import BaseRuntime, RuntimeContext
from ..core.register import RUNTIME_REGISTER


@RUNTIME_REGISTER.register('ddp')
class DDPRuntime(BaseRuntime):
    """Multi-GPU runtime using native PyTorch DistributedDataParallel.

    Spawns one process per GPU via ``mp.spawn``. Each process initialises
    ``torch.distributed``, and the trainer receives a ``RuntimeContext``
    with the correct rank / world_size / device.
    """

    def launch(self, trainer, args):
        devices = args.devices
        if not isinstance(devices, list):
            devices = list(range(devices))
        world_size = len(devices)

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(d) for d in devices)
        # After setting CUDA_VISIBLE_DEVICES, local ranks are 0..N-1
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

        mp.spawn(
            _worker_fn,
            args=(world_size, trainer, args),
            nprocs=world_size,
            join=True,
        )


def _worker_fn(local_rank, world_size, trainer, args):
    """Entry point executed in each spawned process."""
    # --- init process group ---
    dist.init_process_group(
        backend='nccl',
        rank=local_rank,
        world_size=world_size,
    )
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    ctx = RuntimeContext(
        rank=local_rank,
        world_size=world_size,
        is_main=(local_rank == 0),
        device=device,
    )

    try:
        trainer.train_func(args, ctx)
    finally:
        dist.destroy_process_group()
