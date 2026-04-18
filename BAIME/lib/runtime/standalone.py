import os
import torch

from .base_runtime import BaseRuntime, RuntimeContext
from ..core.register import RUNTIME_REGISTER


@RUNTIME_REGISTER.register('standalone')
class StandaloneRuntime(BaseRuntime):
    """Single-GPU runtime. Runs trainer in the main process — fully
    debuggable via IDE breakpoints (no subprocess spawning)."""

    def launch(self, trainer, args):
        # --- resolve device ---
        devices = args.devices
        device_id = devices[0] if isinstance(devices, list) else devices
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ctx = RuntimeContext(rank=0, world_size=1, is_main=True, device=device)

        # --- run trainer directly in main process ---
        trainer.train_func(args, ctx)
