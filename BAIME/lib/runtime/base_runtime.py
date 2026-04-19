import torch
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Callable


@dataclass
class RuntimeContext:
    """Injected into trainer by the runtime. Trainer uses this for
    rank-aware logging/saving — nothing more."""
    rank: int
    world_size: int
    is_main: bool
    device: torch.device
    # Called by trainer after init_modules(); runtime uses this to
    # post-process modules (e.g. wrap model in DDP, inject sampler).
    prepare_trainer: Callable = field(default=lambda trainer: None)


class BaseRuntime(ABC):
    """Abstract runtime that wraps a trainer and deploys it on device(s)."""

    @abstractmethod
    def launch(self, trainer, args):
        raise NotImplementedError
