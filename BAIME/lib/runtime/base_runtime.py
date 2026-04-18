import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class RuntimeContext:
    """Injected into trainer by the runtime. Trainer uses this instead of
    querying Ray / DDP internals directly."""
    rank: int
    world_size: int
    is_main: bool
    device: torch.device


class BaseRuntime(ABC):
    """Abstract runtime that wraps a trainer and deploys it on device(s)."""

    @abstractmethod
    def launch(self, trainer, args):
        """Launch the trainer on the configured device(s).

        Args:
            trainer: A trainer instance (e.g. BaseTrainer).
            args: Parsed arguments namespace.
        """
        raise NotImplementedError
