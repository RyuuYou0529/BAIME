import os
import importlib

from ..core.register import LOSS_REGISTER

def _auto_import_modules():
    current_dir = os.path.dirname(__file__)
    for fname in os.listdir(current_dir):
        if fname.endswith('_loss.py') and fname != '__init__.py':
            module_name = f"{__name__}.{fname[:-3]}"
            importlib.import_module(module_name)
_auto_import_modules()

def get_loss(args):
    loss_name = args.loss
    loss_class = LOSS_REGISTER.get(loss_name)
    loss_instance = loss_class.init_from_args(args)
    return loss_instance

def list_losses():
    return LOSS_REGISTER.list_available()