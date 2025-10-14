import os
import importlib

from ..core.register import TRAINER_REGISTER

def _auto_import_modules():
    current_dir = os.path.dirname(__file__)
    for fname in os.listdir(current_dir):
        if fname.endswith('_trainer.py') and fname != '__init__.py':
            module_name = f"{__name__}.{fname[:-3]}"
            importlib.import_module(module_name)
_auto_import_modules()

def get_trainer(args):
    trainer_name = args.trainer
    trainer_class = TRAINER_REGISTER.get(trainer_name)
    trainer_instance = trainer_class(args)
    return trainer_instance

def list_trainers():
    return TRAINER_REGISTER.list_available()