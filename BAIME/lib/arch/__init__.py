import os
import importlib

from ..core.register import ARCH_REGISTER

def _auto_import_modules():
    current_dir = os.path.dirname(__file__)
    for fname in os.listdir(current_dir):
        if fname.endswith('_arch.py') and fname != '__init__.py':
            module_name = f"{__name__}.{fname[:-3]}"
            importlib.import_module(module_name)
_auto_import_modules()

def get_model(args):
    arch_name = args.arch
    model_class = ARCH_REGISTER.get(arch_name)
    model_instance = model_class.init_from_args(args)
    return model_instance

def list_models():
    return ARCH_REGISTER.list_available()