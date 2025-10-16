import os
import importlib
from ..core.register import DATASET_REGISTER

def _auto_import_modules():
    current_dir = os.path.dirname(__file__)
    for fname in os.listdir(current_dir):
        if fname.endswith('_dataset.py') and fname != '__init__.py':
            module_name = f"{__name__}.{fname[:-3]}"
            importlib.import_module(module_name)
_auto_import_modules()

def get_dataset(args, mode='train'):
    if mode == 'train':
        dataset_name = args.train_dataset
    elif mode == 'val':
        dataset_name = args.val_dataset
    dataset_class = DATASET_REGISTER.get(dataset_name)
    dataset_instance = dataset_class.init_from_args(args)
    return dataset_instance

def list_datasets():
    return DATASET_REGISTER.list_available()