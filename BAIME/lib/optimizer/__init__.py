from ..core.register import OPTIMIZER_REGISTER

from .optimizer import *

def get_optimizer(args, model):
    optim_name = args.optimizer
    optim_class = OPTIMIZER_REGISTER.get(optim_name)
    optim_instance = optim_class.init_from_args(args, model)
    return optim_instance

def list_optimizers():
    return OPTIMIZER_REGISTER.list_available()