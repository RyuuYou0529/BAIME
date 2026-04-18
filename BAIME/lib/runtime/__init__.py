from ..core.register import RUNTIME_REGISTER

from .standalone import *
from .ddp import *


def get_runtime(args):
    runtime_name = args.runtime
    runtime_class = RUNTIME_REGISTER.get(runtime_name)
    return runtime_class()


def list_runtimes():
    return RUNTIME_REGISTER.list_available()
