from ..core.register import SCHEDULER_REGISTER

from .scheduler import *

def get_scheduler(args):
    sched_name = args.scheduler
    sched_class = SCHEDULER_REGISTER.get(sched_name)
    sched_instance = sched_class.init_from_args(args)
    return sched_instance

def list_schedulers():
    return SCHEDULER_REGISTER.list_available()