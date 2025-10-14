import os
import sys
import shutil
import yaml

def read_cfg(args):
    # === Read CFG File === #
    if args.cfg:
        with open(args.cfg, 'r') as f:
            yml = yaml.safe_load(f)
        cmd = [c[1:] for c in sys.argv if c[0] == '-']
        for k, v in yml.items():
            if k not in cmd:
                args.__dict__[k] = v
    return args

def checkdir(path, reset=True):
    if os.path.exists(path):
        if reset:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def parse_trial_dir(args, check=True):
    args.out_trial_path = os.path.join(args.out_path, args.model)
    args.out_tb_path = os.path.join(args.out_trial_path, 'tensorboard')
    args.out_ckpt_path = os.path.join(args.out_trial_path, 'checkpoints')
    args.out_log_path = os.path.join(args.out_trial_path, 'logs')
    if args.slurm:
        args.out_slurm_path = os.path.join(args.out_trial_path, 'slurm')
    if check:
        checkdir(args.out_trial_path, reset=args.reset)
        checkdir(args.out_tb_path, reset=args.reset)
        checkdir(args.out_ckpt_path, reset=args.reset)
        checkdir(args.out_log_path, reset=args.reset)
        if args.slurm:
            checkdir(args.out_slurm_path, reset=args.reset)
    return args


class Tee(object):
    def __init__(self, file_path:str, keep_stdout:bool=True, keep_stderr:bool=False):
        file = open(file_path, 'a')
        self.files = [file]
        if keep_stdout:
            self.files.append(sys.__stdout__)
        if keep_stderr:
            self.files.append(sys.__stderr__)
    
    def write(self, message):
        for f in self.files:
            f.write(message)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


