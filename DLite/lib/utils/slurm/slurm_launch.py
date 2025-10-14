# slurm-launch.py
# Usage:
# python slurm-launch.py --exp-name test \
#     --command "rllib train --run PPO --env CartPole-v0"

import argparse
import subprocess
import sys
import time
from pathlib import Path

from ..file import read_cfg, parse_trial_dir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-name",
        type=str,
        required=True,
        help="The job name and path to logging file (exp_name.log).",
    )
    parser.add_argument(
        "--num-nodes", "-n", type=int, default=1, help="Number of nodes to use."
    )
    parser.add_argument(
        "--node",
        "-w",
        type=str,
        help="The specified nodes to use. Same format as the "
        "return of 'sinfo'. Default: ''.",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=1,
        help="Number of CPUs to use in each node. (Default: 1)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs to use in each node. (Default: 0)",
    )
    parser.add_argument(
        "--partition",
        "-p",
        type=str,
    )
    parser.add_argument(
        "--load-env",
        type=str,
        help="The script to load your environment ('module load cuda/10.1')",
        default="",
    )
    parser.add_argument(
        "--command",
        type=str,
        required=True,
        help="The command you wish to execute. For example: "
        " --command 'python test.py'. "
        "Note that the command must be a string.",
    )
    parser.add_argument(
        "--config",
        "-cfg",
        type=str,
        required=True,
        help="The config file for the experiment.",
    )
    args = parser.parse_args()
    return args

def build_sbatch_script(args, cfg_args):
    template_file = Path(__file__).parent / "slurm_template.sh"
    slurm_path = Path(cfg_args.out_slurm_path)

    JOB_NAME = "${JOB_NAME}"
    LOG_PATH = "${LOG_PATH}"
    NUM_NODES = "${NUM_NODES}"
    NUM_CPUS_PER_NODE = "${NUM_CPUS_PER_NODE}"
    NUM_GPUS_PER_NODE = "${NUM_GPUS_PER_NODE}"
    PARTITION_OPTION = "${PARTITION_OPTION}"
    COMMAND_PLACEHOLDER = "${COMMAND_PLACEHOLDER}"
    GIVEN_NODE = "${GIVEN_NODE}"
    LOAD_ENV = "${LOAD_ENV}"
    
    if args.node:
        # assert args.num_nodes == 1
        node_info = "#SBATCH -w {}".format(args.node)
    else:
        node_info = ""

    job_name = "{}_{}".format(
        args.job_name, time.strftime("%m%d-%H%M", time.localtime())
    )

    partition_option = (
        "#SBATCH --partition={}".format(args.partition) if args.partition else ""
    )

    # ===== Modified the template script =====
    with open(template_file, "r") as f:
        text = f.read()
    text = text.replace(JOB_NAME, job_name)
    text = text.replace(LOG_PATH, str(slurm_path / f"{job_name}"))
    text = text.replace(NUM_NODES, str(args.num_nodes))
    text = text.replace(NUM_CPUS_PER_NODE, str(args.num_cpus))
    text = text.replace(NUM_GPUS_PER_NODE, str(args.num_gpus))
    text = text.replace(PARTITION_OPTION, partition_option)
    text = text.replace(COMMAND_PLACEHOLDER, str(args.command))
    text = text.replace(LOAD_ENV, str(args.load_env))
    text = text.replace(GIVEN_NODE, node_info)
    text = text.replace(
        "# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO PRODUCTION!",
        "# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE "
        "RUNNABLE!",
    )

    # ===== Save the script =====
    script_file = slurm_path / "{}.sh".format(job_name)
    with open(script_file, "w") as f:
        f.write(text)

    # ===== Submit the job =====
    print("Starting to submit job...")
    subprocess.Popen(["sbatch", script_file])
    print("Job submitted!")
    print(f"Script and Log path: {slurm_path}")
    sys.exit(0)

def main():
    args = parse_args()
    cfg_args = read_cfg(argparse.Namespace(cfg=args.config, slurm=True))
    parse_trial_dir(cfg_args, True)
    build_sbatch_script(args, cfg_args)

if __name__ == "__main__":
    main()
    