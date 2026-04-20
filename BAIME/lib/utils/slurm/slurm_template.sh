#!/bin/bash
# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO PRODUCTION!
${PARTITION_OPTION}
${GIVEN_NODE}

#SBATCH --nodes=${NUM_NODES}
#SBATCH --ntasks-per-node=${NUM_GPUS_PER_NODE}
#SBATCH --cpus-per-task=${NUM_CPUS_PER_NODE}
#SBATCH --gres=gpu:${NUM_GPUS_PER_NODE}

#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_PATH}_out.log
#SBATCH --error=${LOG_PATH}_err.log

# Load modules or your own conda environment here
# module load cuda/12.1
# conda activate ${CONDA_ENV}
${LOAD_ENV}

# ===== DDP Environment Setup =====
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NUM_NODES=$SLURM_NNODES"
echo "GPUS_PER_NODE=${NUM_GPUS_PER_NODE}"

# ===== Run Training =====
srun ${COMMAND_PLACEHOLDER}

exit
