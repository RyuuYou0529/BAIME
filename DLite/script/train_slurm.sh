JOB_NAME=DLite_SLURM
CONFIG_NAME=DLite.yaml
SLURM_PARTITION=compute
SLURM_NODE=c002
NUM_CPUS=4
NUM_GPUS=2

SCRIPT_DIR=$(dirname $(realpath $0))
WORKSPACE=$(realpath $SCRIPT_DIR/../)
export PYTHONPATH=$WORKSPACE
CONFIG_PATH=$WORKSPACE/config/$CONFIG_NAME
COMMAND="python -m lib.train -cfg $CONFIG_PATH -slurm"

python -m lib.utils.slurm.slurm_launch \
       --job-name $JOB_NAME \
       --command "$COMMAND" \
       --config $CONFIG_PATH \
       --partition $SLURM_PARTITION \
       --node $SLURM_NODE \
       --num-cpus $NUM_CPUS \
       --num-gpus $NUM_GPUS