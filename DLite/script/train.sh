CONFIG_NAME=DLite.yaml

SCRIPT_DIR=$(dirname $(realpath $0))
WORKSPACE=$(realpath $SCRIPT_DIR/../)
export PYTHONPATH=$WORKSPACE

python -m lib.train -cfg $WORKSPACE/config/$CONFIG_NAME