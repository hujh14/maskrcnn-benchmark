

# Set NGPUS with first arg. Default 1.
NGPUS=${1:-"1"}

MASKRCNN_DIR="."
CONFIG_FILE="$MASKRCNN_DIR/configs/e2e_keypoint_rcnn_R_50_FPN_1x.yaml"
OUTPUT_DIR="$MASKRCNN_DIR/output/coco"
OPTS="--config-file $CONFIG_FILE OUTPUT_DIR $OUTPUT_DIR"

# Weights
OPTS="$OPTS MODEL.WEIGHT models/model_final.pkl"

# Inference
if [ "$NGPUS" = "1" ] ; then
	python $MASKRCNN_DIR/tools/test_net.py $OPTS 
else
	python -m torch.distributed.launch --nproc_per_node=$NGPUS $MASKRCNN_DIR/tools/test_net.py $OPTS
fi
