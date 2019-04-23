
NGPUS=4

MASKRCNN_DIR="."
CONFIG_FILE="$MASKRCNN_DIR/configs/apollo/e2e_keypoint_rcnn_R_50_FPN_1x.yaml"
OUTPUT_DIR="$MASKRCNN_DIR/output/apollo"
OPTS="--config-file $CONFIG_FILE OUTPUT_DIR $OUTPUT_DIR"

# Datasets
# TRAIN="keypoints_apollo_train"
# TEST="keypoints_apollo_val"
# OPTS="$OPTS DATASETS.TRAIN (\"$TRAIN\",) DATASETS.TEST (\"$TEST\",)"

# Training schedule
if [ "$NGPUS" = "1" ] ; then
	MORE_OPTS="SOLVER.IMS_PER_BATCH 2"
	OPTS="$OPTS $MORE_OPTS"
elif [ "$NGPUS" = "4" ] ; then
	MORE_OPTS="SOLVER.BASE_LR 0.01 SOLVER.STEPS (120000, 160000) SOLVER.MAX_ITER: 180000 SOLVER.IMS_PER_BATCH: 8"
	OPTS="$OPTS $MORE_OPTS"
fi

# Training
if [ "$NGPUS" = "1" ] ; then
	python $MASKRCNN_DIR/tools/train_net.py $OPTS 
else
	python -m torch.distributed.launch --nproc_per_node=$NGPUS $MASKRCNN_DIR/tools/train_net.py $OPTS
fi

# Inference
#python $MASKRCNN_DIR/tools/test_net.py $OPTS

