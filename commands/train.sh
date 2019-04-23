
NGPUS=4
#GPU_IDS=4,5,6,7
MASKRCNN_DIR=/data/vision/oliva/scenedataset/scaleplaces/active_projects/maskrcnn-benchmark

CONFIG_FILE=$MASKRCNN_DIR/configs/apollo/keypoint.yaml
TRAIN="keypoints_apollo_train"
TEST="keypoints_apollo_val"
OUTPUT_DIR=$MASKRCNN_DIR/output/apolloscape
OPTS="--config-file $CONFIG_FILE OUTPUT_DIR $OUTPUT_DIR DATASETS.TRAIN (\"$TRAIN\",) DATASETS.TEST (\"$TEST\",)"

# Training
python $MASKRCNN_DIR/tools/train_net.py $OPTS SOLVER.IMS_PER_BATCH 2
#CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NGPUS $MASKRCNN_DIR/tools/train_net.py $OPTS

# Inference
#python $MASKRCNN_DIR/tools/test_net.py $OPTS

