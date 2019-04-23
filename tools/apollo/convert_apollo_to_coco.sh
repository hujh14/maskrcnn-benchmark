
python convert_apollo_mask.py -s val
python convert_apollo_keypoints.py -s val
python associate_masks_and_keypoints.py -s val

python convert_apollo_mask.py -s train
python convert_apollo_keypoints.py -s train
python associate_masks_and_keypoints.py -s train
