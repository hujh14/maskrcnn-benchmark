import os
import sys
sys.path.append('../coco_utils')
import cv2
import argparse
import numpy as np
from tqdm import tqdm

import car_models
from render_car_instances_cv2 import CarPoseVisualizer
from coco_format import *

def make_apollo_annotations(data_dir, split, im_list):
    args = argparse.Namespace()
    args.data_dir = data_dir
    args.split = "train" # Both train and val splits are under train
    visualizer = CarPoseVisualizer(args)

    annotations = []
    imgId = 0

    print("Processing car masks...")
    for image_name in tqdm(im_list):
        image_name = image_name.replace(".jpg", "")
        imgId += 1

        # Get all segmentation masks
        all_masks = []
        car_pose_file = '%s/%s.json' % (visualizer._data_config['pose_dir'], image_name)
        car_poses = read_json(car_pose_file)
        image = visualizer.get_image(image_name)
        intrinsic = visualizer.get_intrinsic(image_name)

        car_poses.sort(key=visualizer.get_distance)
        claimed_mask = np.zeros((image.shape[0], image.shape[1]), dtype='uint8') # For handling occlusions
        for car_pose in car_poses:
            mask = visualizer.render_car(car_pose, image, intrinsic, fill=True)
            mask[claimed_mask != 0] = 0
            claimed_mask[mask != 0] = 255

            ann = make_ann(mask)
            ann["id"] = len(annotations) + 1
            ann["image_id"] = imgId
            ann["category_id"] = 1
            ann["car_pose"] = car_pose
            annotations.append(ann)
    return annotations

def make_apollo_categories():
    categories = []
    car = {}
    car["id"] = 1
    car["name"] = "car"
    categories.append(car)
    return categories

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=str, default="val")
    parser.add_argument('--data_dir', default='../../datasets/apollo', help='Data directory')
    args = parser.parse_args()
    print(args)

    im_dir = os.path.join(args.data_dir, "images/")
    ann_dir = os.path.join(args.data_dir, "annotations/")
    raw_dir = os.path.join(args.data_dir, "raw_data/")
    
    # Load image list
    if args.split == "train":
        im_list_fn = os.path.join(raw_dir, "train/split/train-list.txt")
    elif args.split == "val":
        im_list_fn = os.path.join(raw_dir, "train/split/validation-list.txt")
    elif args.split == "sample_data":
        raise Exception("Sample data not implemented")
    im_list = read_list(im_list_fn)

    annotations = make_apollo_annotations(raw_dir, args.split, im_list)
    categories = make_apollo_categories()
    images = make_images(im_list, im_dir)

    out_fn = os.path.join(ann_dir, "car_maskonly_{}.json".format(args.split))
    save_ann_fn(images, annotations, categories, out_fn, indent=None)
    print_ann_fn(out_fn)




    