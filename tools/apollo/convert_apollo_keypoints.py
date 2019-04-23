import os
import sys
sys.path.append('../coco_utils')
import argparse
import numpy as np
from tqdm import tqdm

from coco_format import *

def make_apollo_keypoints(data_dir, split, im_list):
    annotations = []
    imgId = 0

    print("Processing car keypoints...")
    for image_name in tqdm(im_list):
        image_name = image_name.replace(".jpg", "")
        imgId += 1

        # Get all car keypoints
        all_keypoints = []
        kp_dir = os.path.join(data_dir, "train/keypoints", image_name)
        for fn in os.listdir(kp_dir):
            kp_list = read_list(os.path.join(kp_dir, fn))
            kps = {}
            for entry in kp_list:
                kp, x, y = entry.split("\t")
                kps[int(kp)] = (float(x), float(y))
            all_keypoints.append(kps)

        # Make annotations
        for kps in all_keypoints:
            a = np.zeros((66,3))
            for kp in kps:
                a[kp-1] = [kps[kp][0], kps[kp][1], 1]

            ann = {}
            ann["id"] = len(annotations) + 1
            ann["image_id"] = imgId
            ann["category_id"] = 1
            ann["num_keypoints"] = len(kps)
            ann["keypoints"] = a.flatten().tolist()
            annotations.append(ann)
    return annotations

def make_apollo_categories():
    categories = []
    car = {}
    car["id"] = 1
    car["name"] = "car"
    car["keypoints"] = [str(i) for i in range(66)]
    car["skeleton"] = []
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

    annotations = make_apollo_keypoints(raw_dir, args.split, im_list)
    categories = make_apollo_categories()
    images = make_images(im_list, im_dir)

    out_fn = os.path.join(ann_dir, "car_keypointsonly_{}.json".format(args.split))
    save_ann_fn(images, annotations, categories, out_fn, indent=None)
    print_ann_fn(out_fn)




    