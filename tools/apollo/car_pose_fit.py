import os
import sys
sys.path.append('../coco_utils')
import argparse
from tqdm import tqdm

import utils.utils as uts
from render_car_instances_cv2 import CarPoseVisualizer

from pycocotools.coco import COCO
from coco_format import *

args = argparse.Namespace()
args.data_dir = "../../datasets/apollo/raw_data/"
args.split = "train" # Both train and val splits are under train
visualizer = CarPoseVisualizer(args)


def fit_car_pose(coco, ann):
    img = coco.imgs[ann["image_id"]]
    h = img["height"]
    w = img["width"]
    keypoints = np.array(ann["keypoints"]).reshape(-1, 3)

    intrinsic = visualizer.get_intrinsic()
    intrinsic = uts.intrinsic_vec_to_mat(intrinsic, (h, w))

    car = visualizer.get_car(20)
    vertices = car["vertices"]
    faces = car["faces"]
    pose = ann["car_pose"]["pose"]

    # cv2.solvePnP(objectPoints, imagePoints, intrinsic)

def vis_car(car, pose):
    # project 3D points to 2d image plane
    rmat = uts.euler_angles_to_rotation_matrix(pose[:3])
    rvect, _ = cv2.Rodrigues(rmat)
    imgpts, jac = cv2.projectPoints(np.float32(car['vertices']), rvect, pose[3:], intrinsic, distCoeffs=None)
    






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str, required=True, help='Annotation file')
    parser.add_argument('-d', '--im_dir', type=str, required=True, help='Images directory')
    parser.add_argument('-o', '--out_dir', type=str, default=None, help='Output visualization directory')
    args = parser.parse_args()
    if args.out_dir == None:
        args.out_dir = args.ann_fn.replace(".json", "")
    print(args)

    coco = COCO(args.ann_fn)
    ann = coco.dataset["annotations"][0]
    fit_car_pose(coco, ann)

    # images = coco.dataset["images"]
    # annotations = coco.dataset["annotations"]
    # categories = coco.dataset["categories"]
    # save_ann_fn(images, annotations, categories, out_fn)
    # print_ann_fn(out_fn)
