import os
import sys
sys.path.append('../coco_utils')
import argparse
from tqdm import tqdm

import car_models
import utils.utils as uts
from render_car_instances_cv2 import CarPoseVisualizer

from pycocotools.coco import COCO
from coco_format import *

args = argparse.Namespace()
args.data_dir = "../../datasets/apollo/raw_data/"
args.split = "train" # Both train and val splits are under train
visualizer = CarPoseVisualizer(args)

def vis_car_pose(img, car_pose, fill=False):
    h, w = img.shape[:2]
    intrinsic = visualizer.get_intrinsic()
    intrinsic = uts.intrinsic_vec_to_mat(intrinsic, (h, w))

    car = visualizer.get_car(car_pose['car_id'])
    pose = np.array(car_pose["pose"])

    # project 3D points to 2d image plane
    rmat = uts.euler_angles_to_rotation_matrix(pose[:3])
    rvect, _ = cv2.Rodrigues(rmat)
    imgpts, jac = cv2.projectPoints(np.float32(car['vertices']), rvect, pose[3:], intrinsic, distCoeffs=None)

    mask = np.zeros((h,w), dtype='uint8')
    for face in car['faces'] - 1:
        pts = np.array([[imgpts[idx, 0, 0], imgpts[idx, 0, 1]] for idx in face], np.int32)
        pts = pts.reshape((-1, 1, 2))
        if fill:
            cv2.fillPoly(mask, [pts], 255)
        else:
            cv2.polylines(mask, [pts], True, 255, thickness=2)
    return mask

def vis_annotations(img, anns, coco):
    for ann in anns:
        car_pose = ann["car_pose"]

        roll, pitch, yaw, x, y ,z = car_pose["pose"]
        masks = []
        for d in np.linspace(0,5.5,7):
            car_pose["pose"] = [roll, pitch + d, yaw, x, y ,z]

            mask = vis_car_pose(img, car_pose)
            mask = vis_mask(mask, ann)
            masks.append(mask)

        masks = np.concatenate(masks, axis=1)
        cv2.imwrite("temp.jpg", masks)
        raise


def vis_mask(mask, ann):
    bbox = np.round(ann["bbox"])
    d = bbox[3] * 0.5
    x = max(0, int(bbox[0] - d))
    y = max(0, int(bbox[1] - d))
    w = int(bbox[2] + 2*d)
    h = int(bbox[3] + 2*d)
    mask = mask[y:y+h, x:x+w]

    mask = cv2.resize(mask, (300, 200))
    # cv2.imshow("", img)
    # cv2.waitKey(0)
    return mask

def vis_coco(coco, im_dir, out_dir):
    for imgId in tqdm(coco.imgs):
        im = coco.imgs[imgId]

        # Load image
        img_fn = os.path.join(im_dir, im["file_name"])
        img = cv2.imread(img_fn)
        if img is None:
            print("Warning: Could not find ", img_fn)
            img = np.zeros((im["height"], im["width"], 3))

        # Visualize annotations
        annIds = coco.getAnnIds(imgIds=[imgId])
        anns = coco.loadAnns(annIds)
        img = vis_annotations(img, anns, coco)

        # Save image
        # out_fn = os.path.join(out_dir, im["file_name"])
        # if not os.path.exists(os.path.dirname(out_fn)):
        #     os.makedirs(os.path.dirname(out_fn))
        # cv2.imwrite(out_fn, img)


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
    vis_coco(coco, args.im_dir, args.out_dir)


    # images = coco.dataset["images"]
    # annotations = coco.dataset["annotations"]
    # categories = coco.dataset["categories"]
    # save_ann_fn(images, annotations, categories, out_fn)
    # print_ann_fn(out_fn)
