import os
import sys
sys.path.append('../coco_utils')
import argparse
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask

from coco_format import *

def associate_annotations(coco_mask, coco_kps):
    annotations = []
    print("Associating masks and keypoints...")
    for imgId in tqdm(coco_mask.imgs):
        maskIds = coco_mask.getAnnIds(imgIds=[imgId])
        masks = coco_mask.loadAnns(maskIds)
        kpsIds = coco_kps.getAnnIds(imgIds=[imgId])
        kps = coco_kps.loadAnns(kpsIds)

        anns_associated = assocatiate_masks_and_keypoints(masks, kps)
        annotations.extend(anns_associated)
    return annotations

def assocatiate_masks_and_keypoints(masks, kps):
    annotations = []

    for mask_ann in masks:
        if "segmentation" not in mask_ann:
            print(mask_ann)
        mask = COCOmask.decode(mask_ann["segmentation"])

        best_num_matching_kps = 0
        best_match = None
        for keyp_ann in kps:
            if mask_ann["category_id"] != keyp_ann["category_id"]:
                continue

            # Count number of keypoints to lie in the mask
            num_matching_kps = 0
            keypoints = np.array(keyp_ann["keypoints"]).reshape(-1, 3)
            for x,y,v in keypoints:
                if v != 0 and mask[int(y), int(x)] != 0:
                    num_matching_kps += 1

            if num_matching_kps > best_num_matching_kps:
                best_num_matching_kps = num_matching_kps
                best_match = keyp_ann
        
        if best_num_matching_kps > 0:
            mask_ann["keypoints"] = best_match["keypoints"]
            mask_ann["num_keypoints"] = best_match["num_keypoints"]
            kps.remove(best_match)
            annotations.append(mask_ann)
    return annotations

def print_annotation_types(anns):
    skp_anns = [ann for ann in anns if "keypoints" in ann and "segmentation" in ann]
    s_anns = [ann for ann in anns if "keypoints" not in ann and "segmentation" in ann]
    kp_anns = [ann for ann in anns if "keypoints" in ann and "segmentation" not in ann]
    print("{} Mask+KP, {} Mask only, {} KP only annotations".format(len(skp_anns), len(s_anns), len(kp_anns)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=str, default="val")
    parser.add_argument('--data_dir', default='../../datasets/apollo', help='Data directory')
    args = parser.parse_args()
    print(args)

    ann_dir = os.path.join(args.data_dir, "annotations/")
    maskonly_fn = os.path.join(ann_dir, "car_maskonly_{}.json".format(args.split))
    kpsonly_fn = os.path.join(ann_dir, "car_keypointsonly_{}.json".format(args.split))
    out_fn = os.path.join(ann_dir, "car_keypoints_{}.json".format(args.split))

    coco_mask = COCO(maskonly_fn)
    coco_kps = COCO(kpsonly_fn)
    print_annotation_types(coco_mask.dataset["annotations"])
    print_annotation_types(coco_kps.dataset["annotations"])

    annotations = associate_annotations(coco_mask, coco_kps)
    print_annotation_types(annotations)

    images = coco_mask.dataset["images"]
    categories = coco_kps.dataset["categories"]
    save_ann_fn(images, annotations, categories, out_fn)
    print_ann_fn(out_fn)
