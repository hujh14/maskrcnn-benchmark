import os
import sys
sys.path.append("../coco_utils")
import argparse
from tqdm import tqdm

from pycocotools.coco import COCO

from coco_format import *

def make_annotations_from_detections(bboxes, segms, keypoints):
    annotations = []
    for bbox_ann, keyp_ann in zip(bboxes, keypoints):
        assert bbox_ann["image_id"] == keyp_ann["image_id"]
        assert bbox_ann["category_id"] == keyp_ann["category_id"]
        assert bbox_ann["score"] == keyp_ann["score"]

        ann = {}
        ann["id"] = len(annotations) + 1
        ann["image_id"] = bbox_ann["image_id"]
        ann["category_id"] = bbox_ann["category_id"]
        ann["score"] = bbox_ann["score"]
        ann["bbox"] = bbox_ann["bbox"]
        ann["keypoints"] = keyp_ann["keypoints"]
        ann["iscrowd"] = 0
        annotations.append(ann)
    return annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str, required=True, help='Annotation file')
    parser.add_argument('-o', '--out_dir', type=str, help='Inference directory')
    args = parser.parse_args()
    print(args)

    keypoints_fn = os.path.join(args.out_dir, "keypoints.json")
    bbox_fn = os.path.join(args.out_dir, "bbox.json")
    out_fn = os.path.join(args.out_dir, "predictions.json")

    keypoints = read_json(keypoints_fn)
    bboxes = read_json(bbox_fn)
    annotations = make_annotations_from_detections(bboxes, None, keypoints)

    coco = COCO(args.ann_fn)
    images = coco.dataset["images"]
    annotations = annotations
    categories = coco.dataset["categories"]
    save_ann_fn(images, annotations, categories, out_fn)
    print_ann_fn(out_fn)
