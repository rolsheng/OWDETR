from pycocotools.coco import COCO
import os
annotation_file = 'CODA/val/annotations.json'
file  = COCO(annotation_file)
image_path = 'CODA/val/images'
save_path = 'gt'
os.makedirs(save_path,exist_ok=True)
