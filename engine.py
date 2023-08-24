# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
 
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
from tqdm import tqdm
from PIL import Image,ImageDraw,ImageFont
import cv2
import json
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.open_world_eval import OWEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from util.plot_utils import plot_prediction
import matplotlib.pyplot as plt
from copy import deepcopy
from datasets.torchvision_datasets.open_world import Base_CLASS_NAME,VOC_COCO_CLASS_NAMES
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, nc_epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(samples, outputs, targets, epoch) ## samples variable needed for feature selection
        weight_dict = deepcopy(criterion.weight_dict)
        ## condition for starting nc loss computation after certain epoch so that the F_cls branch has the time
        ## to learn the within classes seperation.
        if epoch < nc_epoch: 
            for k,v in weight_dict.items():
                if 'NC' in k:
                    weight_dict[k] = 0
         
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        ## Just printing NOt affectin gin loss function
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
 
        loss_value = losses_reduced_scaled.item()
 
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
 
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
 
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
 
        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

## ORIGINAL FUNCTION
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = OWEvaluator(base_ds, iou_types,args=args)
 
    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
 
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
 
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
 
        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name
 
            panoptic_evaluator.update(res_pano)
 
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()
 
    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        res = coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['metrics'] = res
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
def submmit_format(results,image_info = None,viz = False):

    def box_xyxy_to_xywh(boxes):
        xmin,ymin,xmax,ymax = boxes
        b = [xmin,ymin,xmax-xmin,ymax-ymin]
        return b
    image_id = os.path.basename(image_info['targets'][0]['img_path']).split('.jpg')[0]
    targets = image_info['targets'][0]
    output_dir = image_info['output_dir']
    assert isinstance(results,list)
    results = results[0]
    if viz:
        image = cv2.imread(targets['img_path'])
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results_format = []
    h,w = targets['orig_size']
    results['boxes'][:,0::2].clamp_(min=0,max=w)
    results['boxes'][:,1::2].clamp_(min=0,max=h)
    
    labels = results['labels'].tolist()
    scores = results['scores'].tolist()
    boxes = results['boxes'].tolist()
    assert len(labels)==len(scores) and len(scores)== len(boxes)
    for label,score,box in zip(labels,scores,boxes):
        class_name = VOC_COCO_CLASS_NAMES[label]
        if viz:
            cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),2)
            # cv2.rectangle(image,(int(box[0]),int(box[1])),(min(int(box[0]+100),w),min(int(box[1]+50),h)),(255,255,255),thickness=-1)
            cv2.putText(image,
                        "{}:{:.2f}".format(class_name if class_name in Base_CLASS_NAME else "unknown",score),
                        (int(box[0]),int(box[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255,255,255),
                        2)
        result = {
            "image_id":int(image_id),
            "category_id":Base_CLASS_NAME.index(class_name)+1 if class_name in Base_CLASS_NAME else 8,
            "bbox":box_xyxy_to_xywh(box),
            "score":score
        }
        results_format.append(result)
    if viz:
        cv2.imwrite(os.path.join(output_dir,targets['img_path'].split('/')[-1]),image)
    return results_format


@torch.no_grad()
def inference(model, postprocessors, data_loader, device, output_dir,viz = False):
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    json_results = []
    for batch_indx, batch_input in tqdm(enumerate(data_loader),total=len(data_loader)):
        samples,targets = batch_input
        samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        orig_target_sizes = torch.stack([torch.tensor(t['orig_size'],device=device) for t in targets],dim=0)
        results = postprocessors['bbox'](outputs,orig_target_sizes)
        # assert data_loader.batch_size==1
        json_results.extend(
            submmit_format(
                        results,
                        image_info={"targets":targets,"output_dir":output_dir},
                        viz = viz
                    )
        )
    with open(os.path.join(output_dir,'inference.json'),'w') as fp:
        json.dump(json_results,fp)
        print('done!')
        
  
       

