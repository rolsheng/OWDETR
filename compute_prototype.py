from clip import clip
from models.regionclip import build_Regioncip
from datasets.torchvision_datasets.open_world import VOC_COCO_CLASS_NAMES
from datasets.torchvision_datasets.region_coda import build_datasets
from torch.utils.data import DataLoader
from collections import defaultdict
import torch
import os
import torch.nn as nn
import util.misc as utils
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from datasets.coco import make_region_coda_transforms
def load_clip_model(model_name,device):
    url = clip._MODELS[model_name]
    model_path = clip._download(url)

    try:
        clip_model = torch.jit.load(model_path,map_location='cpu').eval()
        state_dict = None
    except RuntimeError as e:
        state_dict = torch.load(model_path,map_location='cpu')
    
    clip_model = clip.build_model(state_dict or clip_model.state_dict())
    return clip_model.to(device=device)

@torch.no_grad()
def evaluator(model,dataloader):
    model.eval()
    #init bank
    num_cls = model.num_cls
    class_prototype = {}

    all_res = []
    for batch_indx,batch in tqdm(enumerate(dataloader),total=len(dataloader)):
        imgs,labels = batch[0].tensors,batch[1]
        res = model(imgs,labels)
        all_res.append(res)
    saved_features = model._forward_saved_features['region_features']
    saved_labels = model._forward_saved_features['gt_labels']

    region_features = torch.cat(saved_features)
    gt_labels = torch.cat(saved_labels)
    torch.save(region_features,'output_prototype/region_features.pth')
    torch.save(gt_labels,'output_prototype/gt_labels.pth')
    
    for label_id in range(num_cls):
        class_prototype[label_id] = torch.mean(region_features[gt_labels==label_id],dim=0)
    torch.save(class_prototype,"output_prototype/prototype.pth")
    
    return class_prototype
        
def main():
    #basic config
    os.makedirs('output_prototype',exist_ok=True)
    batch_size = 32
    num_workers = 4
    #model config
    clip_model_name = 'RN50'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model = load_clip_model(clip_model_name,device=device)
    num_cls  = len(VOC_COCO_CLASS_NAMES)-1
    print(VOC_COCO_CLASS_NAMES[:num_cls])
    print(num_cls)
    model = build_Regioncip(num_cls=num_cls,clip_model=clip_model,device=device)
    #dataset config
    data_root = 'data/OWDETR/object_crops'
    data_transforms = make_region_coda_transforms()
    dataset = build_datasets(root=data_root,transforms=data_transforms)
    #dataloader config
    sampler_train = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = torch.utils.data.BatchSampler(sampler_train, batch_size, drop_last=False)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler,
                                   collate_fn=utils.collate_fn, num_workers=num_workers,
                                   pin_memory=True)
    evaluator(model=model,dataloader=dataloader)
    

    


    
    
if __name__ =="__main__":
    main()