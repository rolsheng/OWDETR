import torch
import torch.nn as nn
from collections import defaultdict
from datasets.torchvision_datasets.open_world import VOC_COCO_CLASS_NAMES
import torch.nn.functional as F
class RegionClip(nn.Module):
    def __init__(self,num_cls,clip_model,device,scale=1.0,norm=False):
        super(RegionClip,self).__init__()
        self.visual_encoder = clip_model.visual
        self._forward_saved_features = {
            "gt_labels":[],
            "region_features":[],
        }
        self.dtype = clip_model.dtype
        self.num_cls = num_cls
        self.device = device
        self.scale = scale
        self.norm = norm
    def save_feature(self,features,labels):
        assert len(features)==len(labels),"The shape of feature and labels mismatch"
        self._forward_saved_features['gt_labels'].append(labels)
        self._forward_saved_features['region_features'].append(features)
        return {"gt_labels":labels,"region_features":features}    
        
    def forward(self,images,targets):
        images = images.to(self.device)
        features = self.scale*self.visual_encoder(images.type(self.dtype))
        if self.norm:
            features = F.normalize(features,p=2,dim=-1)
        labels = torch.tensor([target['label'] for target in targets],dtype=torch.int64,device=self.device)
        results = self.save_feature(features=features.detach().cpu(),labels=labels.detach().cpu())
     
        return results

        
        
def build_Regioncip(num_cls,clip_model,device):
    return RegionClip(num_cls,clip_model,device)
        


