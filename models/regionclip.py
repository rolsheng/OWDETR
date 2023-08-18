import torch
import torch.nn as nn
from collections import defaultdict
from datasets.torchvision_datasets.open_world import VOC_COCO_CLASS_NAMES
class RegionClip(nn.Module):
    def __init__(self,num_cls,clip_model) -> None:
        super(RegionClip,self).__init__()
        self.visual_encoder = clip_model.visual
        self._forward_saved_features = defaultdict(list)
        self.dtype = clip_model.dtype
    def forward(self,images,targets):
        features = self.visual_encoder(images.type(self.dtype))
        labels = torch.tensor([target['label'] for target in targets],dtype=torch.int64)
        
        


