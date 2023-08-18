import os
import os.path as osp
from typing import Callable, Optional
from torchvision.datasets import VisionDataset
from open_world import VOC_COCO_CLASS_NAMES
from typing import Any
from PIL import Image
class RegionCodaDataset(VisionDataset):
  
    def __init__(self, 
                 root: str, 
                 transforms = None, 
                 transform =None,
                 target_transform = None):
        super(RegionCodaDataset,self).__init__(root, transforms, transform, target_transform)
        
        self.images = []
        self.labels = []
        self.class_names = []
        self.root = root
        dir_names = os.listdir(self.root)
        self.class_names = [dir_name.strip() for dir_name in dir_names]

        for cls_name in self.class_names:
            label = [cls_name]
            source_path = osp.join(self.root,cls_name)
            file_names = os.listdir(source_path)
            file_names = [osp.join(self.root,file_name) for file_name in file_names if file_name.endswith('.jpg')]
            self.images.extend(file_names)
            self.labels.extend(label*len(file_names))
        
    def __len__(self) -> int:
        return len(self.images)
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        w,h = img.size
        target = {
            'height':h,
            'width':w,
            'file_name':self.images[index],
            'label':VOC_COCO_CLASS_NAMES.index(self.labels[index])
        }
        if self.transforms is not None:
            img,target = self.transforms(img,target)
        return img,target


        
        
        
