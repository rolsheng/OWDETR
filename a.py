from pycocotools.coco import COCO
import os
import torch
from PIL import Image
import torchvision.transforms.functional as F
image = Image.open('helpers/labels/edge/helpers/images/0001.png')
image_tensor = F.to_tensor(image)
image_L = Image.open('helpers/labels/edge/helpers/images/0001.png').convert('L')
image_L_tensor = F.to_tensor(image_L)
print("OK")

model = torch.nn.Linear(100,10)
print(model.weight.requires_grad)
