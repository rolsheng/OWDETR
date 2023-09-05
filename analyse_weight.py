import torch
import torch.nn.functional as F
from datasets.torchvision_datasets.open_world import VOC_CLASS_NAMES,T2_CLASS_NAMES,T3_CLASS_NAMES,T4_CLASS_NAMES,VOC_COCO_CLASS_NAMES
import matplotlib.pyplot as plt

CLASS_NAME = ['sentry_box', 'machinery', 'motorcycle', 'suitcace', 'stroller', 'traffic_island', 
              'plastic_bag', 'garbage', 'traffic_light', 'cart', 'stone', 'debris', 'concrete_block', 
              'bicycle', 'dustbin', 'dog', 'moped', 'traffic_sign', 'bus', 'misc', 'tricycle', 
              'barrier', 'bollard', 'construction_vehicle', 'tram', 'traffic_cone', 'cyclist', 
              'pedestrian', 'truck', 'car']
CLASS_COUNT = [12, 13, 18, 21, 24, 29, 32, 68, 72, 85, 88, 94, 98, 122, 
               161, 270, 392, 445, 730, 787, 976, 1477, 1822, 2785, 3072, 
               4985, 11457, 13081, 16284, 59574]
model_path = 'exps_baseline/OWDETR_t4/checkpoint0159.pth'
state_dict = torch.load(model_path,map_location='cpu')['model']
# for layer in range(6):
layer = 5

plt.figure(figsize=(20,15))
weights = state_dict['class_embed.{}.weight'.format(layer)]
norm_weights = weights.norm(dim=-1,p=1)
x_axis = [name+":"+str(CLASS_COUNT[CLASS_NAME.index(name)]) for name in T4_CLASS_NAMES]
y_axis = [norm_weights[VOC_COCO_CLASS_NAMES.index(namecount.split(':')[0])] for idx,namecount in enumerate(x_axis)]
plt.plot(x_axis,y_axis)
plt.savefig("{}fc_weight_norm.jpg".format(model_path.split('checkpoint')[0]))

    
    