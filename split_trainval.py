import os,json
import random
import torch
import numpy as np
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
xml_list = os.listdir('data/OWDETR/VOC2007/Annotations')
num_xml = len(xml_list)
random.shuffle(xml_list)
test_img = random.sample(xml_list,k=int(num_xml*0.2))
test_txt = []
train_txt = []
for img in xml_list:
    if img in test_img:
        test_txt.append(img.split('.')[0])
    else:
        train_txt.append(img.split('.')[0])
print(len(train_txt),len(test_txt))
with open("data/OWDETR/VOC2007/ImageSets/Main/train.txt",'w') as txt:
    for file_name in sorted(train_txt):
        txt.write(file_name)
        txt.write("\n")
with open("data/OWDETR/VOC2007/ImageSets/Main/test.txt",'w') as txt:
    for file_name in sorted(test_txt):
        txt.write(file_name)
        txt.write("\n")

