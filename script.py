import torch
import os
import shutil
import json
#提交到dev分支
os.makedirs('SODA10M/labeled/new_train',exist_ok=True)
os.makedirs('SODA10M/labeled/new_val',exist_ok=True)
for file_name in os.listdir('SODA10M/labeled/train'):
    src = os.path.join('SODA10M/labeled/train',file_name)
    new_name = '8'+file_name[9:15]+'.jpg'
    dst = os.path.join('SODA10M/labeled/new_train',new_name)
    shutil.copy(src,dst)
with open('SODA10M/labeled/annotations/instance_train.json','r') as fp:
    json_file = json.load(fp)
    images = []
    for img_info in json_file['images']:
        img_info['file_name'] = '8'+img_info['file_name'][9:15]+'.jpg'
        images.append(img_info)
    json_file['images'] = images
    with open("SODA10M/labeled/annotations/instance_train_md.json",'w') as fp:
        json.dump(json_file,fp)
for file_name in os.listdir('SODA10M/labeled/val'):
    src = os.path.join('SODA10M/labeled/val',file_name)
    new_name = '9'+file_name[7:13]+'.jpg'
    dst = os.path.join('SODA10M/labeled/new_val',new_name)
    shutil.copy(src,dst)
with open('SODA10M/labeled/annotations/instance_val.json','r') as fp:
    json_file = json.load(fp)
    images = []
    for img_info in json_file['images']:
        img_info['file_name'] = '9'+img_info['file_name'][7:13]+'.jpg'
        images.append(img_info)
    json_file['images'] = images
    with open("SODA10M/labeled/annotations/instance_val_md.json",'w') as fp:
        json.dump(json_file,fp)