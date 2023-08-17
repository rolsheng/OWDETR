import os
import xml.etree.ElementTree as ET 
import random
from collections import defaultdict
random.seed(42)
is_finetuning = True #
train_txts = [
    't1_train.txt',
    't2_train.txt',
    't3_train.txt',
    't4_train.txt'
    ]
categories = [
            'pedestrian','cyclist','dog','misc',
            'car','truck','tram','tricycle','bus','bicycle','moped','motorcycle','stroller','cart','construction_vehicle',
            'barrier','bollard','sentry_box','traffic_cone','traffic_island','traffic_light','traffic_sign',
            'debris', 'suitcace', 'dustbin', 'concrete_block', 'machinery', 'garbage','plastic_bag','stone'
            ]

if is_finetuning:
    cat_2_img = defaultdict(list)
    txt_list = []
    for train_split in train_txts:
        file_path = os.path.join("data/OWDETR/VOC2007/ImageSets/Main",train_split)
        with open(file_path,'r') as txt:
            # txt_list.extend(txt.readlines())
            for img in txt.readlines():
                anno = ET.parse(os.path.join("data/OWDETR/VOC2007/Annotations",img.strip()+'.xml')).getroot()
                for object in anno.iter('object'):
                    name = object.find('name').text.lower()
                    if name in categories:
                        cat_2_img[name].append(img.strip())
    num_exemplar = 10e10
    for cat_name,img_list in cat_2_img.items():
        # img_list = list(set(img_list))
        if len(img_list)<num_exemplar:
            num_exemplar = len(img_list)
    for cat_name,img_list in cat_2_img.items():
        txt_list.extend(random.sample(img_list,k=num_exemplar))
    txt_list = list(set(txt_list))
    with open('data/OWDETR/VOC2007/ImageSets/Main/t4_ft.txt','w') as txt:
        for img in txt_list:
            txt.write(img)
            txt.write("\n")

else:
    with open("data/OWDETR/VOC2007/ImageSets/Main/train.txt",'r') as txt:
        train_list = txt.readlines()
    with open("data/OWDETR/VOC2007/ImageSets/Main/t4_train.txt",'w') as txt:
        for xml_file in train_list:
            xml_file = xml_file.strip()
            anno = ET.parse(os.path.join("data/OWDETR/VOC2007/Annotations",xml_file+".xml")).getroot()
            for object in anno.iter("object"):
                name = object.find("name").text.lower()
                if name in categories:
                    txt.write(xml_file.split('.')[0])
                    txt.write("\n")
                    break



