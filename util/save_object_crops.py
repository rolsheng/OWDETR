import json
import os
import argparse
from PIL import Image
from tqdm import tqdm
import xml.etree.ElementTree as ET
import itertools
#0+4
VOC_CLASS_NAMES = [
    'pedestrian','cyclist','dog','misc'
]
#4+11
T2_CLASS_NAMES = [
    'car','truck','tram','tricycle','bus','bicycle','moped',
    'motorcycle','stroller','cart','construction_vehicle'
]
#15+7
T3_CLASS_NAMES = [
    'barrier','bollard','sentry_box','traffic_cone',
    'traffic_island','traffic_light','traffic_sign'
]
#22+8
T4_CLASS_NAMES = [
   'debris', 'suitcace', 'dustbin', 'concrete_block', 
   'machinery', 'garbage','plastic_bag','stone'
]

UNK_CLASS = ["unknown"]

VOC_COCO_CLASS_NAMES = tuple(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))
print(len(VOC_COCO_CLASS_NAMES))
def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate the object crops from Coda')
    parser.add_argument('--anno-path', default="data/OWDETR/VOC2007/Annotations",help='path of the xml-format annotation file')
    parser.add_argument('--img-root', default="data/OWDETR/VOC2007/JPEGImages",help='path of the image')
    parser.add_argument('--img-set',default='train.txt')
    parser.add_argument('--save-root', default="data/OWDETR/object_crops",help='path to save the object crops')
    parser.add_argument('--scaling-factor', type=float, default=0.2,
                        help='factor for extending the object box')
    parser.add_argument('--min-size', type=float, default=20.0,
                        help='the min size of the box')
    args = parser.parse_args()
    return args

def parse_xml(xml_path):
    tree = ET.parse(xml_path).getroot()
    #parse bbox annotations
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    #parse image info 
    image_info = {}
    image_info['file_name'] = tree.find('filename').text
    width = tree.find('size').find('width').text
    height = tree.find('size').find('height').text
    image_info['size'] = (width,height)
    
    return image_info,objects


    
def cropping_fun(image_anns, image_info,img_root, save_root, scaling_factor, min_size):
    for i in range(len(image_anns)):
        image_ann = image_anns[i]
    
        cat_name = image_ann['name']
        img_name = image_info['file_name']
        img_path = os.path.join(img_root, img_name)
        save_cat_root = os.path.join(save_root, cat_name)
        save_path = os.path.join(save_cat_root, str(i) + '_' + img_name)

        xmin, ymin, xmax, ymax = image_ann['bbox']
        w,h = xmax-xmin,ymax-ymin
        if w * h < min_size:
            print(f"skip {save_path.split('/')[-1]} with size {w * h}!")
            continue

        image = Image.open(img_path)
        xmin = max(0, int(xmin - scaling_factor * w))
        ymin = max(0, int(ymin - scaling_factor * h))
        xmax = min(image.size[0], int(xmin + (1 + scaling_factor) * w))
        ymax = min(image.size[1], int(ymin + (1 + scaling_factor) * h))
        image = image.crop([xmin, ymin, xmax, ymax])

        os.makedirs(save_cat_root, exist_ok=True)
        try:
            image.save(save_path)
        except:
            print(f"{save_path.split('/')[-1]} can not be saved!")

    return True


def main():
    args = parse_args()

    anno_path = args.anno_path
    img_root = args.img_root
    save_root = args.save_root
    scaling_factor = args.scaling_factor
    min_size = args.min_size
    img_set = os.path.join('data/OWDETR/VOC2007/ImageSets/Main',args.img_set)
    img_sets = []
    with open(img_set,'r') as txt:
        img_sets = [line.strip() for line in txt.readlines()]
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    image_infos = {}
    image_anns = {}
    image_cats = VOC_COCO_CLASS_NAMES
    for xml_file in img_sets:
        image_info,image_anno = parse_xml(os.path.join(anno_path,xml_file+'.xml'))
        image_infos[xml_file]=image_info
        image_anns[xml_file]=image_anno


    num_imgs = len(list(image_infos.keys()))
    for img_name,img_info in tqdm(image_infos.items(),total=num_imgs):
        cropping_fun(image_anns[img_name],img_info,img_root,save_root=save_root,scaling_factor=scaling_factor,min_size=min_size)




    


if __name__ == '__main__':
    main()
