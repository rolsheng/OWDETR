# partly taken from https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py
import functools
from typing import Any, Callable, Optional
import torch

import os
import tarfile
import collections
import logging
import copy
from torchvision.datasets import VisionDataset
import itertools

import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg

#OWOD splits
# VOC_CLASS_NAMES_COCOFIED = [
#     "airplane",  "dining table", "motorcycle",
#     "potted plant", "couch", "tv"
# ]

# BASE_VOC_CLASS_NAMES = [
#     "aeroplane", "diningtable", "motorbike",
#     "pottedplant",  "sofa", "tvmonitor"
# ]
Base_CLASS_NAME = [
    "pedestrian", "cyclist", "car", "truck", "tram", "tricycle", "bus"
]
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

class OWDetection(VisionDataset):
    """`OWOD in Pascal VOC format <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 args,
                 root,
                 years='2012',
                 image_sets='train',
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 no_cats=False,
                 filter_pct=-1):
        super(OWDetection, self).__init__(root, transforms, transform, target_transform)
        self.images = []
        self.annotations = []
        self.imgids = []
        self.imgid2annotations = {}
        self.image_set = []

        self.CLASS_NAMES = VOC_COCO_CLASS_NAMES
        self.MAX_NUM_OBJECTS = 64
        self.no_cats = no_cats
        self.args = args
        # import pdb;pdb.set_trace()

        for year, image_set in zip(years, image_sets):

            if year == "2007" and image_set == "test":
                year = "2007-test"
            valid_sets = ["t1_train", "t2_train", "t3_train", "t4_train", "t2_ft", "t3_ft", "t4_ft", "test", "all_task_test"]
            if year == "2007-test":
                valid_sets.append("test")

            # base_dir = DATASET_YEAR_DICT[year]['base_dir']
            voc_root = self.root
            annotation_dir = os.path.join(voc_root, 'Annotations')
            image_dir = os.path.join(voc_root, 'JPEGImages')

            if not os.path.isdir(voc_root):
                raise RuntimeError('Dataset not found or corrupted.' +
                                   ' You can use download=True to download it')
            file_names = self.extract_fns(image_set, voc_root)
            self.image_set.extend(file_names)

            self.images.extend([os.path.join(image_dir, x + ".jpg") for x in file_names])
            self.annotations.extend([os.path.join(annotation_dir, x + ".xml") for x in file_names])

            self.imgids.extend(self.convert_image_id(x, to_integer=True) for x in file_names)
            self.imgid2annotations.update(dict(zip(self.imgids, self.annotations)))

        if filter_pct > 0:
            num_keep = float(len(self.imgids)) * filter_pct
            keep = np.random.choice(np.arange(len(self.imgids)), size=round(num_keep), replace=False).tolist()
            flt = lambda l: [l[i] for i in keep]
            self.image_set, self.images, self.annotations, self.imgids = map(flt, [self.image_set, self.images,
                                                                                   self.annotations, self.imgids])
        assert (len(self.images) == len(self.annotations) == len(self.imgids))

    @staticmethod
    def convert_image_id(img_id, to_integer=False, to_string=False, prefix='2023'):
        if to_integer:
            return int(prefix + img_id.replace('_', ''))
        if to_string:
            x = str(img_id)
            assert x.startswith(prefix)
            x = x[len(prefix):]
            if len(x) == 7 or len(x) == 4:
                return x
            return x[:4] + '_' + x[4:]

    @functools.lru_cache(maxsize=None)
    def load_instances(self, img_id):
        tree = ET.parse(self.imgid2annotations[img_id])
        target = self.parse_voc_xml(tree.getroot())
        image_id = target['annotation']['filename']
        instances = []
        for obj in target['annotation']['object']:
            cls = obj["name"].lower()
            # if cls in VOC_CLASS_NAMES_COCOFIED:
            #     cls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]
            bbox = obj["bndbox"]
            bbox = [float(bbox[x]) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instance = dict(
                category_id=VOC_COCO_CLASS_NAMES.index(cls),
                bbox=bbox,
                area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                image_id=img_id
            )
            instances.append(instance)
        unk_instances = []
        if 'unk_object' in target['annotation']:
            unk_objects = target['annotation']['unk_object']
            if not isinstance(unk_objects,list):
                unk_objects = [unk_objects]
            for unk_obj in unk_objects:
                bbox = unk_obj["bndbox"]
                bbox = [float(bbox[x]) for x in ["xmin", "ymin", "xmax", "ymax"]]
                bbox[0] -= 1.0
                bbox[1] -= 1.0
                score = float(unk_obj['score'])
                unk_instance = dict(
                    bbox=bbox,
                    score = score,
                    area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    image_id=img_id
                )
                unk_instances.append(unk_instance)
        return target, instances,unk_instances

    def extract_fns(self, image_set, voc_root):
        splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        return file_names

    ### OWOD
    def remove_prev_class_and_unk_instances(self, target,unk_target):
        # For training data. Removing earlier seen class objects.
        # However store gt objects in the following task as unknow objects and pesudo bboxs
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        valid_classes = range(prev_intro_cls, prev_intro_cls + curr_intro_cls)
        unk_classes = range(prev_intro_cls+curr_intro_cls,self.args.num_classes-1) # exclude unknow class
        entry = copy.copy(target)
        unk_entry = copy.copy(unk_target)
        for annotation in copy.copy(entry):
            if annotation["category_id"] not in valid_classes:
                entry.remove(annotation)
            if annotation['category_id'] in unk_classes:
                # store gt box in following task as unk_objects
                annotation.pop('category_id')
                annotation['score'] = float(1.0)
                unk_entry.append(annotation)
        return entry,unk_entry

    def remove_unknown_instances(self, target,unk_target):
        # For finetune data. 
        # Storing gt box in the following task as  unknown objects...
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        valid_classes = range(0, prev_intro_cls+curr_intro_cls)
        unk_classes = range(prev_intro_cls+curr_intro_cls,self.args.num_classes-1)
        entry = copy.copy(target)
        unk_entry = copy.copy(unk_target)
        for annotation in copy.copy(entry):
            if annotation["category_id"] not in valid_classes:
                entry.remove(annotation)
            if annotation['category_id'] in unk_classes:
                annotation.pop('category_id')
                annotation['score'] = float(1.0)
                unk_entry.append(annotation)
        return entry,unk_entry

    def label_known_class_and_unknown(self, target):
        # For test and validation data.
        # Label known instances the corresponding label and unknown instances as unknown.
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        total_num_class = self.args.num_classes 
        known_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in  copy.copy(entry):
        # for annotation in entry:
            if annotation["category_id"] not in known_classes:
                annotation["category_id"] = total_num_class - 1
        return entry

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        image_set = self.transforms[0]
        img = Image.open(self.images[index]).convert('RGB')
        target, instances, unk_instances= self.load_instances(self.imgids[index])
        if 'train' in image_set:
            instances,unk_instances = self.remove_prev_class_and_unk_instances(instances,unk_instances)
        elif 'val' in image_set:
            instances = self.label_known_class_and_unknown(instances)
        elif 'ft' in image_set:
            instances,unk_instances = self.remove_unknown_instances(instances,unk_instances)

        w, h = map(target['annotation']['size'].get, ['width', 'height'])
        if len(unk_instances)>0:
            target = dict(
                image_id=torch.tensor([self.imgids[index]], dtype=torch.int64),
                labels=torch.tensor([i['category_id'] for i in instances], dtype=torch.int64),
                area=torch.tensor([i['area'] for i in instances], dtype=torch.float32),
                boxes=torch.as_tensor([i['bbox'] for i in instances], dtype=torch.float32),
                orig_size=torch.as_tensor([int(h), int(w)]),
                size=torch.as_tensor([int(h), int(w)]),
                iscrowd=torch.zeros(len(instances), dtype=torch.uint8),
                unk_boxes = torch.as_tensor([i['bbox'] for i in unk_instances], dtype=torch.float32),
                unk_scores = torch.as_tensor([i['score'] for i in unk_instances], dtype=torch.float32)
            )
        else :
            target = dict(
                image_id=torch.tensor([self.imgids[index]], dtype=torch.int64),
                labels=torch.tensor([i['category_id'] for i in instances], dtype=torch.int64),
                area=torch.tensor([i['area'] for i in instances], dtype=torch.float32),
                boxes=torch.as_tensor([i['bbox'] for i in instances], dtype=torch.float32),
                orig_size=torch.as_tensor([int(h), int(w)]),
                size=torch.as_tensor([int(h), int(w)]),
                iscrowd=torch.zeros(len(instances), dtype=torch.uint8)
            )
        if self.transforms[-1] is not None:
            img, target = self.transforms[-1](img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
                # def_dic['unk_object'] = [def_dic['unk_object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)
    
class CustomImageDataset(VisionDataset):
    """
    Args:

    """
    def __init__(self, 
                 root, 
                 transforms=None,
                 transform = None,
                 target_transform = None) :
        super().__init__(root, transforms, transform, target_transform)
        self.images = []
        self.CLASS_NAMES = VOC_COCO_CLASS_NAMES
        self.root = root
        self.transforms = transforms
        self.images = [os.path.join(root,file_name) for file_name in os.listdir(self.root)]
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index):
        img = Image.open(self.images[index])
        w,h = img.size
        target={
            "img_path":self.images[index],
            "orig_size":(int(h),int(w)),
        }

        if self.transforms[-1] is not None:
            img,target = self.transforms[-1](img,target)
        return img,target
        
        

        