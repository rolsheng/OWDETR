from torch.utils.data.sampler import Sampler
import torch
import itertools
from collections import defaultdict
import math
import util.misc as util
import os
import tarfile
import collections
import logging
import copy
from torchvision.datasets import VisionDataset
import itertools
from datasets.torchvision_datasets.open_world import VOC_COCO_CLASS_NAMES
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
class RepeatFactorTrainingSampler(Sampler):
    """
    Similar to TrainingSampler, but a sample may appear more times than others based
    on its "repeat factor". This is suitable for training on class imbalanced datasets like LVIS.
    """

    def __init__(self, repeat_factors, *, shuffle=True, seed=None):
        """
        Args:
            repeat_factors (Tensor): a float vector, the repeat factor for each indice. When it's
                full of ones, it is equivalent to ``TrainingSampler(len(repeat_factors), ...)``.
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._shuffle = shuffle
        self._seed = int(seed)

        self._rank = util.get_rank()
        self._world_size = util.get_world_size()
        self.epoch = 0
        # Split into whole number (_int_part) and fractional (_frac_part) parts.
        self._int_part = torch.trunc(repeat_factors)
        self._frac_part = repeat_factors - self._int_part

    @staticmethod
    def repeat_factors_from_category_frequency(args,dataset, repeat_thresh):
        """
        Args:
            dataset
            repeat_thresh (float): frequency threshold below which data is repeated.
                If the frequency is half of `repeat_thresh`, the image will be
                repeated twice.

        Returns:
            torch.Tensor: the i-th element is the repeat factor for the dataset image
                at index i.
        """
        dataset_dicts=[]
        annotations = dataset.annotations 
        for anno_idx,anno_path in enumerate(annotations):
            target = RepeatFactorTrainingSampler.parse_rec(anno_path)
            dataset_dicts.append(target)
        # 1. For each seen category c, compute the fraction of images that contain it: f(c)
        prev_intro_cls = args.PREV_INTRODUCED_CLS
        curr_intro_cls = args.CUR_INTRODUCED_CLS
        seen_classes = prev_intro_cls + curr_intro_cls
        if 'ft' in args.train_set:
            valid_cls_idx = list(range(0,seen_classes))
        elif 'train' in args.train_set:
            valid_cls_idx = list(range(prev_intro_cls,seen_classes))

        num_instances = 0
        bbox_freq = defaultdict(int)
        image_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"] if ann['category_id'] in valid_cls_idx}
            num_instances+=len(cat_ids)
            for cat_id in cat_ids:
                bbox_freq[cat_id] += 1
            for cat_id in set(cat_ids):
                image_freq[cat_id] +=1
        num_images = len(dataset_dicts)
        for k, v in bbox_freq.items():
            bbox_freq[k] = v / num_instances
        for k,v in image_freq.items():
            image_freq[k] = v/num_images

       
        category_rep = {
            cat_id: max(1.0, math.sqrt(repeat_thresh / math.sqrt(image_freq[cat_id]*bbox_freq[cat_id])))
            for cat_id in image_freq.keys()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        for dataset_dict in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"] if ann["category_id"] in valid_cls_idx}
            rep_factor = max({category_rep[cat_id] for cat_id in cat_ids})
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)
    @staticmethod
    def parse_rec(filename):
        tree = ET.parse(filename).getroot()
        # import pdb;pdb.set_trace()
        objects = {}
        objects['annotations'] = []
        for obj in tree.findall('object'):
            obj_struct = {}
            cls_name = obj.find('name').text
            obj_struct['name'] = cls_name
            obj_struct['category_id'] = VOC_COCO_CLASS_NAMES.index(cls_name)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            objects['annotations'].append(obj_struct)

        return objects

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.

        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.

        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()
        # Construct a list of indices in which we repeat images as specified
        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))
        return torch.tensor(indices, dtype=torch.int64)

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)
    def __len__(self):
        return self._int_part.shape[0]
    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        while True:
            # Sample indices with repeats determined by stochastic rounding; each
            # "epoch" may have a slightly different size due to the rounding.
            indices = self._get_epoch_indices(g)
            if self._shuffle:
                randperm = torch.randperm(len(indices), generator=g)
                yield from indices[randperm]
            else:
                yield from indices
    def set_epoch(self,epoch):
        self.epoch = epoch