import torch
import torch.nn as nn
import numpy as np 

class MemoryBank(nn.Module):
    def __init__(self,
                 min_cache = None,
                 max_cache = None,
                 cache_category_file = None,
                 **kwargs):
        super().__init__(**kwargs)

        with open(cache_category_file,'r') as txt:
            self.cache_categories = np.array([int(cls_indx.rstrip()) for cls_indx in txt.readlines()],dtype=np.int8)
        
        self.num_classes = self.cache_categories.shape[0]
        self.memory_cache = {
            c:{
                "query_features":np.empty((max_cache,256))
            } 
            for c in self.cache_categories
         }
        self.max_cache = max_cache
        self.min_cache = min_cache

        self.memory_cache_max_idx = np.zeros(self.num_classes,dtype=int)

    def forward(self,query_features,gt_classes):

        augmented_query_features = query_features
        #target classes in the current batch
        target_classes = []
        cur_gt_classes = gt_classes.cpu().numpy()
        tail_idxs = np.where(np.isin(cur_gt_classes,self.cache_categories))[0]
        target_classes.extend(cur_gt_classes[tail_idxs])

        #count number of instances per category for target category
        target_instances = dict()
        for t in target_classes:
            target_instances[t]=target_instances.get(t,0)+1
        
        new_features = torch.tensor([]).cuda()
        for c in set(target_classes):
            num_samp_cache = self.memory_cache_max_idx[c]
            # if target class exist in memory,than sample min(num_samp_cache,self.min_cache)
            if num_samp_cache>0:
                num_new_samps = min(self.min_cache,num_samp_cache)
                # get sampled feature from top memory 
                cache_idx = np.arange(num_samp_cache-num_new_samps,num_samp_cache)

                new_feats = torch.from_numpy(self.memory_cache[c]['query_features'][cache_idx])
                new_features = torch.cat([new_features,new_feats.cuda()],dim=0)
        # update memory bank with query_feature and gt_classes
        self.update_memory_bank(query_features,gt_classes)

        # concate sampled feature and query feature into together
        if new_features.shape[0]>0:
            augmented_query_features = torch.cat([augmented_query_features,new_features],dim=0)


        return augmented_query_features


            





