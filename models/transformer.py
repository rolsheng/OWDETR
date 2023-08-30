import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import torch.nn.functional as F
import copy
class Fusion(nn.Module):
    def __init__(self,
                 num_feature_levels,
                 in_dim=512,
                 out_dim=256,
                 dropout=0.1
                 ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.prompt_proj = nn.Linear(in_dim,out_dim,bias=False)

    def single_level_forward(self,src,memory):

        tgt = src.clone()
        # src_norm = F.normalize(src,p=2,dim=-1)
        # memory has been normalized
        reduced_memory = self.prompt_proj(memory.to(src.dtype))


        emb_dim = torch.as_tensor(src.shape[-1],device=src.device,dtype=src.dtype)
        attn_qk = torch.matmul(src,reduced_memory.transpose(1,2)) 
        # attn_qk = torch.div(attn_qk,torch.sqrt(emb_dim))
        attn_qk = F.softmax(attn_qk,dim=-1)
        attn_val = torch.matmul(attn_qk,reduced_memory)
        return tgt+attn_val
    def forward(self,features,prompt):

        if isinstance(features,list):
            attn_features = []
            for indx,feature in enumerate(feature):
                attn_features.append(self.single_level_forward(feature,prompt))
            return attn_features

        out = self.single_level_forward(features,prompt)
        return out
