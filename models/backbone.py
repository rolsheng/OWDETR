# ------------------------------------------------------------------------
# OW-DETR: Open-world Detection Transformer
# Akshita Gupta^, Sanath Narayan^, K J Joseph, Salman Khan, Fahad Shahbaz Khan, Mubarak Shah
# https://arxiv.org/pdf/2112.01513.pdf
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict
from torchvision.models.resnet import resnet50

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool,experts:list):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        #modal-specific stem conv
        self.conv1=nn.ModuleDict()
        self.experts = experts
        self.return_layers = return_layers
        for e in self.experts:
            if e in ['depth','edge']:
                self.conv1[e] = nn.Sequential(
                    nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=2,padding=1,bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1,bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,stride=1,padding=0,bias=False),

                )
            else:
                self.conv1[e] = nn.Sequential(
                    nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=1,bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1,bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,stride=1,padding=0,bias=False)
                )
            

    def forward(self, tensor_list: NestedTensor):
        out = {e:OrderedDict() for e in tensor_list.tensors}
        out['rgb'] = self.body(tensor_list.tensors['rgb'])
        for e in self.experts:
            xe = tensor_list.tensors[e]
            xe = self.conv1[e](xe)
            for name,module in self.body.items():
                if 'layer' in name:
                    xe = module(xe)
                    if name in self.return_layers:
                        out_name = self.return_layers[name]
                        out[e][out_name] = xe
        
        wrapped_out: Dict[str, NestedTensor] = {}
        for name, x in out['rgb'].items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            lvl_out = OrderedDict()
            lvl_out['rgb'] = out['rgb'][name]
            for e in self.experts:
                lvl_out[e] = out[e][name]
            wrapped_out[name] = NestedTensor(lvl_out, mask)
        return wrapped_out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 experts:list,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        if name == 'resnet50':
            print("resnet50")
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=norm_layer)
        else:
            print("DINO resnet50")
            backbone = resnet50(pretrained=False, replace_stride_with_dilation=[False, False, dilation], norm_layer=norm_layer)
            if is_main_process():
                state_dict = torch.load("ckpt/dino_resnet50_pretrain.pth")
                backbone.load_state_dict(state_dict, strict=False)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers,experts)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors['rgb'].dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.experts,args.dilation)
    model = Joiner(backbone, position_embedding)
    return model