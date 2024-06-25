'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel
from models.transform_layers import NormalizeLayer
from torch.nn.utils import spectral_norm
from torchvision import models
import random

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Pretrain_ResNet(BaseModel):
    def __init__(self, num_classes=10):
        expansion = 1
        last_dim = 512 * expansion
        super(Pretrain_ResNet, self).__init__(last_dim, num_classes)

        self.in_planes = 64
        self.last_dim = last_dim

        mu = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
        self.norm = lambda x: (x - mu) / std
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        i = 0
        num = 30
        for param in self.backbone.parameters():
            if i<num:
                param.requires_grad = False
            i+=1

    def penultimate(self, x, all_features=False):
        x = self.norm(x)
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n


class Pretrain_Wide_ResNet(BaseModel):
    def __init__(self, num_classes=10):
        expansion = 1
        last_dim = 2048 * expansion
        super(Pretrain_Wide_ResNet, self).__init__(last_dim, num_classes)

        self.in_planes = 64
        self.last_dim = last_dim

        mu = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
        self.norm = lambda x: (x - mu) / std
        self.backbone = models.wide_resnet50_2(pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        i = 0
        num = 72
        for param in self.backbone.parameters():
            if i<num:
                param.requires_grad = False
            i+=1

    def penultimate(self, x, all_features=False):
        x = self.norm(x)
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n

def Pretrain_ResNet18_Model(num_classes):
    return Pretrain_ResNet(num_classes=num_classes)

def Pretrain_Wide_ResNet_Model(num_classes):
    return Pretrain_Wide_ResNet(num_classes=num_classes)
