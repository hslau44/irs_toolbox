import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
import torchvision
from models.utils import *


class DataAugmentation(nn.Module):

    def __init__(self, size):
        super(DataAugmentation, self).__init__()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=size),
#             torchvision.transforms.ToTensor()
        ])

    def forward(self, x):
        return self.transform(x), self.transform(x)


class Projection_head(nn.Module):
    """
    Projection head:

    head: linear/mlp
    """
    def __init__(self,dim_in,feat_dim,head='linear'):
        super(Projection_head, self).__init__()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self,X):
        X = self.head(X)
        X = F.normalize(X, dim=1)
        return X



class SimCLR(nn.Module):
    def __init__(self, encoder, decoder):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,t): # tuple
        batch_size = t[0].shape[0]
        t = torch.cat(t,dim=0) # tensor
        t = self.encoder(t)
        t = self.decoder(t)
        t = torch.split(t,batch_size,dim=0)
        return t # tuple
