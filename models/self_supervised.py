import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
import torchvision
from models.utils import *


class DataAugmentation(nn.Module):

    def __init__(self, size, transforms):
        """
        Data augmentation pipeline (nn.Module)
        """
        super(DataAugmentation, self).__init__()

        if transforms:
            self.transform = transform
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(size=size),
#             torchvision.transforms.ToTensor()
            ])

    def forward(self, x):
        return self.transform(x), self.transform(x)


class Projection_head(nn.Module):

    def __init__(self,dim_in,feat_dim,head='linear'):
        """
        Projection head, either
        1). linear: linear projection
        2). mlp: multi-layer perceptron, with dim_in -> dim_in -> feat_dim

        Arguments:
        dim_in (int): input size of the projection
        feat_dim (int): output size of the projection
        head (str): {'linear','mlp'}
        """
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
        """
        Encoder-Decoder architecture
        all views in the input tensors will pass through the same model

        Arguments:
        encoder (nn.Module): encoder for the 1st Encoder-Decoder architecture
        decoder (nn.Module): decoder for the 1st Encoder-Decoder architecture
        """
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,t): # tuple
        """
        Arguments:
        t (tuple<torch.Tensor>): a tuple of tensors, each tensor represents the single view of a batch of data (n,c,w,h)
        where n, c, w, h is number of samples, number of channel, width and hight of the image respectively

        Returns:
        t (tuple<torch.Tensor>): a tuple of the forward-passed tensors
        """
        batch_size = t[0].shape[0]
        t = torch.cat(t,dim=0) # tensor
        t = self.encoder(t)
        t = self.decoder(t)
        t = torch.split(t,batch_size,dim=0)
        return t # tuple



class SimCLR_multi(nn.Module):
    def __init__(self, enc1, enc2, dec1, dec2):
        """
        Two parallel Encoder-Decoder architecture
        1st tensor and 2nd tensor the 1st and 2nd model respectively

        Arguments:
        enc1 (nn.Module): encoder for the 1st Encoder-Decoder architecture
        enc2 (nn.Module): encoder for the 2nd Encoder-Decoder architecture
        dec1 (nn.Module): decoder for the 1st Encoder-Decoder architecture
        dec2 (nn.Module): decoder for the 2nd Encoder-Decoder architecture
        """
        super(SimCLR_multi, self).__init__()
        self.encoder = enc1
        self.decoder = dec1
        self.encoder2 = enc2
        self.decoder2 = dec2

    def forward(self,t): # tuple
        """
        Arguments:
        t (tuple<torch.Tensor>): a tuple of tensors, each tensor represents the single view of a batch of data (n,c,w,h)
        where n, c, w, h is number of samples, number of channel, width and hight of the image respectively
        length of t must be equal to 2

        Returns:
        t (tuple<torch.Tensor>): a tuple of the forward-passed tensors
        """
        o1 = self.encoder(t[0])
        o1 = self.decoder(o1)
        o2 = self.encoder2(t[1])
        o2 = self.decoder2(o2)
        return o1,o2 # tuple
