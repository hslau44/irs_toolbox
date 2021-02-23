import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
import torchvision

from models.baseline import *
from models.cnn import *
from models.self_supervised import *
from models.utils import *


def freeze_network(model):
    for _, p in model.named_parameters():
        p.requires_grad = False
    return model


def add_SimCLR(enc,out_size):
    """
    enc: encoder(nn.Module)
    out_size: output size of encoder
    """
    clf = Projection_head(out_size,128,head='linear')
    model = SimCLR(enc,clf)
    return model


def add_classifier(enc,out_size,freeze):
    if freeze == True:
        enc = freeze_network(enc)
    clf = Classifier(out_size,128,6)
    model = ED_module(encoder=enc,decoder=clf)
    return model


def create_baseline():
    out = 96
    enc = Encoder([32,64,out])
    model = add_classifier(enc,out_size=10*out,freeze=False)
    return model
