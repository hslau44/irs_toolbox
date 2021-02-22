import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
import torchvision

from models.baseline import *
from models.cnn import *
from models.self_supervised import *
from models.utils import *

def create_pretrain_model(out_size=(2,3)):
    # External libraries required
    enc = create_vgg16(out_size)
    clf = Projection_head(512*out_size[0]*out_size[1],128,head='linear')
    model = ED_module(encoder=enc,decoder=clf)
    return model

def freeze_network(model):
    for _, p in model.named_parameters():
        p.requires_grad = False
    return model

def create_finetune_model(enc=None,out_size=(2,3)):
    # External libraries required
    if enc == None:
        enc = create_vgg16(out_size)
    else:
        enc = freeze_network(enc)
    clf = Classifier(512*out_size[0]*out_size[1],128,6)
    model = ED_module(encoder=enc,decoder=clf)
    return model


def create_model_SimCLR(out_size=(2,3)):
    # External libraries required
    enc = create_vgg16(out_size)
    clf = Projection_head(512*out_size[0]*out_size[1],128,head='linear')
    model = SimCLR(enc,clf)
    return model
