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
    """freeze all trainable parameter in model, return model"""
    for _, p in model.named_parameters():
        p.requires_grad = False
    return model


def add_classifier(enc,in_size,out_size,freeze):
    """
    add classifier (2 layers, first layer size: 128) on top of an encoder

    Arugments:
    enc (torch.nn.Module): encoder
    in_size (int): output/input size of the encoder/classifier
    out_size (int): output of the classifier
    freeze (bool): if true, freeze the encoder

    Return:
    model (torch.nn.Module)
    """
    if freeze == True:
        enc = freeze_network(enc)
    clf = Classifier(in_size,128,out_size)
    model = ED_module(encoder=enc,decoder=clf)
    return model

def add_SimCLR(enc,out_size):
    """
    add projection network

    Arugments:
    enc: encoder(nn.Module)
    out_size: output size of encoder

    Return:
    model (nn.Module)
    """
    clf = Projection_head(out_size,128,head='linear')
    model = SimCLR(enc,clf)
    return model

def add_SimCLR_multi(enc1,enc2,out_size1,out_size2):
    """
    add 2 projection networks on the two encoders to form one model

    Arugments:
    enc1: 1st encoder(nn.Module)
    out_size1: output size of the 1st encoder
    enc2: 2nd encoder(nn.Module)
    out_size2: output size of the 2nd encoder

    Return:
    model (nn.Module):
    """
    dec1 = Projection_head(out_size1,128,head='linear')
    dec2 = Projection_head(out_size2,128,head='linear')
    model = SimCLR_multi(enc1, enc2, dec1, dec2)
    return model

# def create_baseline_model():
#     out = 96
#     enc = Encoder([32,64,out])
#     model = add_classifier(enc,out_size=10*out,freeze=False)
#     return model

def create_encoder(network,pairing):
    """
    Create CNN encoder used for the experiment

    Arguments:
    network (str): network architecutre of the encoder, options {'shallow','alexnet','resnet','vgg16'}
    pairing (str): pairing modality, options {'csi','nuc2','pwr'}

    Returns:
    encoder:


    """
    if network == "shallow":
        if pairing == 'csi':
            encoder = create_baseline_encoder(scale_factor=1)
            outsize = 960
        elif pairing == 'nuc2':
            encoder = create_baseline_encoder(scale_factor=1)
            outsize = 960
        elif pairing == 'pwr':
            encoder = create_baseline_encoder(scale_factor=3)
            outsize = 1152
        else:
            raise ValueError("pairing must be in {'csi','nuc2','pwr'}")
    elif network == "alexnet":
        if pairing == 'csi':
            encoder,outsize = create_alexnet((1,4),scale_factor=1)
        elif pairing == 'nuc2':
            encoder,outsize = create_alexnet((1,4),scale_factor=1)
        elif pairing == 'pwr':
            encoder,outsize = create_alexnet((4,1),scale_factor=2)
        else:
            raise ValueError("pairing must be in {'csi','nuc2','pwr'}")
    elif network == "resnet":
        if pairing == 'csi':
            encoder,outsize = create_resnet18((2,2))
        if pairing == 'nuc2':
            encoder,outsize = create_resnet18((2,2))
        elif pairing == 'pwr':
            encoder,outsize = create_resnet18((2,2))
        else:
            raise ValueError("pairing must be in {'csi','nuc2','pwr'}")
    elif network == "vgg16":
        if pairing == 'csi':
            encoder,outsize = create_vgg16((2,2))
        if pairing == 'nuc2':
            encoder,outsize = create_vgg16((2,2))
        elif pairing == 'pwr':
            encoder,outsize = create_vgg16((2,2))
        else:
            raise ValueError("pairing must be in {'csi','nuc2','pwr'}")
    else:
        raise ValueError("network must be in {'shallow','alexnet','resnet'}")
    return encoder, outsize
