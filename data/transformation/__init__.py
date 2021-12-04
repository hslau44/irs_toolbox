import torch
import numpy as np
import pandas as pd
import torchvision.transforms as T

class Transform(object):
    """
    Custom Transform

    Level 1: ReduceRes(),CutFrame()
    Level 2: Unsqueeze(),UnsqueezebyRearrange()
    Level 3: StackChannel()
    Level 3: ToStackImg()
    """

    def __init__(self,func=None):
        self.func = func

    def __call__(self, X):
        if isinstance(self.func,list):
            for f in self.func:
                X = f(X)
        elif hasattr(self.func,'__call__'):
            X = self.func(X)
        return X

# Lv1
class ReduceRes(Transform):
    """
    Reduce time resolution by factors

    Arguements:
    x (int) - factor on axis 0
    y (int) - factor on axis 1
    """
    def __init__(self,x=1,y=4):
        assert x>0 and y>0, f'factor x and y must be > 1, get{x,y}'
        self.x = x
        self.y = y

    def __call__(self, X):
        assert len(X.shape) == 2, 'input array dimensions must be equal to 2'
        return X[::self.x,::self.y]

class CutFrame(Transform):
    """
    Cut frame
    """
    def __init__(self,x0=0,x1=70,y0=0,y1=None):
        if x1:
            assert x0 < x1
        if y1:
            assert y0 < y1
        self.idx = dic[keep]
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

    def __call__(self, X):
        x1,y1 = self.x1,self.y1
        if x1 == None:
            x1 = X.shape[0]
        if y1 == None:
            y1 = X.shape[1]
        assert x1 <= X.shape[0]
        assert y1 <= X.shape[1]
        return X[self.x0:x1,self.y0:y1]

# Lv2
class Unsqueeze(Transform):
    """
    unsqueeze channel

    Example:
    input: tensor with shape (70,1600)
    dim: 0
    return: tensor with shape (1,70,1600)
    """
    def __init__(self,dim=0):
        self.dim = dim

    def __call__(self, X):
        return np.expand_dims(X,axis=self.dim)

class StackChannel(Transform):
    """
    Make copy of the image and stack along the channel to increase the number of channel

    Example:
    input: tensor with shape (1,70,1600)
    stack: 3
    dim: 0
    return: tensor with shape (3,70,1600)
    """
    def __init__(self,stack=3,dim=0):
        self.stack = stack
        self.dim = dim

    def __call__(self, X):
        return np.concatenate([X for _ in range(self.stack)],axis=self.dim)

class UnsqueezebyRearrange(Transform):
    """
    **Custom** transform 1-channel Amptitude-PhaseShift frame into 2 channel frame, with Amptitude on top of PhaseShift

    Example:
    input: tensor with shape (1,140,1600)
    return: tensor with shape (2,70,1600)
    """
    def __call__(self, X):
        r,w = X.shape
        return X.reshape(2,r//2,w)

# Lv3
class ToStackImg(Transform):
    """
    Transform a 1 long-frame into numbers of small-frames

    Arugments:
    n_seq:  number of frames to be return

    Example:
    input: tensor with shape (1,70,1600)
    n_seq: 10
    return: tensor with shape (10,1,70,160)
    """
    def __init__(self,n_seq):
        self.n_seq = n_seq

    def __call__(self, X):
        c,r,w = X.shape
        assert w%self.n_seq == 0, 'length must be able to be divided by n_seq'
        X = X.reshape(c,r,self.n_seq,w//self.n_seq)
        return np.transpose(X,(2,0,1,3))

def Transform_CnnLstmS():
    """
    Torch Transformation (torchvision.transforms.transforms.Compose)
    for resolution-reduced CNN-LSTM
    """
    return T.Compose([ReduceRes(),Unsqueeze(),ToStackImg(25)])

def Transform_CnnS():
    """
    Torch Transformation (torchvision.transforms.transforms.Compose)
    for resolution-reduced CNN
    """
    return T.Compose([ReduceRes(),Unsqueeze()])

def Transform_Cnn():
    """
    Torch Transformation (torchvision.transforms.transforms.Compose)
    for resolution-reduced CNN
    """
    return T.Compose([Unsqueeze()])

def Transform_ReduceRes():
    """
    Torch Transformation (torchvision.transforms.transforms.Compose)
    to reduce resolution by a factor of 4
    """
    return T.Compose([ReduceRes()])
