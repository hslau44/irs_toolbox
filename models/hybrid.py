import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Encoder(nn.Module):
    """
    Single channel for radio image (30,30,1)
    """
    def __init__(self):
        super(Encoder, self).__init__()
        ### 1st ###
        self.conv1 = nn.Conv2d(1,64,kernel_size=(5,5),stride=(3,3))
        self.actv1 = nn.ReLU()
        self.norm1 = Lambda(lambda x:x)
        self.pool1 = Lambda(lambda x:x) # nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        ### 2nd ###
        self.conv2 = nn.Conv2d(64,128,kernel_size=(4,4),stride=(2,2))
        self.actv2 = nn.ReLU()
        self.norm2 = Lambda(lambda x:x)
        self.pool2 = Lambda(lambda x:x)
        ### 3rd ###
        self.conv3 = nn.Conv2d(128,256,kernel_size=(3,3),stride=(2,2))
        self.actv3 = nn.ReLU()
        self.norm3 = Lambda(lambda x:x)
        self.pool3 = Lambda(lambda x:x)

    def forward(self,X):
        X = X.permute(0,3,1,2)
        X = self.pool1(self.norm1(self.actv1(self.conv1(X))))
        X = self.pool2(self.norm2(self.actv2(self.conv2(X))))
        X = self.pool3(self.norm3(self.actv3(self.conv3(X))))
#         X = self.actv4(self.conv4(X))
        X = torch.flatten(X, 1)
        return X


class Encoder_View(nn.Module):
    """

    This model first divides a spectrogram into number of pieces based on window and channel,
    each piece is processed with the single encoder and output a matrix with shape (batch_size, seq_size, num_features),
    input shape: (batch_size,num_frame,num_channel)
    """
    def __init__(self, encoder, window=30, channel=30):
        super(Encoder_View, self).__init__()
        self.encoder = encoder
        self.window = window
        self.channel = channel
        self.size = None

    def forward(self,x):
        # pre
        self.size = x.shape
        assert self.size[1]%self.window == 0
        assert self.size[2]%self.channel == 0
        x = x.reshape(-1,self.size[1]//self.window,self.window,self.channel,self.size[2]//self.channel)
        x = x.permute(0,1,4,2,3)
        assert x.shape[3:] == (self.window,self.channel)
        self.size = x.shape[:3]
        x = x.reshape(-1,self.window,self.channel,1)
        # processing
        x = self.encoder(x)
        # post
        x = x.reshape(*self.size[:-1],self.size[-1]*x.shape[-1])
        return x


class CNN_LSTM_Module(nn.Module):
    def __init__(self,cnn,lstm):
        """
        CNN-LSTM model

        attr:
        seq_size: length of the sequence
        feature_size: feature size of each interval in the sequence

        """
        super(CNN_LSTM_Module, self).__init__()
        self.cnn = cnn
        self.lstm = lstm

    def forward(self,X):
        X = self.cnn(X)
        X = self.lstm(X)
        return X
