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



class Split_Channel(nn.Module):
    def __init__(self,unsqueeze=True):
        super(Split_Channel, self).__init__()
        self.unsqueeze = unsqueeze

    def forward(self, x):
        return self.split_channel(x)

    def split_channel(self, x):
        if self.unsqueeze == True:
            return [x[:,i].unsqueeze(1) for i in range(x.shape[1])]
        else:
            return [x[:,i] for i in range(x.shape[1])]


class CNN_SingleStream(nn.Module):

    def __init__(self):
        super(CNN_SingleStream, self).__init__()
        ### 1st ###
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=2)
        self.actv1 = nn.ReLU()
        self.norm1 = Lambda(lambda x:x)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(1,1))
        ### 2nd ###
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=2)
        self.actv2 = nn.ReLU()
        self.norm2 = Lambda(lambda x:x)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=(1,1))
        ### 3rd ###
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,stride=2)
        self.actv3 = nn.ReLU()
        self.norm3 = Lambda(lambda x:x)
        self.pool3 = Lambda(lambda x:x)
        ### Global_pooling ###
        self.final = nn.AdaptiveAvgPool2d((1,1))

    def forward(self,X):
        X = self.pool1(self.norm1(self.actv1(self.conv1(X))))
        X = self.pool2(self.norm2(self.actv2(self.conv2(X))))
        X = self.pool3(self.norm3(self.actv3(self.conv3(X))))
        X = self.final(X)
        X = torch.flatten(X,1)
        return X

class CNN_MultiStream(nn.Module):

    def __init__(self):
        super(CNN_MultiStream, self).__init__()
        self.split = Split_Channel()
        self.convnet_c1 = CNN_SingleStream()
        self.convnet_c2 = CNN_SingleStream()
        self.convnet_c3 = CNN_SingleStream()
        self.concat = Lambda(lambda a,b,c: torch.cat((a,b,c),-1))

    def forward(self,x):
        a,b,c = self.split(x)
        a = self.convnet_c1(a)
        b = self.convnet_c2(b)
        c = self.convnet_c2(c)
        x = torch.cat((a,b,c),-1)
        return x

class LSTM_Baseline(nn.Module):
    def __init__(self,seq_size,feature_size):
        """
        Baseline LSTM model

        attr:
        seq_size: length of the sequence
        feature_size: feature size of each interval in the sequence

        """
        super(LSTM_Baseline, self).__init__()

        self.lstm1 = nn.LSTM(feature_size,200)
        self.lstm2 = nn.LSTM(200,3)
        self.linear1 = nn.Linear(3*seq_size,30)
        self.linear2 = nn.Linear(30,8)

    def forward(self,X):
        X, _ = self.lstm1(X)
        X, _ = self.lstm2(X)
        X = torch.flatten(X,1)
        X = self.linear1(X)
        X = self.linear2(X)
        return F.log_softmax(X,dim=1)

class CNN_LSTM_OneToOne(nn.Module):
    def __init__(self,seq_size):
        """
        CNN-LSTM model

        attr:
        seq_size: length of the sequence
        feature_size: feature size of each interval in the sequence

        """
        super(CNN_LSTM_OneToOne, self).__init__()
        self.split = Split_Channel(unsqueeze=False)
        self.cnn = CNN_MultiStream()
        self.lstm = LSTM_Baseline(seq_size=seq_size, feature_size=3*128)

    def forward(self,X):
        X = self.split(X)
        X = [self.cnn(X[i]).unsqueeze(1) for i in range(len(X))]
        X = torch.cat(X,1)
        X = self.lstm(X)
        return X

# sample_size = 128
# tensor_sample  = torch.ones(size=(sample_size, 20, 3, 40, 30))
# model = CNN_LSTM_OneToOne(seq_size=20)
# output = model.forward(tensor_sample)
# output.shape
