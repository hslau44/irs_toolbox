import torch
import torch.nn as nn
from torch.nn import functional as F
# from .utils import Lambda
from models.utils import *


class LSTM(nn.Module):
    def __init__(self,seq_size,feature_size):
        """
        2 layer LSTM model: feature_size --> 200 --> 3

        Arguments:
        seq_size (int): length of the sequence
        feature_size (int): feature size of each interval in the sequence
        """
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(feature_size,200)
        self.lstm2 = nn.LSTM(200,3)


    def forward(self,X):
        X, _ = self.lstm1(X)
        X, _ = self.lstm2(X)
        X = torch.flatten(X,1)
        return X


class Encoder(nn.Module):
    def __init__(self,num_filters):
        """
        Three layers Convolutional Layers for spectrogram with size (1,65,501)

        Arguments:
        num_filters (list<int>): number of filters for each of the convolutional Layer length of list == 3
        """
        super(Encoder, self).__init__()
        assert len(num_filters) == 4
        l0,l1,l2,l3 = num_filters
        ### 1st ###
        self.conv1 = nn.Conv2d(l0,l1,kernel_size=5,stride=1)
        self.norm1 = nn.BatchNorm2d(l1) # nn.BatchNorm2d()
        self.actv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        ### 2nd ###
        self.conv2 = nn.Conv2d(l1,l2,kernel_size=4,stride=2)
        self.norm2 = nn.BatchNorm2d(l2)
        self.actv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        ### 3rd ###
        self.conv3 = nn.Conv2d(l2,l3,kernel_size=3,stride=3)
        self.norm3 = Lambda(lambda x:x)
        self.actv3 = nn.Tanh()
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))

    def forward(self,X):
        X = self.pool1(self.actv1(self.norm1(self.conv1(X))))
        X = self.pool2(self.actv2(self.norm2(self.conv2(X))))
        X = self.pool3(self.actv3(self.norm3(self.conv3(X))))
        X = torch.flatten(X, 1)
        # print(X.shape)
        return X




class Encoder_F(nn.Module):

    def __init__(self,num_filters):
        """
        Four layers Convolutional Layers for spectrogram with size (1,65,501)

        Arguments:
        num_filters (list<int>): number of filters for each of the convolutional Layer, length of list == 4
        """
        super(Encoder_F, self).__init__()
        l1,l2,l3,l4 = num_filters
        ### 1st ###
        self.conv1 = nn.Conv2d(1,l1,kernel_size=(5,5),stride=(2,2))
        self.norm1 = nn.BatchNorm2d(l1)
        self.actv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((1,2))
        ### 2nd ###
        self.conv2 = nn.Conv2d(l1,l2,kernel_size=(4,4),stride=(2,2))
        self.norm2 = nn.BatchNorm2d(l2)
        self.actv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((1,2))
        ### 3rd ###
        self.conv3 = nn.Conv2d(l2,l3,kernel_size=(3,3),stride=(2,2))
        self.norm3 = nn.BatchNorm2d(l3)
        self.actv3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d((1,2)) # nn.AdaptiveAvgPool2d((2,2))
        ### 4th ###
        self.conv4 = nn.Conv2d(l3,l4,kernel_size=(2,2),stride=(2,2))
        self.norm4 = Lambda(lambda x:x)
        self.actv4 = nn.Tanh()
        self.pool4 = nn.AdaptiveAvgPool2d((1,2))

    def forward(self,X):
        X = self.pool1(self.actv1(self.norm1(self.conv1(X))))
        X = self.pool2(self.actv2(self.norm2(self.conv2(X))))
        X = self.pool3(self.actv3(self.norm3(self.conv3(X))))
        X = self.pool4(self.actv4(self.norm4(self.conv4(X))))
        X = torch.flatten(X, 1)

        return X

class SimpleCNN(nn.Module):

    def __init__(self,setting='1st'):
        """
        Simple Convolutional Neural Networks used in the previous work (Li et al. 2020)

        parameter:
        setting (str): must be either {'1st','2nd'}
        """
        super(SimpleCNN, self).__init__()
        if setting == '1st':
            num_filter,kernel_size,latent = 32,5,238080
        elif setting == '2nd':
            num_filter,kernel_size,latent = 64,2,512000
        else:
            raise ValueError("setting must be either {'1st','2nd'}")
        ### 1st ###
        self.conv1 = nn.Conv2d(1,num_filter,kernel_size)
        self.actv_cnn = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.dropout_cnn = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(latent,256)
        self.actv_1 = nn.ReLU()
        self.linear2 = nn.Linear(256,128)
        self.actv_2 = nn.ReLU()
        self.linear3 = nn.Linear(128,6)

    def forward(self,X):
        X = self.dropout_cnn(self.pool1(self.actv_cnn(self.conv1(X))))
        X = torch.flatten(X, 1)
        X = self.actv_1(self.linear1(X))
        X = self.actv_2(self.linear2(X))
        X = self.linear3(X)
        return X
