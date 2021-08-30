import torch
from torch import Tensor
import torch.nn as nn
from torchsummary import summary
from models.utils import Lambda

class SmallEncoder(nn.Module):
    def __init__(self,l1,l2,l3):
        """
        3-layer CNN Encoder for CNN-LSTM model

        Arguments:
        l1 (int): number of neuron on the 1st layer
        l2 (int): number of neuron on the 2nd layer
        l3 (int): number of neuron on the 3rd layer
        """
        super(SmallEncoder, self).__init__()
        ### 1st ###
        self.conv1 = nn.Conv2d(1,l1,kernel_size=4,stride=2)
        self.norm1 = nn.BatchNorm2d(l1) # nn.BatchNorm2d()
        self.actv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        ### 2nd ###
        self.conv2 = nn.Conv2d(l1,l2,kernel_size=3,stride=2)
        self.norm2 = nn.BatchNorm2d(l2)
        self.actv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        ### 3rd ###
        self.conv3 = nn.Conv2d(l2,l3,kernel_size=2,stride=2)
        self.norm3 = Lambda(lambda x:x)
        self.actv3 = nn.Tanh()
        self.pool3 = nn.MaxPool2d(kernel_size=(1,1))
        self.adapool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()


    def forward(self,X):
        X = self.pool1(self.actv1(self.norm1(self.conv1(X))))
        X = self.pool2(self.actv2(self.norm2(self.conv2(X))))
        X = self.pool3(self.actv3(self.norm3(self.conv3(X))))
        X = self.adapool(X)
        X = self.flatten(X)
        return X

class LSTM(nn.Module):
    def __init__(self,seq_size,feature_size):
        """
        2 layer LSTM model: feature_size --> 200 --> 3

        Arguments:
        seq_size (int): length of the sequence
        feature_size (int): feature size of each interval in the sequence
        """
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(feature_size,feature_size,num_layers=2,bidirectional=True)
        # self.lstm2 = nn.LSTM(feature_size,feature_size)


    def forward(self,X):
        X, (h_0,c_0) = self.lstm1(X)
        # X, (h_0,c_0) = self.lstm2(X, (h_0,c_0))
        X = torch.flatten(X,1)
        return X


class CNN_LSTM(nn.Module):

    def __init__(self,n_seq=25,n_feature=128,n_classes=10):
        super(CNN_LSTM, self).__init__()

        self.n_seq = n_seq
        self.n_feature = n_feature
        self.n_classes = n_classes
        self.den_input = self.n_seq*self.n_feature*2 # 2 for bidirectional

        self.cnn = SmallEncoder(32,64,self.n_feature)
        self.lstm = LSTM(self.n_seq,self.n_feature)
        self.dcn = nn.Sequential(nn.Linear(self.den_input,64),
                                 nn.ReLU(),
                                 nn.Linear(64,self.n_classes),
                                 nn.Softmax(1))

    def forward(self,X):
        imgsize = X.shape[2:]
        X = X.view(-1,*imgsize)
        X = self.cnn(X)
        X = X.view(-1,self.n_seq,self.n_feature)
        X = self.lstm(X)
        X = self.dcn(X)
        return X
