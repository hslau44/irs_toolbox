import torch
import torch.nn as nn
from torch.nn import functional as F
# from .utils import Lambda
from models import Lambda


class V1(nn.Module):
    """
    The baseline encoder take a batch of CSI spectrogram (n,1,channel,timestamp) and output flatten latent feature,
    channel = 3*30
    timestamp > 900

    """
    def __init__(self):
        super(V1, self).__init__()
        ### 1st ###
        self.conv1 = nn.Conv2d(1,64,kernel_size=5,stride=5)
        self.norm1 = nn.BatchNorm2d(64)
        self.actv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(1,2))
        ### 2nd ###
        self.conv2 = nn.Conv2d(64,128,kernel_size=3,stride=3)
        self.norm2 = nn.BatchNorm2d(128)
        self.actv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=(1,2))
        ### 3rd ###
        self.conv3 = nn.Conv2d(128,256,kernel_size=2,stride=2)
        self.norm3 = Lambda(lambda x:x)
        self.actv3 = nn.Tanh()
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=(1,2))

    def forward(self,X):
        X = self.pool1(self.actv1(self.norm1(self.conv1(X))))
        X = self.pool2(self.actv2(self.norm2(self.conv2(X))))
        X = self.pool3(self.actv3(self.norm3(self.conv3(X))))
        X = torch.flatten(X, 1)
        return X

class V2(nn.Module):
    """
    The 4 layers encoder take a batch of CSI spectrogram (n,1,num_frame,num_channel) and output flatten latent feature
    output size = 2304
    """
    def __init__(self):
        super(V2, self).__init__()
        self.norm0 = Lambda(lambda x:x)
        ### 1st ###
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=(2,2))
        self.actv1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2))
        ### 2nd ###
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=(2,2))
        self.actv2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=(1,2))
        ### 3rd ###
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,stride=(2,2))
        self.actv3 = nn.ReLU()
        self.norm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=(1,2))
        ### 4rd ###
        self.conv4 = nn.Conv2d(128,256,kernel_size=3,stride=(2,2))
        self.actv4 = nn.Tanh()
        self.norm4 = Lambda(lambda x:x)
        self.pool4 = nn.AdaptiveAvgPool2d(output_size=(3,3))
        ### 5rd ###
        # self.conv5 = nn.Conv2d(256,512,kernel_size=2,stride=(2,2))
        # self.actv5 = nn.ReLU()
        # self.norm5 = Lambda(lambda x:x)
        # self.pool5 = Lambda(lambda x:x)

    def forward(self,X):
        X = self.norm0(X)
        X = self.pool1(self.actv1(self.norm1(self.conv1(X))))
        X = self.pool2(self.actv2(self.norm2(self.conv2(X))))
        X = self.pool3(self.actv3(self.norm3(self.conv3(X))))
        X = self.pool4(self.actv4(self.norm4(self.conv4(X))))
        # X = self.pool5(self.actv5(self.norm5(self.conv5(X))))
        X = torch.flatten(X, 1)
        return X


class LSTM(nn.Module)::
    def __init__(self,seq_size,feature_size):
        """
        2 layer LSTM model: feature_size --> 200 --> 3

        attr:
        seq_size: length of the sequence
        feature_size: feature size of each interval in the sequence
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
    """
    Encoder for spectrogram (1,65,65), 3 layer
    """
    def __init__(self,num_filters):
        super(Encoder, self).__init__()
        l1,l2,l3 = num_filters
        ### 1st ###
        self.conv1 = nn.Conv2d(1,l1,kernel_size=5,stride=2)
        self.norm1 = nn.BatchNorm2d(l1)
        self.actv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        ### 2nd ###
        self.conv2 = nn.Conv2d(l1,l2,kernel_size=3,stride=2)
        self.norm2 = nn.BatchNorm2d(l2)
        self.actv2 = nn.ReLU()
        self.pool2 = Lambda(lambda x:x) # nn.MaxPool2d(kernel_size=(2,2))
        ### 3rd ###
        self.conv3 = nn.Conv2d(l2,l3,kernel_size=2,stride=2)
        self.norm3 = Lambda(lambda x:x)
        self.actv3 = nn.Tanh()
        self.pool3 = nn.AvgPool2d(kernel_size=(2,2))

    def forward(self,X):
        X = self.pool1(self.actv1(self.norm1(self.conv1(X))))
        X = self.pool2(self.actv2(self.norm2(self.conv2(X))))
        X = self.pool3(self.actv3(self.norm3(self.conv3(X))))
        X = torch.flatten(X, 1)
        # print(X.shape)
        return X
