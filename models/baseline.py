import torch
import torch.nn as nn
from torch.nn import functional as F
# from .utils import Lambda

class Lambda(nn.Module):
    """
    transform tensor according to function func
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class V1(nn.Module):
    """
    The baseline encoder take a batch of CSI spectrogram (n,1,num_frame,num_channel) and output flatten latent feature
    """
    def __init__(self):
        super(V1, self).__init__()
        ### 1st ###
        self.conv1 = nn.Conv2d(1,64,kernel_size=5,stride=5)
        self.actv1 = nn.ReLU()
        self.norm1 = Lambda(lambda x:x)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,1))
        ### 2nd ###
        self.conv2 = nn.Conv2d(64,128,kernel_size=3,stride=3)
        self.actv2 = nn.ReLU()
        self.norm2 = Lambda(lambda x:x)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,1))
        ### 3rd ###
        self.conv3 = nn.Conv2d(128,256,kernel_size=2,stride=2)
        self.actv3 = nn.ReLU()
        self.norm3 = Lambda(lambda x:x)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,1))

    def forward(self,X):
        X = self.pool1(self.norm1(self.actv1(self.conv1(X))))
        X = self.pool2(self.norm2(self.actv2(self.conv2(X))))
        X = self.pool3(self.norm3(self.actv3(self.conv3(X))))
        X = torch.flatten(X, 1)
        return X

class V2(nn.Module):
    """
    The baseline encoder take a batch of CSI spectrogram (n,1,num_frame,num_channel) and output flatten latent feature
    """
    def __init__(self):
        super(V2, self).__init__()
        self.norm0 = nn.BatchNorm2d(1)
        ### 1st ###
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=(2,2))
        self.actv1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2))
        ### 2nd ###
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=(2,2))
        self.actv2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2))
        ### 3rd ###
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,stride=(2,2))
        self.actv3 = nn.ReLU()
        self.norm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(1,2))
        ### 4rd ###
        self.conv4 = nn.Conv2d(128,256,kernel_size=3,stride=(2,2))
        self.actv4 = nn.ReLU()
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

class Classifier(nn.Module):
    """
    THe baseline classifier with hidden layer {128,8}
    """
    def __init__(self,input_shape):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(input_shape,128)
        self.linear2 = nn.Linear(128,8)

    def forward(self,X):
        X = F.dropout(F.leaky_relu(self.linear1(X)),0.1)
        X = self.linear2(X)
        return X


class CNN_module(nn.Module):
    """
    Combine encoder and decoder to form CNN
    """
    def __init__(self, encoder, decoder):
        super(CNN_module, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X):
        X = self.encoder(X)
        X = self.decoder(X)
        return X


class Autoencoder(nn.Module):

    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
