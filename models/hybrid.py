import torch
import torch.nn as nn
from torch.nn import functional as F



class SI_Encoder(nn.Module):
    """
    Single Tx-Rx pair , 4 layers
    """
    def __init__(self):
        super(SI_Encoder, self).__init__()
        ### 1st ###
        self.conv1 = nn.Conv2d(1,32,kernel_size=(3,3),stride=(1,1),padding=0)
        self.actv1 = nn.ReLU()
        self.norm1 = nn.InstanceNorm2d(32)
        self.pool1 = nn.AdaptiveMaxPool2d((20,300))
        ### 2nd ###
        self.conv2 = nn.Conv2d(32,64,kernel_size=(3,3),stride=(1,1),padding=0)
        self.actv2 = nn.ReLU()
        self.norm2 = nn.InstanceNorm2d(64)
        self.pool2 = nn.AdaptiveMaxPool2d((10,100))
        ### 3rd ###
        self.conv3 = nn.Conv2d(64,128,kernel_size=(3,3),stride=(1,1),padding=0)
        self.actv3 = nn.ReLU()
        self.norm3 = nn.InstanceNorm2d(128)
        self.pool3 = nn.AdaptiveMaxPool2d((5,30))
        ### 4rd ###
        self.conv4 = nn.Conv2d(128,256,kernel_size=(3,3),stride=(1,1),padding=0)
        self.actv4 = nn.Tanh()
        self.norm4 = Lambda(lambda x:x)
        self.pool4 = nn.AdaptiveAvgPool2d((1,3))

    def forward(self,X):
        X = self.pool1(self.actv1(self.norm1(self.conv1(X))))
        X = self.pool2(self.actv2(self.norm2(self.conv2(X))))
        X = self.pool3(self.actv3(self.norm3(self.conv3(X))))
        X = self.pool4(self.actv4(self.norm4(self.conv4(X))))
        X = torch.flatten(X, 1)
        return X




class MISO_cat(nn.Module):
    """

    This model first divides a spectrogram into number of pieces based on window and channel,
    each piece is processed with the single encoder and output a matrix with shape (batch_size, seq_size, num_features),
    input shape: (batch_size,num_frame,num_channel)
    """
    def __init__(self, encoder):
        super(MISO_cat, self).__init__()
        self.encoder = encoder

    def forward(self,x):
        ### Method 1 ###
        # size = x.shape
        # x = x.view(-1,1,self.channel,self.timestamp)
        # x = self.encoder(x)
        # x = x.view(size[0],-1)
        ### Method 2 ###
        a,b,c = torch.split(x,1,dim=1)
        a = self.encoder(a)
        b = self.encoder(b)
        c = self.encoder(c)
        x = torch.cat([a,b,c],dim=1)
        return x

class MISO_avg(nn.Module):
    """

    This model first divides a spectrogram into number of pieces based on window and channel,
    each piece is processed with the single encoder and output a matrix with shape (batch_size, seq_size, num_features),
    input shape: (batch_size,num_frame,num_channel)
    """
    def __init__(self, encoder):
        super(MISO_avg, self).__init__()
        self.encoder = encoder


    def forward(self,x):
        ### Method 2 ###
        a,b,c = torch.split(x,1,dim=1)
        a = self.encoder(a)
        b = self.encoder(b)
        c = self.encoder(c)
        x = (a + b + c)/3
        return x
