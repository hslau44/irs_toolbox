import torch
import torch.nn as nn
from torch.nn import functional as F
# from .utils import Lambda

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)



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



class Residual_Block(nn.Module):
    """
    Single channel for radio image (30,30,1)
    """
    def __init__(self,in_feature,out_feature,kernel_size,stride, padding='zeros', bias=False):
        super(Residual_Block, self).__init__()
        ### 1st ###
        self.conv1 = nn.Conv2d(in_feature,out_feature,kernel_size=kernel_size,stride=stride, bias=bias)
        self.norm1 = nn.BatchNorm2d(out_feature)
        self.actv1 = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=kernel_size,stride=stride)
        self.covr1 = nn.Conv2d(in_feature,out_feature,kernel_size=1,stride=1, bias=False)
        ### 2nd ###
        self.conv2 = nn.Conv2d(out_feature,out_feature,kernel_size=kernel_size,stride=stride, bias=bias)
        self.norm2 = nn.BatchNorm2d(out_feature)
        self.actv2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=kernel_size,stride=stride)

        self.covr1.weight.requires_grad = False

    def forward(self,X):
        R = X
        ### 1st ###
        X = self.actv1(self.norm1(self.conv1(X)))
        R = self.covr1(self.pool1(R))
        ### 2nd ###
        X = self.norm2(self.conv2(X))
        R = self.pool2(R)
        # print(X.shape,R.shape)
        X += R
        X = self.actv2(X)
        return X

class Residual(nn.Module):

    def __init__(self):
        super(Residual, self).__init__()
        self.norm = nn.BatchNorm2d(1)
        self.block1 = Residual_Block(in_feature=  1, out_feature=  64, kernel_size = (5,5), stride = (1,3))
        self.block2 = Residual_Block(in_feature= 64, out_feature= 128, kernel_size = (4,4), stride = (1,2))
        self.block3 = Residual_Block(in_feature=128, out_feature= 256, kernel_size = (3,3), stride = (2,2))
        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self,x):
        x = self.norm(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(self.pool(x), 1)
        return x


class Encoder_View(nn.Module):
    """

    This model first divides a spectrogram into number of pieces based on window and channel,
    each piece is processed with the single encoder and output a matrix with shape (batch_size, seq_size, num_features),
    input shape: (batch_size,num_frame,num_channel)
    """
    def __init__(self, encoder=None):
        super(Encoder_View, self).__init__()
        self.encoder = encoder

    def forward(self,x):
        ### Method 1 ###
#         size = x.shape
#         x = x.view(-1,1,30,900)
#         x = self.encoder(x)
#         x = x.view(size[0],-1)
        ### Method 2 ###
        a,b,c = torch.split(x,1,dim=1)
        a = self.encoder(a)
        b = self.encoder(b)
        c = self.encoder(c)
        x = torch.cat([a,b,c],dim=1)
        return x


class CNN_module(nn.Module):
    def __init__(self, encoder, decoder):
        super(CNN_module, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X):
        X = self.encoder(X)
        X = self.decoder(X)
        return X
