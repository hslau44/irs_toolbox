import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet18, vgg16
from functools import partial
from models import Lambda, Stack, Flatten

def resnet_finetune(model, n_classes):
    """
    This function prepares resnet to be finetuned by:
    1) freeze the model weights
    2) cut-off the last layer and replace with a new one with the correct classes number
    """
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(512, n_classes)
    return model


class Attention(nn.Module):
    def __init__(self,feature_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(feature_size,1)

    def forward(self,X):
        assert len(X.shape) == 3
        a = self.linear(X)
        a = torch.relu(a)
        a = F.softmax(a,dim=1)
        return a*X

def create_vgg16(output_size=(2,2)):
    """
    VGG 16 for 1 channel image, output: 512*output_size
    """
    mdl = vgg16()
    model = torch.nn.Sequential(Stack(),
                                *(list(mdl.children())[:-2]),
                                nn.AdaptiveAvgPool2d(output_size),
                                Flatten())
    return model


def create_vgg16_atn(output_size=(2,2)):
    """
    VGG 16 for 1 channel image, with linear attention, output: 512*output_size
    """
    mdl = vgg16()
    model = torch.nn.Sequential(Stack(),
                                *(list(mdl.children())[:-2]),
                                nn.AdaptiveAvgPool2d(output_size),
                                Flatten(2),
                                Attention(output_size[0]*output_size[1]),
                                Flatten(1)
                                )
    return model

# resnet18 = partial(resnet_finetune, resnet18(pretrained=True))


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
