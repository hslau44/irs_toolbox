import torch
import torch.nn as nn
from torch.nn import functional as F

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
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



class Classifier(nn.Module):
    def __init__(self,input_shape):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(input_shape,128)
        self.linear2 = nn.Linear(128,8)

    def forward(self,X):
        X = F.dropout(F.leaky_relu(self.linear1(X)))
        X = self.linear2(X)
        return F.log_softmax(X,dim=0)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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
        ### Classifier ####
        self.linear1 = nn.Linear(768,128)
        self.linear2 = nn.Linear(128,8)

    def forward(self,X):
        X = self.pool1(self.norm1(self.actv1(self.conv1(X))))
        X = self.pool2(self.norm2(self.actv2(self.conv2(X))))
        X = self.pool3(self.norm3(self.actv3(self.conv3(X))))
        X = torch.flatten(X, 1)
        X = F.dropout(F.leaky_relu(self.linear1(X)))
        X = self.linear2(X)
        return F.log_softmax(X,dim=0)
