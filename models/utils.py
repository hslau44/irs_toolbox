import torch
import torch.nn as nn
from torch.nn import functional as F

class Flatten(nn.Module):

    def __init__(self, start_dim=1):
        """nn.Module implementation of torch.Flatten"""
        super().__init__()
        self.start_dim = start_dim

    def forward(self, input):
        return torch.flatten(input,start_dim=self.start_dim)

class Stack(nn.Module):

    def __init__(self):
        """Increase the number of channel from 1 to 3 by stacking extra 2 sample on the 1st axis"""
        super().__init__()
        pass

    def forward(self, x):
        return torch.cat((x,x,x),axis=1)

class Lambda(nn.Module):
    """transform tensor according to function func

    Argument:
    func (function): the mathematical operation applying in a tensor, accept only one input tensor
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class ED_module(nn.Module):
    """
    Combine Encoder and Decoder to form the model
    """
    def __init__(self, encoder, decoder):
        super(ED_module, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X):
        X = self.encoder(X)
        X = self.decoder(X)
        return X


class Classifier(nn.Module):
    """
    Classifier with LeakyReLU and Dropout between the layers
    """
    def __init__(self,*num_neurons,**kwargs):
        super(Classifier, self).__init__()
        assert len(num_neurons) > 1
        self.bias = kwargs.get('bias',True)
        self.dropout = kwargs.get('dropout',0.1)
        self.last_activation = kwargs.get('last_activation',None)
        self.num_neurons = num_neurons
        self.model = None
        self.build()

    def forward(self,X):
        return self.model(X)

    def build(self):
        layers = []
        for i in range(len(self.num_neurons)-1):

            layers.append(nn.Linear(in_features=self.num_neurons[i],
                                    out_features=self.num_neurons[i+1],
                                    bias=self.bias))

            if i < len(self.num_neurons)-2:

                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(p=self.dropout))

        if self.last_activation:
            layers.append(self.last_activation)

        self.model = nn.Sequential(*layers)
        return
