import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
import torchvision


class DataAugmentation(nn.Module):

    def __init__(self, size):
        super(DataAugmentation, self).__init__()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=size),
#             torchvision.transforms.ToTensor()
        ])

    def forward(self, x):
        return self.transform(x), self.transform(x)



class SimCLR_TXR_model(nn.Module):
    def __init__(self, encoder, decoder):
        super(SimCLR_TXR_model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x):
        a,b,c = torch.split(x,1,dim=1)
        a = self.encoder(a)
        b = self.encoder(b)
        c = self.encoder(c)
        a = self.decoder(a)
        b = self.decoder(b)
        c = self.decoder(c)
        return a,b,c


# ----------------------------------------------------- Helper --------------------------------------------------------

def build_simclr():
    encoder = Encoder()
    dataug = DataAugmentation(size=(1000,90))
    n_features = [1024,128,8]
    simclr = SimCLR(encoder=encoder, n_features=n_features, transformer=dataug)
    return simclr,dataug

def setting_(model):
    criterion = NT_Xent(batch_size=128,temperature=0.1)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.003)
    return criterion, optimizer


# -----------------------------------------------training----------------------------------

def pretrain(model, train_loader, criterion, optimizer, end, start = 1, test_loader = None, auto_save = None, parallel = None, **kwargs):

    # Check device setting
    if parallel == True:
        print('GPU')
        model = model.to(device)
        criterion = criterion.to(device)
    else:
        print('CPU')
        model = model.cpu()

    print('Start Training')
    record = {'train':[],'validation':[]}
    i = start
    #Loop
    while i <= end:
        print(f"Epoch {i}: ", end='')
        for b, (X_train, _ ) in enumerate(train_loader):
            if parallel == True:
                X_train = X_train.to(device)
            print(f">", end='')
            optimizer.zero_grad()
            a,b,c = model(X_train)
            loss = criterion(a,b,c)
            loss.backward()
            optimizer.step()
            del a,b,c
        # One epoch completed
        loss = loss.tolist()
        record['train'].append(loss)
        print(f' loss: {loss} ',end='')
        if (test_loader != None) and i%100 ==0 :
            acc = short_evaluation(model,test_loader,parallel)
            record['validation'].append(acc)
            print(f' accuracy: {acc}')
        else:
            print('')
        i += 1
    model = model.cpu()
    return model, record

# start = 1
# end = 500
# parallel = True
# model, record = pretrain(model, train_loader, criterion, optimizer, end, start, test_loader=None, parallel=parallel)
