import numpy as np
import pandas as pd
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


class PreTraining(nn.Module):

    def __init__(self):
        super(PreTraining, self).__init__()
        pass

    def forward(self,x):
        return x


class SimCLR(nn.Module):
    """
    Modified from Spijkervet
    """
    def __init__(self, encoder, n_features, transformer):
        super(SimCLR, self).__init__()

        assert isinstance(n_features,list), 'n_features must be list'
        assert len(n_features) == 3, print(n_features)

        self.transformer = transformer
        self.encoder = encoder
        self.n_features = n_features

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features[0], self.n_features[1], bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features[1], self.n_features[2], bias=False),
        )

#     def forward(self, x_i, x_j):
    def forward(self, x):
        x_i, x_j = self.transformer(x)
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return z_i, z_j # h_i, h_j, z_i, z_j


class NT_Xent(nn.Module):
    """
    NT_Xent for SimCLR. Work created by Spijkervet
    """

    def __init__(self, batch_size, temperature, world_size=1):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        assert z_i.shape == z_j.shape
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
#         if self.world_size > 1:
#             z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)
#         print(z_i.shape,z.shape,sim.shape,sim_i_j.shape,sim_j_i.shape)
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            N, 1
        )
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).long().cuda()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


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

start = 1
end = 500
parallel = True
model, record = pretrain(model, train_loader, criterion, optimizer, end, start, test_loader=None, parallel=parallel)
