import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# define custom losses

class Contrastive_multiview_loss(nn.Module):

    def __init__(self,loss_func):
        """
        Simple implemntation of Full Graph Contrastive Multiview Coding (Tian et.al. 2020)

        Attribute:
        loss_func (torch.nn.Module): the contrastive loss function
        """
        super(Contrastive_multiview_loss, self).__init__()
        self.loss_func = loss_func

    def forward(self,*vecs):
        loss = sum([self.loss_func(vecs[i],vecs[j]) for i in range(0,len(vecs)-1) for j in range(i+1,len(vecs))])
        return loss



class NT_Xent(nn.Module):


    def __init__(self,batch_size,num_repre=2,temperature=0.1):
        """
        NT_Xent, support for num_repre (number of representation) >= 2

        Attributes:
        batch_size (int): batch size
        num_repre (int): number of representation
        temperature (float): smoothness

        """
        super(NT_Xent, self).__init__()
        if num_repre < 2:
            raise ValueError('num_repre must be >=2')
        self.batch_size = batch_size
        self.num_repre = num_repre
        self.mtx_size = batch_size*num_repre
        self.positive_mask = self.generate_mask(1)
        self.negative_mask = self.generate_mask(0)
        self.temperature = temperature
        self.xent = nn.CrossEntropyLoss(reduction="sum")
        self.cossim = nn.CosineSimilarity(dim=2)


    def forward(self,*tensors):
        """
        Argument
        tensors (torch.Tensor):
        """
        assert tensors[0].shape[0] == self.batch_size,f'batch size not matching, expecting size: {self.batch_size}, get batch size {tensors[0].shape[0]}'
        assert len(tensors) == self.num_repre,f'number of representation not matching, expecting size: {self.num_repre}, get {len(tensors)} '
        z = torch.cat(tensors, dim=0)
        s = self.cossim(z.unsqueeze(1),z.unsqueeze(0))/self.temperature
        positive_samples = s[self.positive_mask]
        positive_samples = positive_samples.reshape(self.mtx_size,-1)
        positive_samples = torch.cat(torch.split(positive_samples,1,1))
        # print(f'positive shape {positive_samples.shape}')
        negative_samples = s[self.negative_mask]
        negative_samples = negative_samples.reshape(self.mtx_size,-1)
        negative_samples = negative_samples.repeat(self.num_repre-1,1)
        # print(f'negative shape {negative_samples.shape}')
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        labels = torch.zeros(logits.shape[0]).long().to(logits.device)
        # print(f'logit shape {logits.shape}')
        loss = self.xent(logits, labels)
        loss /= self.mtx_size
        return loss

    def generate_mask(self,mask_type=0):
        if mask_type != 1 and mask_type != 0:
            raise ValueError('Mask_type must be equal {0,1}')
        if mask_type:
            mask = torch.zeros((self.mtx_size, self.mtx_size), dtype=bool)
        else:
            mask = torch.ones((self.mtx_size, self.mtx_size), dtype=bool)
            mask = mask.fill_diagonal_(0)
        for i in range(self.mtx_size):
            for j in range(i,self.mtx_size,self.batch_size):
                if i != j:
                    mask[i,j] = mask_type
                    mask[j,i] = mask_type
        return mask
