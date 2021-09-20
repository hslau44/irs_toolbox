import torch
from torch import nn
from torch.nn import functional as F
from models.utils import Classifier
from losses import NT_Xent,SupConLoss


class Simclr(nn.Module):
    """
    Under testing
    """

    def __init__(self,enc,enc2,size,size2,**kwargs):
        super(Simclr, self).__init__()
        p_dim = kwargs.get('projection_dim',128)
        p_depth = kwargs.get('projection_depth',2)
        self.encoder = enc
        self.decoder = Classifier(size,*[p_dim for i in range(p_depth)])
        self.encoder2 = enc2
        self.decoder2 = Classifier(size2,*[p_dim for i in range(p_depth)])

    def forward(self,X1,X2):
        X1 = self.encoder(X1)
        X1 = self.decoder(X1)
        X1 = torch.nn.functional.normalize(X1,dim=1)
        X2 = self.encoder2(X2)
        X2 = self.decoder2(X2)
        X2 = torch.nn.functional.normalize(X2,dim=1)
        return X1,X2





class Contrastive_PreTraining(object):
    """
    Args:
    encoder_builder (func): callable function of the primary encoder (torch.nn.Module)
    batch_size (int): batch size

    kwargs:
    encoder_builder2 (func): callable function of the secondary encoder (torch.nn.Module)
    temperature (float): temperature of NT-Xent
    optimizer (func): callable function of optimizer (torch.optim.Optimizer)
    supervision (bool): trained with label with Supervised Contrastive Learning (Tian 2020)
    """

    def __init__(self,encoder_builder,batch_size,supervision=None,**kwargs):
        # model
        enc,size = encoder_builder()
        enc2,size2 = kwargs.get('encoder_builder2',encoder_builder)()
        # criterion
        temperature = kwargs.get('temperature',0.1)
        # optim
        optim = kwargs.get('optimizer',torch.optim.Adam)
        lr = kwargs.get('lr',0.001)
        # overall
        self.model = Simclr(enc,enc2,size,size2,**kwargs)
        self.optimizer = optim(list(self.model.parameters()), lr=lr)
        self.supervision = supervision

        # if self.supervision:
        self.criterion = SupConLoss(temperature=temperature,base_temperature=1)
        # else:
            # self.criterion = NT_Xent(batch_size,num_repre=2,temperature=temperature)

    def train(self,train_loader,epochs=250,verbose=True,rtn_history=True,device=None):
        """
        Return trained encoder (and history if rtn_history = True)

        Args:
        train_loader (torch.utils.data.dataloader.DataLoader) - the pair dataset
        epochs (int) - epochs
        verbose (bool) - verbose
        rtn_history (bool) - return both the encoder and history
        device (torch.device) - model to be trained on

        Return
        """
        history = {'loss':[]}
        torch.optim.Optimizer
        if device:
            self.model = self.model.to(device)
            self.criterion = self.criterion.to(device)

        for i in range(epochs):
            if verbose: print(f'Epoch {i+1} ',end='')
            for items in train_loader:

                if device:
                    X1,X2,y = [i.to(device) for i in items]

                self.optimizer.zero_grad()
                X1,X2 = self.model(X1,X2)
                tensor = torch.cat([X1.unsqueeze(1),X2.unsqueeze(1)],dim=1)

                # SupConLoss
                if self.supervision:
                    loss = self.criterion(tensor,labels=y)
                # NT_Xent
                else:
                    loss = self.criterion(tensor)
                loss.backward()
                self.optimizer.step()

                X1 = X1.cpu()
                X2 = X2.cpu()
                y = y.cpu()
                del X1,X2,y
                if verbose: print('>',end='')

            loss = loss.tolist()
            history['loss'].append(loss)
            if verbose: print(f' loss: {loss}')

        if device:
            self.model = self.model.cpu()
            self.criterion = self.criterion.cpu()

        if rtn_history:
            return self.model.encoder,history
        else:
            return self.model.encoder
