import torch
from torch import nn
from torch.nn import functional as F
from models.transformer import *

class PreTraining(object):

    def __init__(self,encoder_builder,seq_len,embed_dim,depth,n_heads,qkv_bias=False,attn_p=0.0,p=0.0,mlp_ratio=4.0,**kwargs):
        self.source_encoder =  encoder_builder()
        self.target_encoder =  kwargs.get('encoder_builder2',encoder_builder)()
        self.st_transformer = EncoderDecoderTransformer(
            seq_len=seq_len,
            embed_dim=embed_dim,
            depth=depth,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            p=p,
            mlp_ratio=mlp_ratio
        )
        optim = kwargs.get('optimizer',torch.optim.Adam)
        lr = kwargs.get('lr',0.001)

        self.optimizer = optim(
            list(self.source_encoder.parameters()) +
            list(self.target_encoder.parameters()) +
            list(self.st_transformer.parameters()), lr=lr)

        self.criterion = kwargs.get('criterion',nn.MultiLabelSoftMarginLoss())

    def train(self,train_loader,epochs=250,verbose=True,rtn_history=True,device=None):

        history = {'loss':[]}

        if device:
            self.source_encoder = self.source_encoder.to(device)
            self.target_encoder = self.target_encoder.to(device)
            self.st_transformer = self.st_transformer.to(device)
            self.criterion = self.criterion.to(device)

        for i in range(epochs):
            if verbose: print(f'Epoch {i+1} ',end='')
            for items in train_loader:

                if device:
                    X1,X2,y = [i.to(device) for i in items]
                del items

                X1 = self.source_encoder(X1)
                X2 = self.target_encoder(X2)
                targets = X2[:,1:].detach()

                outputs = self.st_transformer(X1,X2)
                outputs = outputs[:,1:-1]

                loss = self.criterion(outputs,targets)
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
            self.source_encoder = self.source_encoder.cpu()
            self.target_encoder = self.target_encoder.cpu()
            self.st_transformer = self.st_transformer.cpu()
            self.criterion = self.criterion.cpu()

        if rtn_history:
            return self.rtn_model(),history
        else:
            return self.rtn_model()

    def rtn_model(self):
        transformer = self.st_transformer.encoder
        transformer.end = True
        return TransformerWrapper(self.source_encoder,transformer)
