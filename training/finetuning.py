import torch
from torch import nn
from models.utils import Classifier

def freeze_network(model):
    """freeze all trainable parameter in model, return model"""
    for _, p in model.named_parameters():
        p.requires_grad = False
    return model

class FineTuneCNN(nn.Module):

    def __init__(self,encoder_builder,model_path,n_classes,**kwargs):
        super(FineTuneCNN, self).__init__()
        # parameters
        self.encoder_builder = encoder_builder
        self.model_path = model_path
        self.n_classes = n_classes
        self.hidden_layer = kwargs.get('hidden_layer',128)
        self.encoder = None
        self.decoder = None
        # build and load model
        self.build()

    def forward(self,X):
        X = self.encoder(X)
        X = self.decoder(X)
        return X

    def build(self):
        encoder,in_size = self.encoder_builder()
        if self.model_path:
            encoder.load_state_dict(torch.load(self.model_path))
            encoder = freeze_network(encoder)
        if self.hidden_layer:
            decoder = Classifier(in_size,self.hidden_layer,self.n_classes)
        else:
            decoder = Classifier(in_size,self.n_classes)
        self.encoder = encoder
        self.decoder = decoder
        return
