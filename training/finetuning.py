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
        hidden_layer = kwargs.get('hidden_layer',128)
        # build and load model
        encoder,in_size = encoder_builder()
        if model_path: 
            encoder.load_state_dict(torch.load(model_path))
            encoder = freeze_network(encoder)
        if hidden_layer:
            decoder = Classifier(in_size,hidden_layer,n_classes)
        else:
            decoder = Classifier(in_size,n_classes)
        # overall
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,X):
        X = self.encoder(X)
        X = self.decoder(X)
        return X
