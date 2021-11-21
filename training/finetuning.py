import torch
from torch import nn
from models.utils import Classifier

def freeze_network(model):
    """freeze all trainable parameter in model, return model"""
    for _, p in model.named_parameters():
        p.requires_grad = False
    return model

class FineTuneCNN(nn.Module):
    """
    Encoder Decoder Architecture, build encoder and add MLP

    Args:
        encoder_builder (func): callable function of the primary encoder and its latent size (tuple <nn.Module, int>)
        model_path (str): Optional, load path model's state dict to the encoder, must be consistent with encoder_builder.
        n_classes (int): number of classes / output size
    kwargs:
        hidden_layer (int/bool): hidden layer of the mlp. If None, mlp become Linear Classifier
        freeze_encoder (int/bool): freeze the encoder, default value True

    Method:
        build: build/reset the model
    """
    def __init__(self,encoder_builder,model_path,n_classes,**kwargs):
        super(FineTuneCNN, self).__init__()
        # parameters
        self.encoder_builder = encoder_builder
        self.model_path = model_path
        self.n_classes = n_classes
        self.hidden_layer = kwargs.get('hidden_layer',128)
        self.encoder = None
        self.decoder = None
        self.freeze_encoder = kwargs.get('freeze_encoder',True)
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
        if self.freeze_encoder:
            encoder = freeze_network(encoder)
        if self.hidden_layer:
            decoder = Classifier(in_size,self.hidden_layer,self.n_classes)
        else:
            decoder = Classifier(in_size,self.n_classes)
        self.encoder = encoder
        self.decoder = decoder
        return
