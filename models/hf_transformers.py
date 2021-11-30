import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureExtractor,Wav2Vec2FeatureProjection
from transformers import Wav2Vec2Config,Wav2Vec2Model,Wav2Vec2ForPreTraining,Wav2Vec2ForSequenceClassification



class Sig2VecConfig(Wav2Vec2Config):
    f"""
    This is the configuration class on top of the transformers.Wav2Vec2Config configuration class

    Args:
        in_channels (int)

    Here are the original docstring:

    {Wav2Vec2Config().__doc__}
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.in_channels = kwargs.get("in_channels",1)


class Identity(nn.Identity):

    def __init__(self, config):
        super().__init__()

    def forward(self,X):
        return X

    def _freeze_parameters(self):
        return


class Sig2VecFeatureExtractor(Wav2Vec2FeatureExtractor):

    def __init__(self, config):
        super().__init__(config)
        self.conv_layers[0].conv = nn.Conv1d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=config.conv_kernel,
            stride = config.conv_stride,
            bias = config.conv_bias
            )


    def forward(self, hidden_states):

        # make sure hidden_states require grad for gradient_checkpointing
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(conv_layer),
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)

        return hidden_states

class DummyFeatureProjection(Wav2Vec2FeatureProjection):

    def __init__(self, config):
        super().__init__(config)
        # self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.projection = Identity()

    def forward(self,X):
        X = X.transpose(1, 2)
        hidden_states, norm_hidden_states = super().forward(X)
        return hidden_states, norm_hidden_states


class Sig2VecModel(Wav2Vec2Model):

    def __init__(self, config):
        super().__init__(config)
        self.feature_extractor = Sig2VecFeatureExtractor(config)
        # self.feature_projection = DummyFeatureProjection(config)


class Sig2VecForPreTraining(Wav2Vec2ForPreTraining):

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Sig2VecModel(config)

class Sig2VecForSequenceClassification(Wav2Vec2ForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Sig2VecModel(config)

class Sig2VecForSequenceClassificationPT(Wav2Vec2ForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Sig2VecModel(config)

    def forward(self,X):
        X = super().forward(input_values=X)
        return X['logits']
