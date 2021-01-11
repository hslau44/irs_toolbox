import torch
import torch.nn as nn
from torch.nn import functional as F


class LSTM_Baseline(nn.Module)::
    def __init__(self,seq_size,feature_size):
        """
        Baseline LSTM model

        attr:
        seq_size: length of the sequence
        feature_size: feature size of each interval in the sequence

        """
        super(LSTM_Baseline, self).__init__()
        self.lstm1 = nn.LSTM(feature_size,200)
        self.lstm2 = nn.LSTM(200,3)
        self.linear1 = nn.Linear(3*seq_size,30)
        self.linear2 = nn.Linear(30,8)

    def forward(self,X):
        X, _ = self.lstm1(X)
        X, _ = self.lstm2(X)
        X = torch.flatten(X,1)
        X = self.linear1(X)
        X = self.linear2(X)
        return F.log_softmax(X,dim=1)
