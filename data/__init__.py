import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from data.raw_csi import import_clean_data
from data.utils import DatasetObject
from data import process_data
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader, random_split
from logger import logging
from torchvision.datasets.folder import ImageFolder



def create_dataloaders(X_train, y_train, X_test, y_test, train_batch_sizes=64, test_batch_sizes=200):
    traindataset = TensorDataset(Tensor(X_train),Tensor(y_train).long())
    testdataset = TensorDataset(Tensor(X_test), Tensor(y_test).long())
    train_loader = DataLoader(traindataset, batch_size=train_batch_sizes, shuffle=True, num_workers=1, drop_last=True)
    test_loader = DataLoader(testdataset, batch_size=test_batch_sizes, shuffle=True, num_workers=1)
    return train_loader, test_loader
