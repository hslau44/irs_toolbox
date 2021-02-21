import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader, random_split
from logger import logging
from torchvision.datasets.folder import ImageFolder

from data.raw_csi import import_clean_data
from data.spectrogram import import_data, import_pair_data, import_CsiPwr_data
from data.transformation import *
from data.utils import DatasetObject

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_data(modality,directory):
    # pair data
    if mode == 'single' or mode == 'double':
        if mode == 'single':
            X1,X2,y = import_pair_data(directory)
        elif mode == 'double':
            X1,X2,y = import_CsiPwr_data(directory)
        # processing
        y,lb = label_encode(y)
        X1 = X1.reshape(*X1.shape,1).transpose(0,3,1,2)
        X2 = X2.reshape(*X2.shape,1).transpose(0,3,1,2)
        return X1,X2,y,lb
    # single data
    elif mode == None:
        X,y = import_data(directory)
        X = X.reshape(*X.shape,1).transpose(0,3,1,2)
        y,lb = label_encode(y)
        return X,y,lb
    else:
        raise ValueError("Must be in {'single','double',None}")
    return

def create_dataloader(*arrays,label='None',batch_size=64,num_workers=0):
    tensors = [torch.Tensor(arr) for arr in arrays]
    if type(label)!=str:
        tensors.append(torch.Tensor(label).long())
    tensordataset = torch.utils.data.TensorDataset(*tensors)
    dataloader = torch.utils.data.DataLoader(tensordataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return dataloader


def prepare_single_source(directory,axis=3,train_size=0.8,resample=None,batch_size=64,num_workers=0):
    ### import data
    X,y = import_data(directory)
    ### add and transpose axis
    if axis:
        X = X.reshape(*X.shape,1)
    if axis == 3:
        X = X.transpose(0,3,1,2)
    ### label encode
    y,lb = label_encode(y)
    ### select training data
    X,y = shuffle(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=train_size)
    ### resample
    if resample:
        X_train, y_train = resampling(X_train, y_train, labels=y_train,oversampling=True)
        X_test, y_test = resampling(X_test, y_test, labels=y_test,oversampling=False)
    ### dataloader
    train_loader = create_dataloader(X_train,label=y_train,batch_size=batch_size,num_workers=num_workers)
    test_loader  = create_dataloader(X_test,label=y_test,batch_size=2000,num_workers=num_workers)
    ### Report
    print('X_train: ',X_train.shape,'\ty_train: ',y_train.shape,'\tX_test: ',X_test.shape,'\ty_test: ',y_test.shape)
    print("class: ",lb.classes_)
    return train_loader,test_loader,lb


def prepare_double_source(directory,modality='single',axis=1,train_size=0.8,joint='first',p=None,resample=None,batch_size=64,num_workers=0):
    """
    Import data with pair source, this is either csi-csi (multi-view) or csi-pwr (multi-modality)

    Argument:
    directory (str)
    modality (str):
        'single' for csi-csi
        'double' for csi-pwr
    axis (int): the axis of the channel for CNN

    """
    ### import data
    print('modality:',modality)
    if modality == 'single':
        X1,X2,y = import_pair_data(directory)
    elif modality == 'double':
        X1,X2,y = import_CsiPwr_data(directory)
        if joint not in ['first','second']:
            joint = 'first'
    else:
        raise ValueError("Must be in {'single','double'}")
    ### add and transpose axis
    if axis:
        X1 = X1.reshape(*X1.shape,1)
        X2 = X2.reshape(*X2.shape,1)
    if axis == 1:
        X1 = X1.transpose(0,3,1,2)
        X2 = X2.transpose(0,3,1,2)
    ### label encode
    y,lb = label_encode(y)
    ### select training data
    X1,X2,y = shuffle(X1,X2,y)
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1,X2,y,train_size=train_size)
    if joint == 'joint':
        X_train = np.concatenate((X1_train,X2_train))
        y_train = np.concatenate((y_train,y_train))
        X_test = np.concatenate((X1_test,X2_test))
        y_test = np.concatenate((y_test,y_test))
    elif joint == 'first':
        X_train = X1_train
        y_train = y_train
        X_test = X1_test
        y_test = y_test
    elif joint == 'second':
        X_train = X2_train
        y_train = y_train
        X_test = X2_test
        y_test = y_test
    else:
        raise ValueError("Must be in {'joint','first','second'}")
    X_train,y_train = shuffle(X_train,y_train)
    X_test,y_test = shuffle(X_test,y_test)
    ### select finetune data
    if p:
        X_train, _ , y_train, _ = train_test_split(X_train,y_train,train_size=p)
    ### resample
    if resample:
        X_train, y_train = resampling(X_train, y_train, labels=y_train,oversampling=True)
        X_test, y_test = resampling(X_test, y_test, labels=y_test,oversampling=False)
    ### dataloader
    pretrain_loader = create_dataloader(X1_train,X2_train,batch_size=batch_size,num_workers=num_workers)
    finetune_loader = create_dataloader(X_train,label=y_train,batch_size=batch_size,num_workers=num_workers)
    validatn_loader  = create_dataloader(X_test,label=y_test,batch_size=2000,num_workers=num_workers)
    ### Report
    print('X1_train: ',X1_train.shape,'\tX2_train: ',X2_train.shape)
    print('X_train: ',X_train.shape,'\ty_train: ',y_train.shape,'\tX_test: ',X_test.shape,'\ty_test: ',y_test.shape)
    print("class: ",lb.classes_)
    return pretrain_loader, finetune_loader, validatn_loader, lb
