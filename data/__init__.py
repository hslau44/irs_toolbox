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
from data.spectrogram import import_data, import_pair_data, import_dummies
from data.transformation import *
from data.utils import DatasetObject

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle




def create_dataloader(*arrays,label='None',batch_size=64,num_workers=0):
    tensors = [torch.Tensor(arr) for arr in arrays]
    if type(label)!=str:
        tensors.append(torch.Tensor(label).long())
    tensordataset = torch.utils.data.TensorDataset(*tensors)
    dataloader = torch.utils.data.DataLoader(tensordataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return dataloader




def process_lab_data(X1,X2,y,joint='first',sampling='weight',batch_size=64,num_workers=0,lb=None):
    X1 = X1.reshape(*X1.shape,1).transpose(0,3,1,2)
    X2 = X2.reshape(*X2.shape,1).transpose(0,3,1,2)
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1,X2,y,train_size=0.8,stratify=y)
    ### pretrain_loader
    pretrain_loader = create_dataloader(X1_train,X2_train,batch_size=batch_size,num_workers=num_workers)
    ### lab_finetune
    ### joining
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
    ### filterning
    X_train,y_train = where(X_train,y_train,condition=(y_train != 'waving'))
    X_train,y_train = where(X_train,y_train,condition=(y_train != 'standff'))
    X_test,y_test = where(X_test,y_test,condition=(y_test != 'waving'))
    X_test,y_test = where(X_test,y_test,condition=(y_test != 'standff'))
    y_train,lb = label_encode(y_train)
    y_test,lb = label_encode(y_test,enc=lb)
    
    ### sampling
    class_weight = None
    if isinstance(sampling,int):
        idx = resampling_(y_train,oversampling=False)
        X_train,_,y_train,_ = train_test_split(X_train[idx],y_train[idx],train_size=len(lb.classes_)*sampling,stratify=y_train[idx])
        finetune_loader = create_dataloader(X_train,label=y_train,batch_size=y_train.shape[0],num_workers=num_workers)
        validatn_loader  = create_dataloader(X_test,label=y_test, batch_size=y_test.shape[0],num_workers=num_workers)
    else:
        if sampling == 'resampling':
            X_train, y_train = resampling(X_train, y_train, labels=y_train,oversampling=True)
            X_test, y_test = resampling(X_test, y_test, labels=y_test,oversampling=False)
        finetune_loader = create_dataloader(X_train,label=y_train,batch_size=batch_size,num_workers=num_workers)
        validatn_loader  = create_dataloader(X_test,label=y_test,batch_size=y_test.shape[0],num_workers=num_workers)
    class_weight = torch.FloatTensor([1-w for w in pd.Series(y_train).value_counts(normalize=True).sort_index().tolist()])
    print('X1_train: ',X1_train.shape,'\tX2_train: ',X2_train.shape)
    print('X_train: ',X_train.shape,'\ty_train: ',y_train.shape,'\tX_test: ',X_test.shape,'\ty_test: ',y_test.shape)
    print("class: ",lb.classes_)
    print("class_size: ",1-class_weight)
    return pretrain_loader, finetune_loader, validatn_loader, lb, class_weight


def process_field_data(X,y,num=1,lb=None):
    """num: number of sample per class in finetuning dataloader"""
    X = X.reshape(*X.shape,1).transpose(0,3,1,2)
    y,lb = label_encode(y,lb)
    idx = resampling_(y,oversampling=False)
    X_finetune, X_validatn, y_finetune, y_validatn = train_test_split(X[idx],y[idx],train_size=len(lb.classes_)*num,stratify=y[idx])
    finetune_loader = create_dataloader(X_finetune, label=y_finetune,batch_size=y_finetune.shape[0],num_workers=0)
    validatn_loader = create_dataloader(X_validatn, label=y_validatn,batch_size=y_validatn.shape[0],num_workers=0)
    print('X_finetune: ', X_finetune.shape, 'y_finetune: ', y_finetune.shape)
    print('X_validatn: ', X_validatn.shape, 'y_finetune: ', y_validatn.shape)
    print('Classes: ',lb.classes_)
    return finetune_loader,validatn_loader,lb



def prepare_lab_dataloaders(fp,modality,joint='first',sampling='weight',batch_size=64,num_workers=0,lb=None):
    X1,X2,y = import_pair_data(fp,modal=['csi',modality])
    pretrain_loader, finetune_loader, validatn_loader, lb, class_weight = process_lab_data(X1,X2,y,joint,sampling,batch_size,num_workers,lb)
    return pretrain_loader, finetune_loader, validatn_loader, lb, class_weight


def prepare_field_dataloaders(fp,num=1,lb=None):
    X,y = import_data(fp)
    finetune_loader,validatn_loader,lb = process_field_data(X,y,num=num,lb=lb)
    return finetune_loader,validatn_loader,lb



def prepare_single_source(directory,axis=3,train_size=0.8,sampling='weight',batch_size=64,num_workers=0):
    """
    TBD
    """
    ### import data
    X,y = import_data(directory)
    ### add and transpose axis
    if axis:
        X = X.reshape(*X.shape,1)
    if axis == 1:
        X = X.transpose(0,3,1,2)
    ### label encode
    y,lb = label_encode(y)
    ### select training data
    X,y = shuffle(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=train_size,stratify=y)
    ### resample
    if sampling == 'resampling':
        X_train, y_train = resampling(X_train, y_train, labels=y_train,oversampling=True)
        X_test, y_test = resampling(X_test, y_test, labels=y_test,oversampling=False)
    elif sampling == 'weight':
        class_weight = torch.FloatTensor([1-w for w in pd.Series(y_train).value_counts(normalize=True).sort_index().tolist()])
    ### dataloader
    train_loader = create_dataloader(X_train,label=y_train,batch_size=batch_size,num_workers=num_workers)
    test_loader  = create_dataloader(X_test,label=y_test,batch_size=y_test.shape[0],num_workers=num_workers)
    ### Report
    print('X_train: ',X_train.shape,'\ty_train: ',y_train.shape,'\tX_test: ',X_test.shape,'\ty_test: ',y_test.shape)
    print("class: ",lb.classes_)
    if sampling == 'weight':
        print("class_size: ",1-class_weight)
        return train_loader,test_loader,lb,class_weight
    else:
        return train_loader,test_loader,lb


def prepare_double_source(directory,modality='single',axis=1,train_size=0.8,joint='first',p=None,sampling='weight',batch_size=64,num_workers=0):
    """
    Import data with pair source, this is either csi-csi (multi-view) or csi-pwr (multi-modality)

    Argument:
    directory (str)
    modality (str):
        'single' for csi-csi
        'double' for csi-pwr
    axis (int): the axis of the channel for CNN


    TBD
    """
    ### import data
    print('modality:',modality)
    if modality == 'single':
        X1,X2,y = import_pair_data(directory,modal=['csi','nuc2'])
    elif modality == 'double':
        X1,X2,y = import_pair_data(directory,modal=['csi','pwr'])
        if joint not in ['first','second']:
            joint = 'first'
    elif modality == 'dummy':
        X1,X2,y = import_dummies(size=64*4,class_num=6)
    else:
        raise ValueError("Must be in {'single','double','dummy'}")

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
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1,X2,y,train_size=train_size,stratify=y)
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
    ### sampling
    if sampling == 'resampling':
        X_train, y_train = resampling(X_train, y_train, labels=y_train,oversampling=True)
        X_test, y_test = resampling(X_test, y_test, labels=y_test,oversampling=False)
    elif sampling == 'weight':
        class_weight = torch.FloatTensor([1-w for w in pd.Series(y_train).value_counts(normalize=True).sort_index().tolist()])
    ### dataloader
    pretrain_loader = create_dataloader(X1_train,X2_train,batch_size=batch_size,num_workers=num_workers)
    finetune_loader = create_dataloader(X_train,label=y_train,batch_size=batch_size,num_workers=num_workers)
    validatn_loader  = create_dataloader(X_test,label=y_test,batch_size=y_test.shape[0],num_workers=num_workers)
    ### Report
    print('X1_train: ',X1_train.shape,'\tX2_train: ',X2_train.shape)
    print('X_train: ',X_train.shape,'\ty_train: ',y_train.shape,'\tX_test: ',X_test.shape,'\ty_test: ',y_test.shape)
    print("class: ",lb.classes_)
    if sampling == 'weight':
        print("class_size: ",1-class_weight)
        return pretrain_loader, finetune_loader, validatn_loader, lb, class_weight
    else:
        return pretrain_loader, finetune_loader, validatn_loader, lb


def prepare_dummy():

    pretrain_loader = create_dataloader(X1_train,X2_train,batch_size=batch_size,num_workers=num_workers)
    finetune_loader = create_dataloader(X_train,label=y_train,batch_size=batch_size,num_workers=num_workers)
    validatn_loader  = create_dataloader(X_test,label=y_test,batch_size=y_test.shape[0],num_workers=num_workers)
    ### Report
    print('X1_train: ',X1_train.shape,'\tX2_train: ',X2_train.shape)
    print('X_train: ',X_train.shape,'\ty_train: ',y_train.shape,'\tX_test: ',X_test.shape,'\ty_test: ',y_test.shape)
    print("class: ",lb.classes_)
    if sampling == 'weight':
        print("class_size: ",1-class_weight)
        return pretrain_loader, finetune_loader, validatn_loader, lb, class_weight
    else:
        return pretrain_loader, finetune_loader, validatn_loader, lb


