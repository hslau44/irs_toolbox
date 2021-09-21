import os
import sys
from time import gmtime, strftime
import numpy as np
import pandas as pd
import torch
import poutyne
from poutyne import Model

from data.selection import Selection
from data.torchData import DataLoading


NUC = 'NUC1'
ROOM = 1
BATCHSIZE = 64
READTYPE = 'npy'
NUM_WORKERS = 0
OPTIMIZER = 'adam'
LOSS = 'cross_entropy'
BATCH_METRICS = ['accuracy']
EPOCH_METRICS = [poutyne.F1('micro'),poutyne.F1('macro')]
EPOCHS = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(DEVICE)


def data_preparation(dataframe,test_sub,transform,**kwargs):
    # parameters
    nuc = kwargs.get('nuc',NUC)
    room = kwargs.get('room',ROOM)
    batch_size = kwargs.get('batch_size',BATCHSIZE)
    readtype = kwargs.get('readtype',READTYPE)
    num_workers = kwargs.get('num_workers',NUM_WORKERS)
    # selection
    selection = Selection(split='loov',test_sub=test_sub,nuc=nuc,room=room)
    df_train,_,df_test = selection(dataframe)
    # loading
    data_loading = DataLoading(transform=transform,batch_size=batch_size,readtype=readtype,
                               num_workers=num_workers,shuffle=False)
    test_loading = DataLoading(transform=transform,batch_size=len(df_test),readtype=readtype,
                               num_workers=num_workers,shuffle=False)
    train_loader = data_loading(df_train)
    test_loader  = test_loading(df_test)
    return train_loader,test_loader

def append_record(history,record):
    df = pd.DataFrame(history)
    row = df.iloc[len(df)-1,1:]
    record.append(row)
    return

def leaveOneOut_crossValidation(model,dataframe,transform,verbose=True,**kwargs):

    device = kwargs.get('device',DEVICE)
    optimizer = kwargs.get('optimizer',OPTIMIZER)
    loss = kwargs.get('loss',LOSS)
    epochs = kwargs.get('epochs',EPOCHS)
    batch_metrics = kwargs.get('batch_metrics',BATCH_METRICS)
    epoch_metrics = kwargs.get('epoch_metrics',EPOCH_METRICS)

    records = []

    for test_sub in dataframe['person'].unique():

        try:

            if verbose: print(test_sub)

            model.build()

            train_loader,test_loader = data_preparation(dataframe,test_sub,transform)

            mdl = Model(model,OPTIMIZER,LOSS,
                        batch_metrics=batch_metrics,epoch_metrics=epoch_metrics).to(device)
            history = mdl.fit_generator(train_loader, test_loader, epochs=EPOCHS)
            append_record(history,records)
            if verbose: print(records[-1],'\n')

        except:
            print(f'Skip {test_sub}')
            continue

    return pd.DataFrame(records)
