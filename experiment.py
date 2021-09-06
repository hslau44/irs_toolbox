
import os
import sys
from time import gmtime, strftime
import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor, nn
from poutyne import Model,Experiment

from data.custom_data import filepath_dataframe
from data.selection import Selection
from data.transformation import Transform_CnnLstmS,Transform_CnnS
from data.torchData import DataLoadings,DataLoading
from train import class_weight,evaluation,record_log,train
import models

#####################################################################################################################

# random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# gpu setting
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(DEVICE)
device = DEVICE

## I/O directory
data_dir  = 'E:\\external_data\\opera_csi\\Session_2\\experiment_data\\experiment_data\\exp_7_amp_spec_only\\npy_format'
readtype = 'npy'
splitchar = '\\'
record_outpath = './record'

# data selection
dataselection_name = 'EXP7-NUC1-Room1-Amp-RandomSplit-ResReduced'

data_selection = Selection(split='random',test_sub=0.2,val_sub=0.1,
                           nuc='NUC1',room=1,sample_per_class=None)

# data loading
transform = Transform_CnnS()
batch_size = 64
num_workers = 0

# training
optimizer_builder = torch.optim.SGD
lr = 0.001
epochs = 100


# Model
def model_builder():
    net,size = models.cnn.create_alexnet((1,4))
    model = models.add_classifier(net,size,6,False)
    return model

network_name = 'AlexNet'

# Experiment Name
comment = 'TestModel'
exp_name = f'{network_name}_Supervised_{dataselection_name}_Comment-{comment}'

# -----------------------------------Main-------------------------------------------




if __name__ == '__main__':
    print('Experiment Name: ',exp_name)
    print('Cuda Availability: ',torch.cuda.is_available())
    # data preparation
    df = filepath_dataframe(data_dir,splitchar)
    df_train,df_val,df_test = data_selection(df)

    # data loading
    data_loading = DataLoading(transform=transform,batch_size=batch_size,readtype=readtype,
                               num_workers=num_workers)
    test_loading = DataLoading(transform=transform,batch_size=len(df_test),readtype=readtype,
                               num_workers=num_workers)
    train_loader = data_loading(df_train)
    val_loader   = data_loading(df_val)
    test_loader  = test_loading(df_test)

    # auto_setting
    weight = class_weight(df_train,'activity').to(device)

    # initial evaluation
    phase = 'lab-initial'
    model = model_builder()
    criterion = nn.CrossEntropyLoss(weight=weight).to(device)
    optimizer = optimizer_builder(list(model.parameters()), lr=lr)
    cmtx,cls = evaluation(model,test_loader)
    record_log(record_outpath,exp_name,phase,cmtx=cmtx,cls=cls)

    # training
    phase = 'lab-finetune'
    model, record = train(model=model,
                          train_loader=train_loader,
                          criterion=criterion,
                          optimizer=optimizer,
                          end=epochs,
                          test_loader=test_loader,
                          device=device,
                          regularize=False)

    cmtx,cls = evaluation(model,test_loader)
    acc_rec = True if epochs >= 10 else False
    record_log(record_outpath,exp_name,phase,record=record,cmtx=cmtx,cls=cls,acc_rec=acc_rec)
