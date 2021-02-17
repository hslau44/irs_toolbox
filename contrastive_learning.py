import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from torchsummary import summary
import torchvision

from data.spectrogram import import_data,import_pair_data,import_CsiPwr_data
from data.process_data import label_encode, create_dataloaders, resampling,selections

from models import ED_module, Classifier
from losses import SupConLoss, NT_Xent
from train import train as finetuning
from train import evaluation,make_directory,save_checkpoint
from models.self_supervised import Projection_head
from models.baseline import Encoder_F as Encoder
from models.cnn import create_vgg16


# random seed
np.random.seed(1024)
torch.manual_seed(1024)

# gpu setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
parallel = True
num_workers = 0

# data setting
loc_dirc = 'E:/external_data/Experiment4/Spectrogram_data_csv_files/CSI_PWR_pair'
remote_dirc = './data/csi_pwr'
PATH = './' # './'

dirc = remote_dirc
mode = 2
p = None
resample = True
pre_train_epochs = 800
fine_tune_epochs = 300
bsz = 64
parallel = True
csi_out_size = (2,3)
pwr_out_size = (3,2)

exp_name = 'Encoder_vgg16_mode_clf_on_exp4csipwr_s_resample'#'Encoder_vgg16_mode_clf_on_exp4csipwr'


def prepare_data(dirc):
    X,y  = import_data(dirc)
    X = X.reshape(*X.shape,1).transpose(0,3,1,2)
    y,lb = label_encode(y)
    return X,y,lb

def prepare_dataloader(dirc,resample=None,train_batch_sizes=bsz):
    X,y,lb = prepare_data(dirc)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    if resample:
        X_train,y_train,_ = resampling(X_train,y_train,y_train,oversampling=True)
        X_test, y_test,_ = resampling(X_test, y_test,y_test,oversampling=False)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, train_batch_sizes=train_batch_sizes, test_batch_sizes=2000, num_workers=num_workers)
    return train_loader, test_loader,lb

def reshape_axis1(X):
    """
    To be depreciated
    """
    assert len(X.shape) > 1, 'must be two dimensional'
    return X.reshape(X.shape[0]*X.shape[1],1,*X.shape[2:])

def prepare_dataloader_pairdata(dirc,mode,p=None,resample=None):
    """
    Import pair data
    """
    if mode == 1:
        X1,X2,y = import_pair_data(dirc)
    elif mode == 2:
        X1,X2,y = import_CsiPwr_data(dirc)
    else:
        raise ValueError('Must be 1 or 2 for pair')

    X1 = X1.reshape(*X1.shape,1).transpose(0,3,1,2)
    X2 = X2.reshape(*X2.shape,1).transpose(0,3,1,2)
    y,lb = label_encode(y)

    training,validation = selections(X1,X2,y,p=0.2)
    X_test, y_test = validation[0],validation[2]

    if p:
        (X_train,y_train) , _ = selections(training[0],training[2],p=p)
    else:
        X_train,y_train = training[0],training[2]

    if resample:
        X_train,y_train,_ = resampling(X_train,y_train,y_train,oversampling=True)
        # X_test, y_test,_ = resampling(X_test, y_test,y_test,oversampling=False)

    ### Dataloader
    print('X1: ',training[0].shape,' X2: ',training[1].shape)
    print('X_train: ',X_train.shape,' y_train: ',y_train.shape,' X_test: ',X_test.shape,' y_test: ',y_test.shape)
    pretraindataset = TensorDataset(Tensor(training[0]),Tensor(training[1]))
    finetunedataset = TensorDataset(Tensor(X_train),Tensor(y_train).long())
    validatndataset = TensorDataset(Tensor(X_test), Tensor(y_test).long())
    pretrain_loader = DataLoader(pretraindataset, batch_size=bsz, shuffle=True, num_workers=num_workers, drop_last=True)
    finetune_loader = DataLoader(finetunedataset, batch_size=bsz, shuffle=True, num_workers=num_workers, drop_last=True)
    validatn_loader = DataLoader(validatndataset, batch_size=2000, shuffle=True, num_workers=num_workers)
    return pretrain_loader, finetune_loader, validatn_loader, lb


def create_pretrain_model(out_size=(2,3)):
    # External libraries required
    enc = create_vgg16(out_size)
    clf = Projection_head(512*out_size[0]*out_size[1],128,head='linear')
    model = ED_module(encoder=enc,decoder=clf)
    return model

def create_finetune_model(enc=None,out_size=(2,3)):
    # External libraries required
    if enc == None:
        enc = create_vgg16(out_size)
    else:
        enc = freeze_network(enc)
    clf = Classifier(512*out_size[0]*out_size[1],128,6)
    model = ED_module(encoder=enc,decoder=clf)
    return model

def create_criterion():
    # External libraries required
    criterion = NT_Xent(bsz, temperature=0.1, world_size=1)
    return criterion


def create_optimizer(mode,model1,model2=None):
    """mode: {'pretrain','pretrain_2','finetuning'}"""
    if mode == 'pretrain':
        optimizer = torch.optim.SGD(list(model1.parameters()), lr=0.0005)
    elif mode == 'pretrain_2':
        optimizer = torch.optim.SGD(list(model1.parameters())+list(model2.parameters()), lr=0.0005)
    elif mode == 'finetuning':
        optimizer = torch.optim.Adam(list(model1.parameters()), lr=0.0005)
    else:
        raise ValueError("mode: {'pretrain','pretrain_2','finetuning'}")

    return optimizer

def freeze_network(model):
    for _, p in model.named_parameters():
        p.requires_grad = False
    return model


def pretrain(model,train_loader,optimizer,criterion,end,start=1,model2=None,parallel=True):

    # Check device setting
    if parallel == True:
        print('GPU')
        model = model.to(device)
        
        if model2:
            model2 = model2.to(device)

    else:
        print('CPU')
        model = model.cpu()

    print('Start Training')
    record = {'train':[]}
    i = start

    #Loop
    while i <= end:

        print(f"Epoch {i}: ", end='')

        for b, (X1, X2) in enumerate(train_loader):

            print(f">", end='')

            optimizer.zero_grad()

            if parallel == True:
                X1 = X1.to(device)

            X1 = model(X1)

            if parallel == True:
                X2 = X2.to(device)

            if model2:
                X2 = model2(X2)
            else:
                X2 = model(X2)

            loss = criterion(X1,X2)

            loss.backward()

            optimizer.step()

        # One epoch completed
        l = loss.tolist()
        record['train'].append(l)
        print(f' loss: {l} ')
        i += 1

        del X1,X2

    model = model.cpu()

    return model,record


def record_log(mode,epochs,record,cmtx=None,cls=None):
    if mode == 'pretrain':
        path = make_directory(exp_name+'_pretrain',epoch=epochs,filepath=PATH+'record/')
        pd.DataFrame(record['train'],columns=['train_loss']).to_csv(path+'_loss.csv')
    elif mode == 'finetuning':
        path = make_directory(exp_name+'_finetuning',epoch=epochs,filepath=PATH+'record/')
        pd.DataFrame(record['train'],columns=['train_loss']).to_csv(path+'_loss.csv')
        pd.DataFrame(record['validation'],columns=['validation_accuracy']).to_csv(path+'_accuracy.csv')
        cls.to_csv(path+'_report.csv')
        cmtx.to_csv(path+'_cmtx.csv')
    return

def save(mode,model,optimizer,epochs):
    if mode == 'pretrain':
        path = make_directory(exp_name+'_pretrain',epoch=epochs,filepath=PATH+'models/saved_models/')
        save_checkpoint(model, optimizer, epochs, path)
    elif mode == 'finetuning':
        path = make_directory(exp_name+'_finetuning',epoch=epochs,filepath=PATH+'models/saved_models/')
        save_checkpoint(model, optimizer, epochs, path)
    return


def main():
    csi_model = create_pretrain_model(csi_out_size)
    pwr_model = create_pretrain_model(pwr_out_size)
    criterion = create_criterion()
    optimizer = create_optimizer('pretrain_2',csi_model,pwr_model)
    # Data
    pretrain_loader, finetune_loader, validatn_loader, lb = prepare_dataloader_pairdata(dirc,mode,p,resample)
    # Pretraining
    csi_model, record = pretrain(csi_model,pretrain_loader,optimizer,criterion,pre_train_epochs,
                                 start=1,
                                 model2=pwr_model,
                                 parallel=parallel)
    # save and log
    record_log('pretrain',pre_train_epochs,record)
    save('pretrain',csi_model,optimizer,pre_train_epochs)
    del criterion, optimizer, record, pretrain_loader
    # Fine-tuning
    finetune_model = create_finetune_model(csi_model.encoder,csi_out_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer('finetuning',finetune_model)
    finetune_model, record = finetuning(finetune_model , finetune_loader, criterion, optimizer, fine_tune_epochs, 1, validatn_loader, parallel)
    cmtx,cls = evaluation(finetune_model,validatn_loader,label_encoder=lb)
    record_log('finetuning',fine_tune_epochs,record,cmtx,cls)
    save('finetuning',finetune_model,optimizer,fine_tune_epochs)
    return

if __name__ == '__main__':
    main()
