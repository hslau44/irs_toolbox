import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
import torchvision

from data.spectrogram import import_data
from data.process_data import label_encode, create_dataloaders, resampling

from models import ED_module, Classifier
from losses import SupConLoss
from train import train as finetuning
from train import evaluation,make_directory,save_checkpoint
from models.self_supervised import Projection_head
from models.baseline import Encoder_F as Encoder

# random seed
np.random.seed(1024)
torch.manual_seed(1024)

# gpu setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
parallel = True
num_workers = 0

# data setting
columns = [f"col_{i+1}" for i in range(501)] # 65*501
window_size=None
slide_size=None
dirc = "E:/external_data/Experiment4/Spectrogram_data_csv_files/CSI_data"
PATH = 'C://Users/Creator/Script/Python/Project/irs_toolbox/' # './'

# Training setting
pre_train_epochs = 4000
fine_tune_epochs = 200
bsz = 128
exp_name = 'Encoder_64-128-256-512-64-7_mode_clf_on_exp4csi_s_resampled_4000epochs'


def prepare_data():
    X,y  = import_data(dirc,columns=columns,window_size=window_size,slide_size=slide_size)
    X = X.reshape(*X.shape,1).transpose(0,3,1,2)
    y,lb = label_encode(y)
    return X,y,lb

def prepare_dataloader():
    X,y  = import_data(dirc,columns=columns,window_size=window_size,slide_size=slide_size)
    X = X.reshape(*X.shape,1).transpose(0,3,1,2)
    y,lb = label_encode(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    X_train,y_train,_ = resampling(X_train,y_train,y_train,oversampling=True)
    X_test, y_test,_ = resampling(X_test, y_test,y_test,oversampling=False)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, train_batch_sizes=bsz, test_batch_sizes=2000, num_workers=num_workers)
    return train_loader, test_loader,lb

def create_pretrain_model():
    # External libraries required
    enc = Encoder([64,128,256,512])
    clf = Projection_head(1024,128,head='linear')
    model = ED_module(encoder=enc,decoder=clf)
    return model

def create_finetune_model(enc=None):
    # External libraries required
    if enc == None:
        enc = Encoder([64,128,256,512])
    else:
        enc = freeze_network(enc)
    clf = Classifier(1024,128,6)
    model = ED_module(encoder=enc,decoder=clf)
    return model

def create_criterion():
    # External libraries required
    criterion = SupConLoss(temperature=0.1,stack=True)
    return criterion

def create_optimizer(mode,model):
    if mode == 'pretrain':
        optimizer = torch.optim.SGD(list(model.parameters()), lr=0.0005)
    elif mode == 'finetuning':
        optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0005)
    else:
        raise ValueError("mode: {'pretrain','finetuning'}")
    return optimizer

def freeze_network(model):
    for _, p in model.named_parameters():
        p.requires_grad = False
    return model

def pretrain(model,train_loader,optimizer,criterion,end,start=1,parallel=True):

    # Check device setting
    if parallel == True:
        print('GPU')
        model = model.to(device)

    else:
        print('CPU')
        model = model.cpu()

    print('Start Training')
    record = {'train':[]}
    i = start

    #Loop
    while i <= end:

        print(f"Epoch {i}: ", end='')

        for b, (X, y) in enumerate(train_loader):

            if parallel == True:
                X = X.to(device)

            print(f">", end='')

            optimizer.zero_grad()

            X = model(X)

            if parallel == True:
                y = y.to(device)

            loss = criterion(X,y)

            loss.backward()

            optimizer.step()

        # One epoch completed
        l = loss.tolist()
        record['train'].append(l)
        print(f' loss: {l} ')
        i += 1

        del X,y

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


def switch(training_mode):
    """training mode = {'normal','pretrain'}"""

    # Normal
    if training_mode == 'normal':
        finetune_model = create_finetune_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = create_optimizer('finetuning',finetune_model)
        # Data
        train_loader, test_loader, lb = prepare_dataloader()
        # Training
        finetune_model, record = finetuning(finetune_model , train_loader, criterion, optimizer, fine_tune_epochs, 1, test_loader, parallel)
        cmtx,cls = evaluation(finetune_model,test_loader,label_encoder=lb)
        record_log('finetuning',fine_tune_epochs,record,cmtx,cls)
        save('finetuning',finetune_model,optimizer,fine_tune_epochs)
    # Pretrain
    elif training_mode == 'pretrain':
        pretrain_model = create_pretrain_model()
        criterion = create_criterion()
        optimizer = create_optimizer('pretrain',pretrain_model)
        # Data
        train_loader, test_loader, lb = prepare_dataloader()
        # Pretraining
        pretrain_model, record = pretrain(pretrain_model,train_loader,optimizer,criterion,pre_train_epochs,start=1,parallel=parallel)
        record_log('pretrain',pre_train_epochs,record)
        # save('pretrain',pretrain_model,optimizer,pre_train_epochs)
        del criterion, optimizer, record
        # Fine-tuning
        finetune_model = create_finetune_model(pretrain_model.encoder)
        criterion = nn.CrossEntropyLoss()
        optimizer = create_optimizer('finetuning',finetune_model)
        finetune_model, record = finetuning(finetune_model , train_loader, criterion, optimizer, fine_tune_epochs, 1, test_loader, parallel)
        cmtx,cls = evaluation(finetune_model,test_loader,label_encoder=lb)
        record_log('finetuning',fine_tune_epochs,record,cmtx,cls)
        save('finetuning',finetune_model,optimizer,fine_tune_epochs)
    return


def main():
    pretrain_model = create_pretrain_model()
    criterion = create_criterion()
    optimizer = create_optimizer('pretrain',pretrain_model)
    # Data
    train_loader, test_loader, lb = prepare_dataloader()
    # Pretraining
    pretrain_model, record = pretrain(pretrain_model,train_loader,optimizer,criterion,pre_train_epochs,start=1,parallel=parallel)
    record_log('pretrain',pre_train_epochs,record)
    # save('pretrain',pretrain_model,optimizer,pre_train_epochs)
    del criterion, optimizer, record
    # Fine-tuning
    finetune_model = create_finetune_model(pretrain_model.encoder)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer('finetuning',finetune_model)
    finetune_model, record = finetuning(finetune_model , train_loader, criterion, optimizer, fine_tune_epochs, 1, test_loader, parallel)
    cmtx,cls = evaluation(finetune_model,test_loader,label_encoder=lb)
    record_log('finetuning',fine_tune_epochs,record,cmtx,cls)
    save('finetuning',finetune_model,optimizer,fine_tune_epochs)
    return

if __name__ == '__main__':
    main()
