
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

from data import torchData
from data.torchData.custom_data import filepath_dataframe
from models.temporal import CNN_LSTM


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
data_dir = 'E:\external_data\opera_csi\Session_2\experiment_data\experiment_data\exp_7_amp_spec_only\spectrogram_multi'
charsplit = '\\'
record_outpath = './record'

# data selection
dataselection_name = 'EXP7-NUC1-Room1-Amp-RandomSplit'
data_selection = torchData.DataSelection(split='random',
                                         test_sub=0.1,
                                         val_sub=0.1,
                                         nuc='NUC1',
                                         room=1,
                                         sample_per_class=11)

# data loading
data_loading = torchData.DataLoading(transform=torchData.CnnLstmS(),
                                     load_data=False,
                                     num_workers=1)

# training
network_name = 'CNNLSTM'
model_builder = CNN_LSTM
optimizer_builder = torch.optim.SGD
lr = 0.0005
epochs = 1

# Experiment Name
comment = 'TestMain'
exp_name = f'{network_name}_Supervised_{dataselection_name}_Comment-{comment}'

#####################################################################################################################

def class_weight(df,col):
    return torch.FloatTensor([1-w for w in df[col].value_counts(normalize=True).sort_index().tolist()])

def reg_loss(model,device,factor=0.0005):
    """
    l2 regularization loss

    Arguments
    model (torch.nn.Module): the model
    device (str): the device whic the model is allocated
    factor (float): factor

    Return:
    loss (torch.Tensor): l2 regularization loss
    """
    l2_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    loss = factor * l2_reg
    return loss

def train(model, train_loader, criterion, optimizer, end, start = 0, test_loader = None, device = None, regularize = None, **kwargs):
    """
    training Loop

    Arguments
    model (torch.nn.Module): the model
    train_loader (torch.utils.Dataset): torch.utils.Dataset of the training set
    criterion (nn.Module): the loss function
    optimizer (torch.optim): the optimizer to backpropagate the network
    end (int): the epoch which the loop end after
    start (int): the epoch which the loop start at
    test_loader (torch.utils.Dataset): torch.utils.Dataset of the validation set
    regularize (bool): add l2 regularization loss on top of the total loss
    device (str): the device whic the model is allocated

    Returns:
    model (torch.nn.Module): trained model
    record (dict): the record of the training, currently have
    'loss' (every epoch), 'accuracy' (every 10 epochs),'f1_weighted' (every 10 epochs),'f1_macro (every 10 epochs)'
    """

    # Check device setting
    if device:
        model = model.to(device)

    print('Start Training')
    record = {'loss':[],'accuracy':[],'f1_weighted':[],'f1_macro':[]}
    i = start
    #Loop
    while i < end:

        print(f"Epoch {i+1}: ", end='')

        for b, (X_train, y_train) in enumerate(train_loader):

            if device:
                X_train = X_train.to(device)

            print(f">", end='')

            optimizer.zero_grad()

            y_pred = model(X_train)

            if device:
                X_train = X_train.cpu()
                del X_train
                y_train = y_train.to(device)

            loss = criterion(y_pred, y_train)

            if regularize:
                loss += reg_loss(model,device)


            loss.backward()
            optimizer.step()

            if device:
                y_pred = y_pred.cpu()
                y_train = y_train.cpu()
                del y_pred,y_train

        # One epoch completed
        loss = loss.tolist()
        record['loss'].append(loss)
        print(f' loss: {loss} ',end='')
        if (test_loader != None) and (i%10 == 0) and (i>0):
            acc = short_evaluation(model,test_loader,device)
            record['accuracy'].append(acc['accuracy'])
            record['f1_weighted'].append(acc['f1_weighted'])
            record['f1_macro'].append(acc['f1_macro'])
            display = acc['accuracy']
            print(f" accuracy: {display}")
        else:
            print('')
        i += 1

    model = model.cpu()
    return model, record

def pretrain_byPair(model, train_loader, criterion, optimizer, end, start = 0, device = None):
    """
    pretraining Loop

    Arguments
    model (torch.nn.Module): the model
    train_loader (torch.utils.Dataset): torch.utils.Dataset of the training set
    criterion (nn.Module): the loss function
    optimizer (torch.optim): the optimizer to backpropagate the network
    end (int): the epoch which the loop end after
    start (int): the epoch which the loop start at
    device (str): the device whic the model is allocated

    Returns:
    model (torch.nn.Module): pretrained model
    record (dict): the record of the training, currently have
    'loss' (every epoch)
    """
    # Check device setting
    if device:
        model = model.to(device)
        criterion = criterion.to(device)

    print('Start Training')
    record = {'loss':[]}
    i = start
    #Loop
    while i < end:
        print(f"Epoch {i+1}: ", end='')
        for b, (items) in enumerate(train_loader):

            if device:
                items = [i.to(device) for i in items]

            print(f">", end='')

            optimizer.zero_grad()

            items = model(items)

            loss = criterion(*items)

            loss.backward()
            optimizer.step()

        # One epoch completed
        loss = loss.tolist()
        record['loss'].append(loss)
        print(f' loss: {loss} ')

        i += 1

    if device:
        items = [i.cpu() for i in items]
        del items
        model = model.cpu()

    return model, record

def short_evaluation(model,test_loader,device):
    """
    quick evaluation of the model during training

    Arguments
    model (torch.nn.Module): the trained model
    test_loader (torch.utils.Dataset): torch.utils.Dataset of the test set
    device (str): the device whic the model is allocated

    Returns:
    acc (dict): the record of the evaluation, currently have 'accuracy', 'f1_weighted', and 'f1_macro'
    """
    # copy the model to cpu
    if device:
        model = model.cpu()
    acc = {'accuracy':0,'f1_weighted':0,'f1_macro':0}
    with torch.no_grad():
        for X_test, y_test in test_loader:
            y_val = model(X_test)
            predicted = torch.max(y_val, 1)[1]
            acc['accuracy'] = accuracy_score(y_test.view(-1), predicted.view(-1))
            acc['f1_weighted'] = f1_score(y_test.view(-1), predicted.view(-1),average='weighted')
            acc['f1_macro'] = f1_score(y_test.view(-1), predicted.view(-1),average='macro')
    # send model back to gpu
    if device:
        model = model.to(device)
    return acc

def evaluation(model,test_loader,label_encoder=None):
    """
    evaluation of the model during training

    Arguments:
    model (torch.nn.Module): the trained model
    test_loader (torch.utils.Dataset): torch.utils.Dataset of the test set
    label_encoder (sklearn.preprocessing.LabelEncoder): the label encoder used to encode the label of the test set

    Returns:
    cmtx (pd.DataFrame): multiclass confusion matrix
    cls(sklearn.metrics.classification_report): classification report
    """
    model = model.cpu()
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test, y_test
            y_val = model(X_test)
            predicted = torch.max(y_val, 1)[1]
    cls = classification_report(y_test.view(-1), predicted.view(-1), output_dict=True)
    cls = pd.DataFrame(cls)
    print(cls)
    cmtx = confusion_matrix(y_test.view(-1), predicted.view(-1))
    cmtx = cmtx_table(cmtx,label_encoder)
    return cmtx,cls

def cmtx_table(cmtx,label_encoder=None):
    """
    rename label of the confusion matrix to its true label

    Arguments
    cmtx (pd.DataFrame): multiclass confusion matrix
    label_encoder (sklearn.preprocessing.LabelEncoder): the label encoder used to encode the label of the test set

    Returns:
    cmtx (pd.DataFrame): multiclass confusion matrix
    """
    if label_encoder != None:
        cmtx = pd.DataFrame(cmtx,
                            index=[f"actual: {i}"for i in label_encoder.classes_.tolist()],
                            columns=[f"predict : {i}"for i in label_encoder.classes_.tolist()])
    else:
        cmtx = pd.DataFrame(cmtx)
    return cmtx

def make_directory(name, epoch=None, filepath='./'):
    """standardized naming convention for the project"""
    time = strftime("%Y_%m_%d_%H_%M", gmtime())
    directory = filepath + name + '_checkpoint_' + str(epoch) + '__' + time
    return directory

def save_checkpoint(model,optimizer,epoch,directory):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, directory)
    print(f"save checkpoint in : {directory}")
    return

def load_checkpoint(model,optimizer,filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model,optimizer

# -----------------------------------------helper---------------------------------------

def record_log(record_outpath,exp_name,phase,record='None',cmtx='None',cls='None',loss_rec=True,acc_rec=False):
    prefix = os.path.join(record_outpath,exp_name+'_Phase-'+phase)
    if type(record) != str:
        if loss_rec:
            pd.DataFrame(record['loss'],columns=['loss']).to_csv(prefix+'_loss.csv')
        if acc_rec:
            df = pd.concat((pd.DataFrame(record['accuracy']),
                            pd.DataFrame(record['f1_weighted']),
                            pd.DataFrame(record['f1_macro'])),
                            axis=1)
            df.columns = ['accuracy','f1_weighted','f1_macro']
            df.to_csv(prefix+'_accuracy.csv')
    if type(cmtx) != str:
        cmtx.to_csv(prefix+'_cmtx.csv')
    if type(cls) != str:
        cls.to_csv(prefix+'_report.csv')
    return

def save_model(model_outpath,exp_name,phase,model):
    model_fp = model_outpath+'/'+exp_name+'_Phase_'+phase
    torch.save(model.state_dict(), model_fp)
    return model_fp

# -----------------------------------Main-------------------------------------------

if __name__ == '__main__':
    print('Experiment Name: ',exp_name)
    print('Cuda Availability: ',torch.cuda.is_available())
    # data preparation
    df = filepath_dataframe(data_dir,charsplit)
    df_train,df_val,df_test = data_selection(df)
    train_loader,val_loader,test_loader = data_loading(df_train,df_val,df_test)

    # auto_setting
    n_classes = df['activity'].nunique()
    weight = class_weight(df_train,'activity').to(device)

    # initial evaluation
    phase = 'lab-initial'
    model = model_builder(n_classes=n_classes)
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
