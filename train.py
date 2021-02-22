import os
import sys
from time import gmtime, strftime
import numpy as np
import pandas as pd
import seaborn as sns  # for heatmaps
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor, nn
from torch.nn import functional as F




np.random.seed(1024)
torch.manual_seed(1024)

DIRC = 'E:/external_data/Experiment4/Spectrogram_data_csv_files/CSI_data_pair'
NUM_EPOCHS = 300
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = DEVICE
PATH = '.'
NUM_WORKERS = 0

torch.cuda.set_device(DEVICE)



def reg_loss(model,factor=0.0005):
    l2_reg = torch.tensor(0.).to(model.device)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    return factor * l2_reg


def train(model, train_loader, criterion, optimizer, end, start = 1, test_loader = None, device = None, regularize = None, **kwargs):

    # Check device setting
    if device:
        model = model.to(device)

    print('Start Training')
    record = {'train':[],'validation':[]}
    i = start
    #Loop
    while i <= end:
        print(f"Epoch {i}: ", end='')
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
        record['train'].append(loss)
        print(f' loss: {loss} ',end='')
        if (test_loader != None) and i%10 ==0 :
            acc = short_evaluation(model,test_loader,device)
            record['validation'].append(acc)
            print(f' accuracy: {acc}')
        else:
            print('')
        i += 1

    model = model.cpu()
    return model, record


def short_evaluation(model,test_loader,device):
    # copy the model to cpu
    if device:
        model = model.cpu()
    with torch.no_grad():
        for X_test, y_test in test_loader:
            y_val = model(X_test)
            predicted = torch.max(y_val, 1)[1]
            acc = f1_score(y_test.view(-1), predicted.view(-1),average='weighted')
    # send model back to gpu
    if device:
        model = model.to(device)
    return acc



def evaluation(model,test_loader,label_encoder=None):
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
    if label_encoder != None:
        cmtx = pd.DataFrame(cmtx,
                            index=[f"actual: {i}"for i in label_encoder.classes_.tolist()],
                            columns=[f"predict : {i}"for i in label_encoder.classes_.tolist()])
    else:
        cmtx = pd.DataFrame(cmtx)
    return cmtx



def make_directory(name, epoch=None, filepath='./models/saved_models/'):
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



def record_log(main_name,epochs,record,cmtx=None,cls=None):
    path = make_directory(main_name,epoch=epochs,filepath=PATH+'/record/')
    pd.DataFrame(record['train'],columns=['train_loss']).to_csv(path+'_loss.csv')
    if cmtx:
        pd.DataFrame(record['validation'],columns=['validation_accuracy']).to_csv(path+'_accuracy.csv')
        cmtx.to_csv(path+'_cmtx.csv')
    if cls:
        cls.to_csv(path+'_report.csv')
    return

def save(main_name,model,optimizer,epochs):
    path = make_directory(main_name,epoch=epochs,filepath=PATH+'/models/saved_models/')
    save_checkpoint(model, optimizer, epochs, path)
    return



# ----------------------------------------------------------------------------------------



def main():
    model = create_model(enc=None,out_size=(2,2))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0005)
    _ , train_loader, test_loader, lb = prepare_double_source(directory=DIRC,
                                                              modality='single',
                                                              axis=1,
                                                              train_size=0.8,
                                                              joint='joint',
                                                              p=None,
                                                              resample=None,
                                                              batch_size=64,
                                                              num_workers=NUM_WORKERS)
    model, record = train(model=model,
                          train_loader=train_loader,
                          criterion=criterion,
                          optimizer=optimizer,
                          end=NUM_EPOCHS,
                          start = 1,
                          test_loader = test_loader,
                          device = DEVICE,
                          regularize = None)
    cmtx,cls = evaluation(model,test_loader,label_encoder=lb)
    record_log(main_name,epochs,record,cmtx=None,cls=None)
    save(main_name,model,optimizer,epochs)
    return



if __name__ == '__main__':
    main()
