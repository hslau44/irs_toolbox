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
from torch.nn import functional as F

from data import prepare_single_source



#####################################################################################################################

# random seed
np.random.seed(1024)
torch.manual_seed(1024)

# gpu setting
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = DEVICE
torch.cuda.set_device(DEVICE)

### data setting
# DIRC = 'E:/external_data/Experiment4/Spectrogram_data_csv_files/CSI_data_pair'
DIRC = './data/experiment_data/exp_2/spectrogram'
AXIS=1
TRAIN_SIZE=0.8
SAMPLING='weight'

### train setting
BATCH_SIZE=64
NUM_WORKERS = 0
REGULARIZE = None
NUM_EPOCHS = 200
MAIN_NAME = 'Trainmode-normal_Network-shallowv2_Data-exp4csi' #'Trainmode_simclr_Network_shallowv2_Data_exp4nuc1'
OUT_PATH = None # '.'
output = OUT_PATH


#####################################################################################################################


def reg_loss(model,device,factor=0.0005):
    l2_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    return factor * l2_reg


def train(model, train_loader, criterion, optimizer, end, start = 1, test_loader = None, device = None, regularize = None, **kwargs):

    # Check device setting
    if device:
        model = model.to(device)

    print('Start Training')
    record = {'loss':[],'accuracy':[],'f1_weighted':[],'f1_macro':[]}
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
        record['loss'].append(loss)
        print(f' loss: {loss} ',end='')
        if (test_loader != None) and i%10 ==0 :
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


def short_evaluation(model,test_loader,device):
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



def make_directory(name, epoch=None, filepath='./'):
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

# def record_log(main_name,epochs,record='None',cmtx='None',cls='None',filepath='./record/'):
#     path = make_directory(main_name,epoch=epochs,filepath=filepath)
#     if type(record) != str:
#         pd.DataFrame(record['train'],columns=['train_loss']).to_csv(path+'_loss.csv')
#     if type(cmtx) != str:
#         if  type(record) != str:
#             pd.DataFrame(record['validation'],columns=['validation_accuracy']).to_csv(path+'_accuracy.csv')
#         cmtx.to_csv(path+'_cmtx.csv')
#     if type(cls) != str:
#         cls.to_csv(path+'_report.csv')
#     return

def record_log(record_outpath,exp_name,phase,record='None',cmtx='None',cls='None',loss_rec=True,acc_rec=False):
    prefix = record_outpath+'/'+exp_name+'_Phase_'+phase
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


