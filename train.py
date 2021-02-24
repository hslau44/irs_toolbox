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
from models import create_baseline



np.random.seed(1024)
torch.manual_seed(1024)

DIRC = 'E:/external_data/Experiment4/Spectrogram_data_csv_files/CSI_data'
NUM_EPOCHS = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = DEVICE
PATH = '.'
NUM_WORKERS = 0
MAIN_NAME = 'Trainmode_normal_Network_shallowv2_Data_exp4nuc1'

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

def record_log(main_name,epochs,record,cmtx='None',cls='None',filepath='./record/'):
    path = make_directory(main_name,epoch=epochs,filepath=filepath)
    pd.DataFrame(record['train'],columns=['train_loss']).to_csv(path+'_loss.csv')
    if type(cmtx) != str:
        pd.DataFrame(record['validation'],columns=['validation_accuracy']).to_csv(path+'_accuracy.csv')
        cmtx.to_csv(path+'_cmtx.csv')
    if type(cls) != str:
        cls.to_csv(path+'_report.csv')
    return

def save(main_name,model,optimizer,epochs,filepath='./models/saved_models/'):
    path = make_directory(main_name,epoch=epochs,filepath=filepath)
    save_checkpoint(model, optimizer, epochs, path)
    return

# ---------------------------------------fast model setup-------------------------------------------

from models.baseline import Encoder,Classifier
from models.utils import Lambda,ED_module
from models import add_classifier

# class Encoder(nn.Module):
#     """
#     Three layer Encoder for spectrogram (1,65,65), 3 layer
#     """
#     def __init__(self,num_filters):
#         super(Encoder, self).__init__()
#         l1,l2,l3 = num_filters
#         ### 1st ###
#         self.conv1 = nn.Conv2d(1,l1,kernel_size=5,stride=1)
#         self.norm1 = nn.BatchNorm2d(l1) # nn.BatchNorm2d()
#         self.actv1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
#         ### 2nd ###
#         self.conv2 = nn.Conv2d(l1,l2,kernel_size=4,stride=2)
#         self.norm2 = nn.BatchNorm2d(l2)
#         self.actv2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
#         ### 3rd ###
#         self.conv3 = nn.Conv2d(l2,l3,kernel_size=3,stride=3)
#         self.norm3 = Lambda(lambda x:x)
#         self.actv3 = nn.Tanh()
#         self.pool3 = nn.MaxPool2d(kernel_size=(2,2))
#
#     def forward(self,X):
#         X = self.pool1(self.actv1(self.norm1(self.conv1(X))))
#         X = self.pool2(self.actv2(self.norm2(self.conv2(X))))
#         X = self.pool3(self.actv3(self.norm3(self.conv3(X))))
#         X = torch.flatten(X, 1)
#         # print(X.shape)
#         return X

def create_baseline():
    out = 96
    enc = Encoder([32,64,out])
    model = add_classifier(enc,out_size=10*out,freeze=False)
    return model


# -----------------------------------Main-------------------------------------------



def main():
    model = create_baseline()
    train_loader,test_loader,lb,class_weight = prepare_single_source(directory=DIRC,
                                                                     axis=3,
                                                                     train_size=0.8,
                                                                     sampling='weight',
                                                                     batch_size=64,
                                                                     num_workers=NUM_WORKERS)
    criterion = nn.CrossEntropyLoss(weight=class_weight).to(DEVICE)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0005)
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
    record_log(MAIN_NAME,NUM_EPOCHS,record,cmtx=cmtx,cls=cls)
    save(MAIN_NAME,model,optimizer,NUM_EPOCHS)
    return



if __name__ == '__main__':
    main()
