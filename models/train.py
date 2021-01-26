import os
import sys
from time import gmtime, strftime
import numpy as np
import pandas as pd
import seaborn as sns  # for heatmaps
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor, nn
from torch.nn import functional as F




root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from data import process_data
from data.import_data import import_experimental_data
from data.datasetobj import DatasetObject
from torchsummary import summary
from models.baseline import Lambda, Classifier,CNN_module
# get_ipython().run_line_magic('matplotlib', 'inline')


def seperate_dataframes(df):
    features_ls,labels_ls = [],[]
    for user in df['user'].unique():
        dataframe = df[df['user']==user]
        features = dataframe[[f'amp_{i}' for i in range(1,91)]].to_numpy()
        features = MinMaxScaler().fit_transform(features)
        features_ls.append(features)
        label = dataframe[['label']].to_numpy()
        labels_ls.append(label)
    return features_ls,labels_ls


def create_datasetobj(X,y):
    datasetobj = DatasetObject()
    datasetobj.import_data(X, y)
    return datasetobj

def transform_datasetobj(datasetobj):
    window_size = 1000
    slide_size = 200
    datasetobj.data_transform(lambda x,y,z : process_data.slide_augmentation(x, y, z,
                                                                window_size=window_size,
                                                                slide_size=slide_size,
                                                                skip_labels=['noactivity']),axis=0)
    datasetobj.data_transform(lambda arr: arr.reshape(-1,window_size,1,90).transpose(0,2,3,1),axis=1, col=0)
    datasetobj.data_transform(lambda x,y,z : process_data.resampling(x, y, z, oversampling = True),axis=0)
    label_encoder = LabelEncoder()
    label_encoder.fit(datasetobj()[1])
    datasetobj.data_transform(lambda arr: label_encoder.transform(arr).reshape(arr.shape),axis=1, col=1)
    return datasetobj, label_encoder


def create_dataloaders(X_train, y_train, X_test, y_test, batch_sizes={'train':64, 'test':200}):
    traindataset = TensorDataset(Tensor(X_train),Tensor(y_train).long())
    testdataset = TensorDataset(Tensor(X_test), Tensor(y_test).long())
    train_loader = DataLoader(traindataset, batch_size=batch_sizes['train'], shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(testdataset, batch_size=batch_sizes['test'], shuffle=True, num_workers=4)
    return train_loader, test_loader

# def create_model():
#     encoder = V2()
#     classifier = Classifier(2304)
#     model = CNN_module(encoder=encoder,decoder=classifier)
#     return model
#
#
# def setting(model):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)
#     return criterion, optimizer


def prepare_data():
    folderpath1 = "E:/external_data/Experiment3/csv_files/exp_1"
    df_exp1 = import_experimental_data(folderpath1)
    X_ls, y_ls = seperate_dataframes(df_exp1)
    del df_exp1
    datasetobj = create_datasetobj(X_ls,y_ls)
    datasetobj, label_encoder = transform_datasetobj(datasetobj)
    datasetobj.shape()
    del X_ls, y_ls
    (X_train, y_train,_),(X_test, y_test,_) = datasetobj([9],return_train_sets=True)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)
    del X_train, y_train, X_test, y_test
    return train_loader, test_loader






def train(model, train_loader, criterion, optimizer, end, start = 1, test_loader = None, auto_save = None, parallel = None, **kwargs):

    # Check device setting
    if parallel == True:
        print('GPU')
        model = model.cuda()
    else:
        print('CPU')

    print('Start Training')
    record = {'train':[],'validation':[]}
    i = start
    #Loop
    while i <= end:
        print(f"Epoch {i}: ", end='')
        for b, (X_train, y_train) in enumerate(train_loader):
            if parallel == True:
                X_train, y_train = X_train.cuda(), y_train.cuda() #.to(device)
            print(f">", end='')
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss   = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            del X_train, y_train, y_pred
        # One epoch completed
        loss = loss.tolist()
        record['train'].append(loss)
        print(f' loss: {loss} ',end='')
        if (test_loader != None) and i%100 ==0 :
            acc = short_evaluation(model,test_loader,parallel)
            record['validation'].append(acc)
            print(f' accuracy: {acc}')
        else:
            print('')
        i += 1

    model = model.cpu()
    return model, record

def short_evaluation(model,test_loader,parallel):
    # copy the model to cpu
    if parallel == True:
        model = model.cpu()
    with torch.no_grad():
        for X_test, y_test in test_loader:
            y_val = model(X_test)
            predicted = torch.max(y_val, 1)[1]
            acc = accuracy_score(y_test.view(-1), predicted.view(-1))
    # send model back to gpu
    if parallel == True:
        model = model.cuda()
    return acc



def evaluation(model,test_loader):
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
    return cmtx,cls

def cmtx_table(cmtx,label_encoder=None):
    if label_encoder != None:
        cmtx = pd.DataFrame(cmtx,
                            index=[f"actual: {i}"for i in label_encoder.categories_[0].tolist()],
                            columns=[f"predict : {i}"for i in label_encoder.categories_[0].tolist()])
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


def seperate_dataframes(df):
    features_ls,labels_ls = [],[]
    for user in df['user'].unique():
        dataframe = df[df['user']==user]
        features = dataframe[[f'amp_{i}' for i in range(1,91)]].to_numpy()
#         features = MinMaxScaler().fit_transform(features)
        features_ls.append(features)
        label = dataframe[['label']].to_numpy()
        labels_ls.append(label)
    return features_ls,labels_ls

# ----------------------------- helper -----------------------------------

def create_datasetobject(X,y):
    window_size = 900
    slide_size = 200
    datasetobj = DatasetObject()
    datasetobj.import_data(X, y)
    datasetobj.data_transform(lambda x,y,z : process_data.slide_augmentation(x, y, z,
                                                                window_size=window_size,
                                                                slide_size=slide_size,
                                                                skip_labels=['noactivity']),axis=0)

    # datasetobj.data_transform(lambda arr: arr.reshape(*arr.shape, 1),axis=1, col=0)
    # datasetobj.data_transform(lambda arr: arr.transpose(0,3,1,2),axis=1, col=0)
    datasetobj.data_transform(lambda arr: arr.reshape(-1,window_size,3,30).transpose(0,2,3,1),axis=1, col=0)
    datasetobj.data_transform(lambda x,y,z : process_data.resampling(x, y, z, oversampling = True),axis=0)

    label_encoder = LabelEncoder()
    label_encoder.fit(datasetobj()[1])
    datasetobj.data_transform(lambda arr: label_encoder.transform(arr).reshape(arr.shape),axis=1, col=1)
    datasetobj.shape()
    return datasetobj, label_encoder



# import data
def main():

    # 1. Import data
    folderpath1 = "E:/external_data/Experiment3/csv_files/exp_1"
    df_exp1 = import_experimental_data(folderpath1)

    # 2. Process data
    X_ls, y_ls = seperate_dataframes(df_exp1)
    del df_exp1

    # 3. DatasetObject
    exp_1 = create_datasetobject(X_ls, y_ls)
    del X_ls,y_ls

    # 4. Load data into dataloaders
    idxs = [0]
    (X_train, y_train,_),(X_test, y_test,_) = exp_1(idxs,return_train_sets=True)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)
    del X_train, y_train, X_test, y_test

    # 5. Build model
    model = CNN_module(encoder=Encoder(),decoder=Classifier(1024))

    # 6. criterion, optimizer
    criterion, optimizer = setting(model)

    # 7. Train
    epochs = 100
    model  = train(model, train_loader, criterion, optimizer, epochs)

    # 8. Evalaute
    arr = evalaute(model, test_loader)
    print(arr)

    return


if __name__ == '__main__':
    main()
