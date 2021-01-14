import os
import sys
from time import gmtime, strftime
import numpy as np
import pandas as pd
import seaborn as sns  # for heatmaps
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor, nn
from torch.nn import functional as F


from models.CNN import Encoder,Classifier

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from data import process_data
from data.import_data import import_experimental_data
from data.process_data import DatasetObject
from torchsummary import summary
# get_ipython().run_line_magic('matplotlib', 'inline')



def create_dataloaders(X_train, y_train, X_test, y_test, batch_sizes={'train':128, 'test':2048}):
    traindataset = TensorDataset(Tensor(X_train),Tensor(y_train).long())
    testdataset = TensorDataset(Tensor(X_test), Tensor(y_test).long())
    train_loader = DataLoader(traindataset, batch_size=batch_sizes['train'], shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(testdataset, batch_size=batch_sizes['test'], shuffle=True, num_workers=0)
    return train_loader, test_loader

def print_summary(model,input_size):
    print(summary(model,input_size))
    return

def setting(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)
    return criterion, optimizer

def train(model, train_loader, criterion, optimizer, end, start = 1, evaluation = None, auto_save = None, **kwargs):
    print('Start Training')
    i = start
    while i <= end:
        print(f"Epoch {i}: ", end='')
        for b, (X_train, y_train) in enumerate(train_loader):
            print(f">", end='')
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss   = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
        print(f' loss: {loss.tolist()}')
        if i%100 == 0:
            if evaluation == True:
                array = evaluate(model,test_loader=kwargs['test_loader'])
                print('Evaluation:')
                print(array)
            if auto_save == True:
                directory = make_directory(name=kwargs['name'],epoch=i)
                save_checkpoint(model,optimizer,i,directory)
        i += 1
    return model

def cross_validation(model_func, setting, datasetobj, cross_valid = False):

    epochs = 100

    cmtxs = []

    for i in range(datasetobj.shape()[0]):

        model = model_func()

        criterion, optimizer = setting(model)

        (X_train, y_train,_),(X_test, y_test,_) = datasetobj([i],return_sets=True)

        # X_train, y_train, X_test, y_test = transform_pipeline(X_train, y_train, X_test, y_test)

        train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)

        del X_train, y_train, X_test, y_test

        model  = train(model, train_loader, criterion, optimizer, epochs)

        cmtx = evalaute(model, test_loader)

        del model

        cmtxs.append(cmtx)

    return cmtxs


def evaluate(model, test_loader):
    with torch.no_grad():
        for X_test, y_test in test_loader:
            y_val = model(X_test)
            predicted = torch.max(y_val, 1)[1]
    print(classification_report(y_test.view(-1), predicted.view(-1)))
    arr = confusion_matrix(y_test.view(-1), predicted.view(-1))
    return arr

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
    datasetobj = DatasetObject()
    datasetobj.import_data(X, y)
    datasetobj.data_transform(lambda x,y,z : process_data.slide_augmentation(x, y, z,
                                                                window_size=900,
                                                                slide_size=200,
                                                                skip_labels=['noactivity']),axis=0)

    # datasetobj.data_transform(lambda arr: arr.reshape(*arr.shape, 1),axis=1, col=0)
    # datasetobj.data_transform(lambda arr: arr.transpose(0,3,1,2),axis=1, col=0)
    datasetobj.data_transform(lambda x,y,z : process_data.resampling(x, y, z, oversampling = True),axis=0)

    label_encoder = LabelEncoder()
    label_encoder.fit(datasetobj()[1])
    datasetobj.data_transform(lambda arr: label_encoder.transform(arr).reshape(arr.shape),axis=1, col=1)
    datasetobj.shape()
    return datasetobj



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
