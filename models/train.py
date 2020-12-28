import numpy as np
import pandas as pd
import seaborn as sns  # for heatmaps
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor, nn
from torch.nn import functional as F
import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from data import process_data
from data.import_data import import_experimental_data
from data.process_data import DatasetObject
from models.CNN import CNN
from torchsummary import summary
# get_ipython().run_line_magic('matplotlib', 'inline')



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

def transform_pipeline(X_train, y_train, X_test, y_test):
    X_train = X_train.transpose(0,3,1,2)
    X_test = X_test.transpose(0,3,1,2)

    lb = LabelEncoder()
    y_train = lb.fit_transform(y_train)
    y_test  = lb.transform(y_test)
    return X_train,y_train,X_test,y_test

def create_dataloaders(X_train, y_train, X_test, y_test):
    traindataset = TensorDataset(Tensor(X_train),Tensor(y_train).long())
    testdataset = TensorDataset(Tensor(X_test), Tensor(y_test).long())
    train_loader = DataLoader(traindataset, batch_size=64,shuffle=True, num_workers=0)
    test_loader = DataLoader(testdataset, batch_size=1024, shuffle=True, num_workers=0)
    return train_loader, test_loader

def print_summary(model,input_size):
    print(summary(model,input_size))
    return

def setting(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)
    return criterion, optimizer

def train(model, train_loader, criterion, optimizer, epochs):

    print('Start Training')

    for i in range(epochs):

        for b, (X_train, y_train) in enumerate(train_loader):

            model.zero_grad()

            y_pred = model(X_train)
            loss   = criterion(y_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch: ', i+1,' done')

    return model

def evalaute(model, test_loader):

    with torch.no_grad():

        for X_test, y_test in test_loader:

            y_val = model(X_test)
            predicted = torch.max(y_val, 1)[1]

    arr = confusion_matrix(y_test.view(-1), predicted.view(-1))
    return arr


# import data

def main():
    folderpath1 = "D:/external_data/Experiment3/csv_files/exp_1"  # CHANGE THIS IF THE PATH CHANGED
    df_exp1 = import_experimental_data(folderpath1) # import_clean_data('exp1',
    df_exp1.head()
    # process data
    X_ls, y_ls = seperate_dataframes(df_exp1)
    del df_exp1
    # DatasetObject
    exp_1 = DatasetObject()
    exp_1.import_data(X_ls, y_ls, window_size=800, slide_size=100, skip_labels=['noactivity'])
    exp_1.data_transform(lambda arr: arr.reshape(*arr.shape, 1), axis=1, col=0)
    exp_1.data_transform(lambda x,y,z : process_data.resampling(x, y, z, True), axis=0, col=0)
    del X_ls,y_ls
    # Extract and Transfrom, and Load data into dataloaders
    idxs = [0]
    (X_train, y_train,_),(X_test, y_test,_) = exp_1.__getitem__(idxs,return_sets=True)
    X_train, y_train, X_test, y_test = transform_pipeline(X_train, y_train, X_test, y_test)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)
    del X_train, y_train, X_test, y_test
    # Build model
    model = CNN()
    # criterion, optimizer
    criterion, optimizer = setting(model)
    # Train
    epochs = 100
    model  = train(model, train_loader, criterion, optimizer, epochs)
    # Evalaute
    arr = evalaute(model, test_loader)
    print(arr)


if __name__ == '__main__':
    main()
