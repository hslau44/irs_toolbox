import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from data.import_data import import_clean_data
from data.datasetobj import DatasetObject
from data import process_data
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset




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
    train_loader = DataLoader(traindataset, batch_size=batch_sizes['train'], shuffle=True, num_workers=1, drop_last=True)
    test_loader = DataLoader(testdataset, batch_size=batch_sizes['test'], shuffle=True, num_workers=1)
    return train_loader, test_loader




def prepare_exp_1():
    fp = "E:/external_data/Experiment3/csv_files/exp_1"
    df = import_clean_data('exp_1',fp)
    X_ls, y_ls = seperate_dataframes(df)
    del df
    datasetobj = create_datasetobj(X_ls,y_ls)
    datasetobj, label_encoder = transform_datasetobj(datasetobj)
    datasetobj.shape()
    del X_ls, y_ls
    (X_train, y_train,_),(X_test, y_test,_) = datasetobj([9],return_train_sets=True)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)
    return train_loader, test_loader
