from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as T
from torch.utils.data import DataLoader
from data.torchData.custom_data import filepath_dataframe
from data.torchData.utils import DatasetObject
from data.torchData.data_selection import *
from data.torchData.transformation import *

def dataSelection_Set1(df):
    """To test the model under lab environemnt by leave-one-person-out validation"""
    lb = LabelEncoder()
    df['activity'] = lb.fit_transform(df['activity'])
    df = df[df['nuc'] == 'NUC1']
    df = df[df['room'] == 1]
    train,val,test = leaveOneOut_split(df,testsub='One',valsub='Four')
    test = resampling(test,'activity',oversampling=False)
    return train,val,test

def dataSelection_Set2(df):
    """To test the model under lab environemnt by random-split validation"""
    lb = LabelEncoder()
    df['activity'] = lb.fit_transform(df['activity'])
    df = df[df['nuc'] == 'NUC1']
    df = df[df['room'] == 1]
    train,test = random_split(df,train_size=0.8)
    train,val = random_split(train,train_size=0.875)
    test = resampling(test,'activity',oversampling=False)
    return train,val,test

def dataSelection_Set3(df):
    """To test the model under unseen environemnt"""
    lb = LabelEncoder()
    df['activity'] = lb.fit_transform(df['activity'])
    df = df[df['nuc'] == 'NUC1']
    train = df[df['room'] == 1]
    test  = df[df['room'] == 2]

    train,val, _ = leaveOneOut_split(train,testsub='One',valsub='Four')
    _ , _ , test = leaveOneOut_split(test,testsub='One',valsub='Four')

    test = resampling(test,'activity',oversampling=False)
    return train,val,test


def dataSelection_Set4(df):
    """ To test if combine data provide better generalizability"""
    nuc = ['NUC1','NUC2']
    rooms = [1,2]

    lb = LabelEncoder()
    df['activity'] = lb.fit_transform(df['activity'])
    df = df[df['nuc'].isin(nuc)]
    df = df[df['room'].isin(rooms)]

    train,val,test = leaveOneOut_split(df,testsub='One',valsub='Four')
    test = resampling(test,'activity',oversampling=False)
    return train,val,test


def dataSelection_Set5(df,spc=5):
    """To test model performance under low number of sample"""
    lb = LabelEncoder()
    df['activity'] = lb.fit_transform(df['activity'])
    df = df[df['nuc'] == 'NUC1']
    df = df[df['room'] == 1]
    num_class = df['activity'].nunique()

    train,val,test = leaveOneOut_split(df,testsub='One',valsub='Four')

    train = resampling(train,'activity',oversampling=False)
    train,_ = random_split(train,train_size=spc*num_class)
    test = resampling(test,'activity',oversampling=False)
    return train,val,test

def dataSelection(df,split='loov',nuc='NUC1',room=1,spc=None):

    if isinstance(nuc,list):
        df = df[df['nuc'].isin(nuc)]
    else:
        df = df[df['nuc'] == nuc]

    if isinstance(room,list):
        df = df[df['room'].isin(room)]
    else:
        df = df[df['room'] == room]

    num_class = df['activity'].nunique()

    if split == 'random':
        train,test = random_split(df,train_size=0.8)
        train,val = random_split(train,train_size=0.875)
    elif split == 'loov':
        train,val,test = leaveOneOut_split(df,testsub='One',valsub='Four')

    if isinstance(spc,int):
        train = resampling(train,'activity',oversampling=False)
        train,_ = random_split(train,train_size=num_spc*spc)
    test = resampling(test,'activity',oversampling=False)
    return train,val,test


def dataLoader_CnnLstmS(df):
    """dataLoader_cnnLstmS"""
    transform = T.Compose([ReduceRes(),Unsqueeze(),ToStackImg(25)])
    datasetobj = DatasetObject(filepaths=df['fullpath'].to_numpy(),
                               label=df['activity'].to_numpy(),
                               transform=transform)
    dataloader = DataLoader(datasetobj, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    return dataloader

def dataLoader_CnnS(df):
    """dataLoader_CnnS"""
    transform = T.Compose([ReduceRes(),Unsqueeze()])
    datasetobj = DatasetObject(filepaths=df['fullpath'].to_numpy(),
                               label=df['activity'].to_numpy(),
                               transform=transform)
    dataloader = DataLoader(datasetobj, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    return dataloader
