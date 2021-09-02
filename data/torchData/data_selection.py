import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data.transformation import resampling_


def room_split(df,trainroom=1,testroom=2):
    """
    split dataset by room

    Arguments:
    df (pd.DataFrame) - dataset
    trainroom(int) - room number of the train set
    testroom(int) - room number of the test set

    Returns:
    train (pd.DataFrame) train set
    test (pd.DataFrame) test set

    """
    train = df[(df['room'] == trainroom)]
    test = df[(df['room'] == testroom)]
    return train,test


def resampling(df,columns,oversampling=False):
    """
    Resample the dataset

    Arguments:
    df (pd.DataFrame) - dataset
    columns (str/list) - columns that dataset to be resampled based on
    oversampling (bool) - if True, oversample the dataset, else undersample the dataset

    Returns:
    df (pd.DataFrame) - resampled dataset

    """
    if isinstance(columns,list):
        for col in columns:
            idx = resampling_(df[col].to_numpy(),oversampling=oversampling)
            df = df.iloc[idx,:]
    elif isinstance(columns,str):
        idx = resampling_(df[columns].to_numpy(),oversampling=oversampling)
        df = df.iloc[idx,:]
    return df

def random_split(df,train_size=0.8,stratify_column='activity'):
    """
    Random split the dataset into train-test set
    Using sklearn.model_selection.train_test_split

    Arguments:
    df (pd.DataFrame) - dataset
    train_size (float/int) -
        if type(train_size) == float, split by fraction with size of train-set equals to train_size
        if type(train_size) == int, extract number of data equals to train_size
    stratify_column (str) - stratify based on df[stratify_column]

    Returns:
    train (pd.DataFrame) train set
    test (pd.DataFrame) test set
    """
    train,test = train_test_split(df,train_size=train_size,stratify=df[stratify_column])
    return train,test

def leaveOneOut_split(df,column='person',testsub='Three',valsub=None):
    """
    Split the dataset into train-validation-test set by leave-One-Out

    Arguments:
    df (pd.DataFrame) - dataset
    column (str) - subject for train set
    testsub (str) - subject of test set
    valsub (str) - subject of validation set, if None, validation set not to be returned

    Returns:
    train (pd.DataFrame) train set
    val (pd.DataFrame) validation set
    test (pd.DataFrame) test set

    """
    assert column in df.columns, 'column is not in df'
    assert testsub in df[column].unique(), f'testsub is not in df.{column}'
    test = df[(df[column] == testsub)]
    train = df[~(df[column] == testsub)]
    if valsub:
        val = train[(train[column] == valsub)]
        train = train[~(train[column] == valsub)]
        return train,val,test
    else:
        return train,test
