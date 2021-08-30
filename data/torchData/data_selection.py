import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data.transformation import resampling_


def room_split(df,trainroom=1,testroom=2):
    train = df[(df['room'] == trainroom)]
    test = df[(df['room'] == testroom)]
    return train,test


def resampling(df,columns,oversampling=False):
    if isinstance(columns,list):
        for col in columns:
            idx = resampling_(df[col].to_numpy(),oversampling=oversampling)
            df = df.iloc[idx,:]
    elif isinstance(columns,str):
        idx = resampling_(df[columns].to_numpy(),oversampling=oversampling)
        df = df.iloc[idx,:]
    return df

def random_split(df,train_size=0.8,stratify_column='activity'):
    train,test = train_test_split(df,train_size=train_size,stratify=df[stratify_column])
    return train,test

def leaveOneOut_split(df,column='person',testsub='Three',valsub=None):
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
