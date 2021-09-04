import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter


def resampling_(arr,oversampling=True):
    """Return a list of index after resampling from array"""
    """
    Stratified Sampling of array, return the index for indexing array

    Parameters:
    arr (numpy.ndarray<obj>): input array, array should be 1 dimension to work properly. Elements type must be consistent and with type {int,str}. Array undergoes flatten.
    oversampling (bool): {True,False}, oversampling if True, else undersampling

    Returns:
    idx_ls (numpy.ndarray<obj>): index of the array that undergoes Stratified Sampling

    """
    series = pd.Series(arr.reshape(-1))
    value_counts = series.value_counts()
    if oversampling == True:
        number_of_sample = value_counts.max()
        replace = True
    else:
        number_of_sample = value_counts.min()
        replace = False
    idx_ls = []
    for item in value_counts.index:
        idx_ls.append([*series[series==item].sample(n=number_of_sample,replace=replace).index])
    idx_ls = np.array(idx_ls).reshape(-1,)
    return idx_ls


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

def random_split(df,stratify_column,train_size=0.8):
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
    assert stratify_column in df.columns, 'stratify column is not in df'
    train,test = train_test_split(df,train_size=train_size,stratify=df[stratify_column])
    return train,test

def leaveOneOut_split(df,column,testsub):
    """
    Split the dataset into train-test set by leave-One-Out

    Arguments:
    df (pd.DataFrame) - dataset
    column (str) - subject for train set
    testsub (str) - subject of test set

    Returns:
    train (pd.DataFrame) train set
    test (pd.DataFrame) test set

    """
    assert column in df.columns, 'column is not in df'
    assert testsub in df[column].unique(), f'testsub is not in df.{column}'
    test = df[(df[column] == testsub)]
    train = df[~(df[column] == testsub)]
    return train,test


# ------------------------------------------OLD----------------------------------------------------------

def selections(*arg,**kwarg):
    """
    **To be tested**
    self implementation of train test split from sklearn, kwarg "p" for train size"""
    size = int(arg[0].shape[0]*kwarg['p'])
    index = np.arange(0,arg[0].shape[0])
    test_selection = np.random.choice(index,size,replace=False)
    train_selection = np.array([i for i in index if i not in test_selection])
    return [i[train_selection] for i in arg],[i[test_selection] for i in arg]

def major_vote_(arr, impurity_threshold=0.01):
    """
    **To be tested**

    Returns the elment of an array which has the majority

    Parameters:
    arr (numpy.ndarray<obj>): input array,  elements type must be consistent and with type {int,str}. Array undergoes flatten.
    impurity_threshold (float): if the frequency of the least common element is bigger than impurity_threshold, return to the least common element

    Returns:
    result (obj): the elment of an array which has the majority

    """
    counter = Counter(list(arr.reshape(-1)))
    lowest_impurity = float(counter.most_common()[-1][-1]/arr.shape[0])
    if lowest_impurity > impurity_threshold:
        result = counter.most_common()[-1][0]
    else:
        result = counter.most_common()[0][0]
    return result


def major_vote(arr,impurity=0.01):
    """
    **To be tested**

    find the element that has the majority portion in the array, depending on the threshold

    Args:
    arr: np.ndarray. The target array
    impurity_threshold: float. If array contain a portion of other elemnets and they are higher than the threshold, the function return element with the smallest portion.

    Return:
    result: the element that has the majority portion in the array, depending on the threshold
    """
    assert len(arr.shape) == 2, "must have only 2 dimension"
    new_arr = np.zeros(arr.shape[0])
    for i in range(len(arr)):
        new_arr[i] = major_vote_(arr[i],impurity)
    return new_arr

def where(*arrays,condition):
    """
    **To be tested**
    
    select item in multi arguments of numpy.ndarrays based on condition, return arguments of numpy.ndarrays"""
    return [arr[np.where(condition)] for arr in arrays]
