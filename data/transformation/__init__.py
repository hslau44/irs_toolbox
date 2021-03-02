import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.utils import shuffle
from collections import Counter
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

# ---------------------------------sub------------------------------------------

def major_vote_(arr, impurity_threshold=0.01):
    """
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


# ---------------------------------main------------------------------------------

def where(*arrays,condition):
    """select item in multi arguments of numpy.ndarrays based on condition, return arguments of numpy.ndarrays"""
    return [arr[np.where(condition)] for arr in arrays]

def slide(*arrays, window_size, slide_size):
    """
    slide the 0 axis of the arrays to create new data with window_size

    Parameters:
    arrays (numpy.ndarray<obj>): input array
    window_size (int): window_size
    slide_size (int): skipping factor

    Returns:
    arrays: with size (number of augmented sample,window_size,...)

    """
    length = arrays[0].shape[0]

    for arr in arrays:
        assert arr.shape[0] == length, "all array in arrays must has same length"

    assert window_size <= length, "window size must not be larger than the length of arrays"

    new_arrays = []

    for arr in arrays:

        data = []

        for i in range(0, length-window_size+1, slide_size):

            d = arr[i:i+window_size]

            data.append(d)

        data = np.array(data)

        new_arrays.append(data)

    return new_arrays

def major_vote(arr,impurity=0.01):
    """ find the element that has the majority portion in the array, depending on the threshold

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

def resampling(*arrays,labels,oversampling=True):
    """Resampling argument"""

    idx_ls = resampling_(labels,oversampling)
    new_arrays = [arr[idx_ls] for arr in arrays]
    return new_arrays

# ---------------------------------helper------------------------------------------

def slide_augmentation(X, y, z, window_size, slide_size, skip_labels=None):
    """helper function of slide for DatasetObj"""
    X, y, z = slide(X, y, z, window_size=window_size, slide_size=slide_size)
    y = major_vote(y,impurity=0.01)
    if skip_labels != None:
        for lb in skip_labels:
            X, y, z = where(X, y, z, condition=(y!=lb))
    return X, y, z

def stacking(x):
    """Increase channel dimension from 1 to 3"""
    x = x.reshape(*x.shape,1)
    return np.concatenate((x,x,x),axis=3)

def breakpoints(ls):
    """find the index where element in ls(list) changes"""
    points = []
    for i in range(len(ls)-1):
        if ls[i+1] != ls[i]:
            points.append(i)
    return points

def selections(*arg,**kwarg):
    """self implementation of train test split from sklearn, kwarg "p" for train size"""
    size = int(arg[0].shape[0]*kwarg['p'])
    index = np.arange(0,arg[0].shape[0])
    test_selection = np.random.choice(index,size,replace=False)
    train_selection = np.array([i for i in index if i not in test_selection])
    return [i[train_selection] for i in arg],[i[test_selection] for i in arg]


def label_encode(label,enc=None):
    """return label-encoded array and its LabelEncoder"""
    if enc == None:
        enc = LabelEncoder()
        enc.fit(label)
    label = enc.transform(label)
    return label, enc
