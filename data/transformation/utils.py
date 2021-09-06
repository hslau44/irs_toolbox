import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


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






def label_encode(label,enc=None):
    """return label-encoded array and its LabelEncoder, or apply a predefined LabelEncoder on the label

    Arguments:
    label (np.ndarry): label, must be flatten
    enc (sklearn.preprocessing.LabelEncoder): the optional predefined LabelEncoder

    Return
    label (np.ndarry): encoded label
    enc (sklearn.preprocessing.LabelEncoder):  newly-create/predefined LabelEncoder
    """
    if enc == None:
        enc = LabelEncoder()
        enc.fit(label)
    label = enc.transform(label)
    return label, enc
