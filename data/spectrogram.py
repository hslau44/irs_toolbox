import numpy as np
import pandas as pd
import glob
import os
from os import listdir
from os.path import isfile, join


columns = [f"col_{i+1}" for i in range(501)]

def csv_to_arr(path,columns=columns):
    """Import single csv file as numpy.ndarray"""
    arr = pd.read_csv(path, names=columns).to_numpy()
    return arr

def sliding(arr,window_size=None,slide_size=None):
    """
    Slide augmentation over time

    Argument:
    array(np.ndarray): orignial array from single csv file
    window_size(int): window size
    slide_size(int): slide size

    Return:
    array (number of augmeted data,*,window_size)
    """
    mat = []
    if (window_size == None) or (slide_size == None):
        window_size = arr.shape[1]
        slide_size = arr.shape[1]
    for i in range(0, arr.shape[1]-window_size+1, slide_size):
        mat.append(arr[:,i:i+window_size])
    return np.array(mat)

def import_spectrograms(filepaths,**kwarg):
    """
    import all spectrogram in the filepaths as one array
    """
    mats = []
    for p in filepaths:
        arr = csv_to_arr(p) # ,kwarg['columns']
        arr = sliding(arr,kwarg['window_size'],kwarg['slide_size'])
        mats.append(arr)
    return np.concatenate(mats,axis=0)


def import_data(directory, columns=columns, **kwarg):
    """
    import all spectrogram in the directory
    """
    print("Import Data")
    data = {'X':[],'y':[]}
    for label in os.listdir(directory):
        files = [directory+'/'+label+'/'+i for i in os.listdir(directory+'/'+label)]
        X = import_spectrograms(files,**kwarg)
        y = np.full(X.shape[0], label)
        data['X'].append(X)
        data['y'].append(y)
    return np.concatenate(data['X']), np.concatenate(data['y'])
