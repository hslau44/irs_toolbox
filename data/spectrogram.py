import numpy as np
import pandas as pd
import glob
import os
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import MinMaxScaler




def csv_to_arr(path):
    """Import single csv file as numpy.ndarray"""
    arr = pd.read_csv(path, header=None).to_numpy()
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

def normalize_arr(arr,axis=0):
    if axis == 0:
        arr = MinMaxScaler().fit_transform(arr)
    elif axis == 1:
        arr = MinMaxScaler().fit_transform(arr.transpose()).transpose()
    return arr


def import_spectrograms(filepaths):
    """
    import all spectrogram in the filepaths as one array
    """
    mats = []
    for p in filepaths:
        arr = csv_to_arr(p)
        # arr = normalize_arr(arr,axis=0)
        arr = arr.reshape(-1,*arr.shape)
        mats.append(arr)
    return np.concatenate(mats,axis=0)


def import_data(directory):
    """
    import all spectrogram in the directory
    """
    print("Importing Data ",end='')
    data = {'X':[],'y':[]}
    for label in os.listdir(directory):
        print('>',end='')
        files = [directory+'/'+label+'/'+i for i in os.listdir(directory+'/'+label)]
        X = import_spectrograms(files)
        y = np.full(X.shape[0], label)
        data['X'].append(X)
        data['y'].append(y)
    print(" Complete")
    return np.concatenate(data['X']), np.concatenate(data['y'])


def import_pair_data(directory):
    """
    import all spectrogram (in pair) in the directory
    """
    print("Importing Data ",end='')
    data = {'X1':[],'X2':[],'y':[]}
    for label in os.listdir(directory):
        print('>',end='')
        # selcting available pairs
        pfiles_nuc1 = [f.split('.')[0][5:] for f in os.listdir(directory+'/'+label+'/'+'nuc1')]
        pfiles_nuc2 = [f.split('.')[0][5:] for f in os.listdir(directory+'/'+label+'/'+'nuc2')]
        available_pairs = np.intersect1d(pfiles_nuc1,pfiles_nuc2).tolist()
        files_nuc1 = [directory+'/'+label+'/'+'nuc1'+'/'+'nuc1_'+pfilename+'.csv' for pfilename in available_pairs]
        files_nuc2 = [directory+'/'+label+'/'+'nuc2'+'/'+'nuc2_'+pfilename+'.csv' for pfilename in available_pairs]
        # importing
        X1 = import_spectrograms(files_nuc1)
        X2 = import_spectrograms(files_nuc2)
        y = np.full(X1.shape[0], label)
        assert X1.shape[0] == X2.shape[0]
        data['X1'].append(X1)
        data['X2'].append(X2)
        data['y'].append(y)
    print(" Complete")
    return np.concatenate(data['X1']), np.concatenate(data['X2']), np.concatenate(data['y'])


def import_CsiPwr_data(directory):
    """
    import all spectrogram (in pair) in the directory
    """
    print("Importing Data ",end='')
    data = {'X1':[],'X2':[],'y':[]}
    for label in os.listdir(directory):
        print('>',end='')
        # selcting available pairs
        pfiles_nuc1 = [f.split('.')[0] for f in os.listdir(directory+'/'+label+'/'+'csi')]
        pfiles_nuc2 = [f.split('.')[0] for f in os.listdir(directory+'/'+label+'/'+'pwr')]
        available_pairs = np.intersect1d(pfiles_nuc1,pfiles_nuc2).tolist()
        files_nuc1 = [directory+'/'+label+'/'+'csi'+'/'+pfilename+'.csv' for pfilename in available_pairs]
        files_nuc2 = [directory+'/'+label+'/'+'pwr'+'/'+pfilename+'.csv' for pfilename in available_pairs]
        # importing
        X1 = import_spectrograms(files_nuc1)
        X2 = import_spectrograms(files_nuc2)
        y = np.full(X1.shape[0], label)
        assert X1.shape[0] == X2.shape[0]
        data['X1'].append(X1)
        data['X2'].append(X2)
        data['y'].append(y)
    print(" Complete")
    return np.concatenate(data['X1']), np.concatenate(data['X2']), np.concatenate(data['y'])
