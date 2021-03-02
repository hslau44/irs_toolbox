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


# def import_pair_data(directory):
#     """
#     import all spectrogram (in pair) in the directory
#     """
#     print("Importing Data ",end='')
#     data = {'X1':[],'X2':[],'y':[]}
#     for label in os.listdir(directory):
#         print('>',end='')
#         # selcting available pairs
#         pfiles_nuc1 = [f.split('.')[0][5:] for f in os.listdir(directory+'/'+label+'/'+'nuc1')]
#         pfiles_nuc2 = [f.split('.')[0][5:] for f in os.listdir(directory+'/'+label+'/'+'nuc2')]
#         available_pairs = np.intersect1d(pfiles_nuc1,pfiles_nuc2).tolist()
#         files_nuc1 = [directory+'/'+label+'/'+'nuc1'+'/'+'nuc1_'+pfilename+'.csv' for pfilename in available_pairs]
#         files_nuc2 = [directory+'/'+label+'/'+'nuc2'+'/'+'nuc2_'+pfilename+'.csv' for pfilename in available_pairs]
#         # importing
#         X1 = import_spectrograms(files_nuc1)
#         X2 = import_spectrograms(files_nuc2)
#         y = np.full(X1.shape[0], label)
#         assert X1.shape[0] == X2.shape[0]
#         data['X1'].append(X1)
#         data['X2'].append(X2)
#         data['y'].append(y)
#     print(" Complete")
#     return np.concatenate(data['X1']), np.concatenate(data['X2']), np.concatenate(data['y'])


def import_pair_data(directory,modal=['csi','nuc2']):
    """
    import all spectrogram (in pair) in the directory
    """
    print("Importing Data ",end='')
    data = {'X1':[],'X2':[],'y':[]}
    for label in os.listdir(directory):
        print('>',end='')
        # selcting available pairs
        pfiles_nuc1 = [f.split('.')[0] for f in os.listdir(directory+'/'+label+'/'+modal[0])]
        pfiles_nuc2 = [f.split('.')[0] for f in os.listdir(directory+'/'+label+'/'+modal[1])]
        available_pairs = np.intersect1d(pfiles_nuc1,pfiles_nuc2).tolist()
        files_nuc1 = [directory+'/'+label+'/'+modal[0]+'/'+pfilename+'.csv' for pfilename in available_pairs]
        files_nuc2 = [directory+'/'+label+'/'+modal[1]+'/'+pfilename+'.csv' for pfilename in available_pairs]
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

def import_dummies(size=64,class_num=6):
    X1 = np.random.rand(size,65,501)
    X2 = np.random.rand(size,65,501)
    y  = np.random.randint(0,6,size=(size,))
    return X1,X2,y


def intersect(d):
    """return array with intersected item in array in d"""
    d = np.array(d)
    ls = d[0]
    for new in d[1:]:
        ls = np.intersect1d(ls,new).tolist()
    return ls


def import_modal_data(directory,return_dict=False):
    """
    import all spectrogram (group by same modality) in the directory

    Arguments:
    directory: file in dictionary must have path: directory/label/modality/files.csv
    return_dict (bool): if True, return dictionary

    Reuturn:
    *arr (np.ndarray): return based on number of modality

    """
    print("Importing Data ")

    data = {}
    first = 0

    for label in os.listdir(directory):

        pfiles = []

        if first == 0:
            for modality in os.listdir(directory+'/'+label):
                data[modality] = []
                first += 1
                
            data['label'] = []

        for modality in os.listdir(directory+'/'+label):

            pfiles.append([f.split('.')[0] for f in os.listdir(directory+'/'+label+'/'+modality)])

        common_files = intersect(pfiles)

        print('label: ',label,'   common file: ',len(common_files))

        for modality in os.listdir(directory+'/'+label):

            files = [directory+'/'+label+'/'+modality+'/'+pfilename+'.csv' for pfilename in common_files]

            X = import_spectrograms(files)

            data[modality].append(X)

            # print(len(data[modality]))

        y = np.full(X.shape[0], label)

        data['label'].append(y)

    print("Complete")

    for key in data.keys():

        data[key] = np.concatenate(data[key])

    if return_dict:
        return data

    else:
        return (data[key] for key in data.keys())
