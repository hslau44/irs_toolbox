import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def list_all_filepaths(directory):
    """
    Return all filepaths in the directory with targeted format

    Arugments:
    directory (str): the directory of the files

    Return
    filepaths (list): all filepaths of the files
    """
    filepaths = []
    for r, d, f in os.walk(directory):
        for item in f:
            filepaths.append(os.path.join(r, item))
    return filepaths

def filepath_dataframe(directory,split='\\'):
    """
    pandas.Dataframe that contains all filepaths in the directory with targeted format,
    the folders in the directory are identified as class based on its levels

    Arguments:
    directory (str): the directory of the files
    split (str)

    Return:
    dataframe (pd.DataFrame): a dataframe with filespath as the first columns, number of columns is
    equal to the available level of the folders

    Example:
    for image with the fllowing filepath "directory/cat/black/blackcat.jpg"
    the dataframe would be shown as below:

    fullpath                         | class_1 | class_2
    ---------------------------------+---------+---------
    directory/cat/black/blackcat.jpg | cat     | black

    """

    filepaths = list_all_filepaths(directory)

    df = pd.DataFrame(data=filepaths,columns=['fullpath'])

    start = len(directory.split(split))

    end = df['fullpath'].apply(lambda x: len(x.split(split))).max()-1

    for i in range(start,end):
        df[f'class_{i-start+1}'] = df['fullpath'].apply(lambda x: x.split(split)[i])

    return df


class DatasetObject(Dataset):

    def __init__(self,filepaths,label,transform=None):
        """
        Customized PyTorch Dataset, currently only support csv files

        Attribute:
        filepaths (numpy.ndarray): 1D array of filepaths, file must be in csv format
        label (numpy.ndarray): 1D array of corresponding label
        transfrom (torchvision.transforms): data transformation pipeline

        """
        # assert len(filepaths) == len(label)
        self.filepaths = filepaths
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        fp = self.filepaths[idx]
        X = pd.read_csv(fp,header=None).to_numpy()
        if self.transform:
            X = self.transform(X)
        y = np.int64(self.label[idx])
        return X,y

    def load_data(self):
        X,Y = [],[]
        for idx in range(self.__len__()):
            x,y = self.__getitem__(idx)
            X.append(x)
            Y.append(y)
            if idx%(self.__len__()/20) == 0: print('>',end='')
        X = torch.Tensor(X)
        Y = torch.Tensor(Y).long()
        tensordataset = torch.utils.data.TensorDataset(X,Y)
        print('')
        return tensordataset

class DatasetObject_Npy(DatasetObject):

    def __init__(self,filepaths,label,transform=None):
        """
        Customized PyTorch Dataset, currently only support npy files

        Attribute:
        filepaths (numpy.ndarray): 1D array of filepaths, file must be in npy format
        label (numpy.ndarray): 1D array of corresponding label
        transfrom (torchvision.transforms): data transformation pipeline

        """
        # assert len(filepaths) == len(label)
        super(DatasetObject_Npy, self).__init__(filepaths,label,transform)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        fp = self.filepaths[idx]
        X = np.load(fp)
        if self.transform:
            X = self.transform(X)
            X = torch.from_numpy(X).float()
        y = np.int64(self.label[idx])
        return X,y

def breakpoints(ls):
    """find the index where element in ls(list) changes"""
    points = []
    for i in range(len(ls)-1):
        if ls[i+1] != ls[i]:
            points.append(i)
    return points
