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

def filepath_dataframe(directory):
    """
    pandas.Dataframe that contains all filepaths in the directory with targeted format,
    the folders in the directory are identified as class based on its levels

    Arguments:
    directory (str): the directory of the files

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

    start = len(directory.split('\\'))

    end = df['fullpath'].apply(lambda x: len(x.split('\\'))).max()-1

    for i in range(start,end):
        df[f'class_{i-start+1}'] = df['fullpath'].apply(lambda x: x.split('\\')[i])

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
        X = torch.Tensor(X)
        if self.transform:
            X = self.transform(X)
        y = np.int64(self.label[idx])
        return X,y
