import os
from os import listdir
from os.path import isfile, join

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


def list_filepaths(directory,format_='.csv'):
    """
    Return all filepaths in the directory with targeted format

    Arugments
    directory (str)
    format (str): format of the files to be returned, must include '.'
    """
    filepaths_ = []
    for r, d, f in os.walk(directory):
        for item in f:
            if format_ in item:
                filepaths_.append(os.path.join(r, item))
    return filepaths_

def dataframe_filepaths(directory,format_='.csv'):
    """
    pandas.Dataframe that contains all filepaths in the directory with targeted format,
    the folders in the directory are identified as class based on its levels

    Arguments:
    directory (str)
    format (str): format of the files to be returned, must include '.'

    Return:
    dataframe (pd.DataFrame): a dataframe with filespath as the first columns, number of columns is
    equal to the available level of the folders

    Example:
    for image with the fllowing filepath "directory/cat/black/blackcat.jpg"
    the dataframe would be shown as below:

    filepaths                        | class_1 | class_2
    ---------------------------------+---------+---------
    directory/cat/black/blackcat.jpg | cat     | black

    """
    filepaths = list_filepaths(directory,format_=format_)

    df = pd.DataFrame(data=filepaths,columns=['filepath'])

    max_col = df['filepath'].apply(lambda x: len(x.split('\\'))).max()-2

    for i in range(1,max_col+1):
        df[f'class_{i}'] = df['filepath'].apply(lambda x: x.split('\\')[i])

    return df

class DatasetObject(Dataset):

    def __init__(self,dataframe,label=None,transform=None):
        """
        Customized PyTorch Dataset, currently only support csv files

        Attribute:
        dataframe (pandas.Dataframe): dataframe contains all filepaths at the first columns, referred to dataframe_filepaths
        label (str): the name for the column in dataframe to be return as the label for the corresponding file
        transfrom (torchvision.transforms): data transformation pipeline
        data (object): all the files from dataframe, import with load_data

        Method:
        load_data: load all files into DatasetObject
        """
        self.dataframe = dataframe
        self.transform = transform
        self.label = label
        self.data = None

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        X = pd.read_csv(dataframe.loc[idx,'filepath'],header=None).to_numpy()
        if self.transform:
            X = self.transform(X)
        if self.label:
            y = dataframe.loc[idx,self.label]
            return X,y
        else:
            return X

    def load_data(self):
        data = []
        for idx in range(len(self.dataframe)):
            data.append(pd.read_csv(dataframe.loc[idx,'filepath'],header=None).to_numpy())
        self.data = np.array(data)
        return

if __name__ == '__main__':
    location = 'E://external_data/opera_csi/experiment_data/exp_4/spectrogram'
    dataframe = dataframe_filepaths(location)
    transform = T.Compose([T.ToTensor()]) # ,T.ToPILImage()
    datasetobj = DatasetObject(dataframe,label='class_1',transform=transform)
    dataloader = DataLoader(datasetobj, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
