import numpy as np
import pandas as pd
import torch
from data.torchData.utils import list_all_filepaths,DatasetObject


def filepath_dataframe(directory,split='\\'):

    filepaths = list_all_filepaths(directory)

    df = pd.DataFrame(data=filepaths,columns=['fullpath'])

    df['filename'] = df['fullpath'].apply(lambda x: x.split(split)[-1])

    df['exp'] = df['filename'].apply(lambda x: int(x.split('_')[1]))

    df['person'] = df['filename'].apply(lambda x: x.split('_')[3])

    df['room'] = df['filename'].apply(lambda x: int(x.split('_')[5]))

    df['activity'] = df['filename'].apply(lambda x: x.split('_')[6])

    df['index'] = df['filename'].apply(lambda x: int(x.split('_')[8]))

    df['nuc'] = df['filename'].apply(lambda x: x.split('_')[9].split('.')[0])

    df['key'] = df['filename'].apply(lambda x: x[:-9])

    df.pop('filename')

    return df

def nucPaired_fpDataframe(dataframe):

    nuc1 = dataframe[dataframe['nuc'] == 'NUC1'].drop('nuc',axis=1)

    nuc2 = dataframe[dataframe['nuc'] == 'NUC2'][['key','fullpath']]

    return pd.merge(nuc1,nuc2,on='key')


class PairDataset(DatasetObject):

    def __init__(self,filepaths,filepaths2,transform=None):
        """
        Customized PyTorch Dataset, currently only support csv files

        Attribute:
        filepaths (numpy.ndarray): 1D array of filepaths on view1, file must be in csv format
        filepaths2 (numpy.ndarray): 1D array of filepaths on view2, file must be in csv format
        transfrom (torchvision.transforms): data transformation pipeline

        """
        super().__init__(filepaths=filepaths,
                         label=None,
                         transform=transform)

        self.filepaths2 = filepaths2

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        fp1 = self.filepaths[idx]
        X1 = pd.read_csv(fp1,header=None).to_numpy()
        fp2 = self.filepaths[idx]
        X2 = pd.read_csv(fp2,header=None).to_numpy()
        if self.transform:
            X1 = self.transform(X1)
            X2 = self.transform(X2)
        return X1,X2
