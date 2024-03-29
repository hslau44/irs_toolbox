import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def read_file(fp,readtype):
    """
    Return numpy.ndarray from fp by readtype: currently support 'csv' or 'npy'
    """
    if readtype == 'csv':
        return pd.read_csv(fp,header=None).to_numpy()
    elif readtype == 'npy':
        return np.load(fp)
    else:
        raise Exception("currently only support 'csv' or 'npy'")


class DatasetObject(Dataset):

    def __init__(self,filepaths,label=None,transform=None,readtype='npy'):
        """
        Customized PyTorch Dataset

        Attribute:
        filepaths (numpy.ndarray): 1D array of filepaths, file must be in csv format
        label (numpy.ndarray): 1D array of corresponding label
        transfrom (torchvision.transforms): data transformation pipeline
        readtype(str): currently support 'csv' and 'npy'
        """
        if isinstance(label,np.ndarray):
            assert len(filepaths) == len(label)
        assert readtype in ['csv','npy']
        self.filepaths = filepaths
        self.readtype = readtype
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # filepath
        fp = self.filepaths[idx]
        X = read_file(fp,self.readtype)
        # transform
        if self.transform:
            X = self.transform(X)
        # if self.readtype == 'npy':
        X = torch.from_numpy(X).float()
        # label
        if isinstance(self.label,np.ndarray):
            y = np.int64(self.label[idx])
            return X,y
        else:
            return X

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

class PairDataset(DatasetObject):

    def __init__(self,filepaths,filepaths2,label=None,transform=None,readtype='npy'):
        """
        Customized PyTorch Dataset, currently only support csv files

        Attribute:
        filepaths (numpy.ndarray): 1D array of filepaths on view1, file must be in csv format
        filepaths2 (numpy.ndarray): 1D array of filepaths on view2, file must be in csv format
        transfrom (torchvision.transforms): data transformation pipeline

        """
        super().__init__(filepaths=filepaths,
                         label=label,
                         transform=transform,
                         readtype=readtype)

        self.filepaths2 = filepaths2

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        fp1 = self.filepaths[idx]
        fp2 = self.filepaths2[idx]
        X1 = read_file(fp1,self.readtype)
        X2 = read_file(fp2,self.readtype)
        if self.transform:
            X1 = self.transform(X1)
            X2 = self.transform(X2)
        # if self.readtype == 'npy':
        X1 = torch.from_numpy(X1).float()
        X2 = torch.from_numpy(X2).float()
        # label
        if isinstance(self.label,np.ndarray):
            y = np.int64(self.label[idx])
            return X1,X2,y
        else:
            return X1,X2

class TransformersDataset(DatasetObject):

    def __init__(self,filepaths,label=None,transform=None,readtype='npy'):
        super().__init__(filepaths=filepaths,
                         label=label,
                         transform=transform,
                         readtype=readtype)

    def __getitem__(self, idx):
        items = super().__getitem__(idx)
        dic = {}
        if isinstance(self.label,np.ndarray):
            X,y = items
            dic['input_values'] = torch.Tensor(X)
            dic['label'] = torch.Tensor(y)
        else:
            X = items
            dic['input_values'] = torch.Tensor(X)
        return dic

    def load_data(self):
        return None
