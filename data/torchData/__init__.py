import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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
