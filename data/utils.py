import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import torch
from torch.utils.data import  Dataset


def list_all_filepaths(directory):
    """
    Return all filepaths in the directory with targeted format

    Arugments
    directory (str)
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
    directory (str)
    format (str): format of the files to be returned, must include '.'

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
        data (object): all the files from dataframe, import with load_data

        Method:
        load_data: load all files into DatasetObject
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
        y = torch.Tensor([self.label[idx]]).long()
        return X,y



# class DatasetObject:
#     """
#     Store data based on the division in the list
#     """
#     def __init__(self):
#
#         self.data = []
#
#         self.encoders = np.empty(shape=(1,3),dtype=np.ndarray)
#
#     def import_data(self, features_ls, labels_ls):
#         """
#         Store data based on the division in the list
#
#         features_ls(list(numpy.ndarray)): the features seperated by list
#         labels_ls(list(numpy.ndarray)): the labels seperated by list
#         """
#         assert len(features_ls) == len(labels_ls)
#         # Create emepty numpy array for storing data
#         self.data = np.empty(shape=(len(features_ls),3),dtype=np.ndarray)
#
#         for item_idx,(X,y) in enumerate(zip(features_ls,labels_ls)):
#
#             z = np.full_like(y,item_idx)
#
#             assert X.shape[0] == y.shape[0] == z.shape[0]
#
#             self.data[item_idx,0] = X
#             self.data[item_idx,1] = y
#             self.data[item_idx,2] = z
#
#             # print(f'index {item_idx} arrays sizes ------ X: ',X.shape,' Y: ',y.shape,' Z: ',z.shape)
#
#         # print('size of DatasetObject ------ : ',self.data.shape)
#
#         return
#
#     def __call__(self, idxs=None, return_train_sets=False):
#         """
#         Return data base on the index, or return all data if not specified
#
#         Input:
#         idxs(list/bool): the list of index that the data to be queried, if None all data will be returned
#         return_train_sets(bool): return remain data no in idxs as training set
#
#         Return:
#         data:  Return based on return_train_sets
#         if return_train_sets == True, data = (tuple(object,object,object), tuple(object,object,object))
#         else, data = (tuple(object,object,object))
#         """
#
#         if idxs != None:
#
#             X  = np.concatenate(self.data[idxs,0],axis=0)
#             y  = np.concatenate(self.data[idxs,1],axis=0)
#             z  = np.concatenate(self.data[idxs,2],axis=0)
#
#             if return_train_sets == True:
#
#                 q = [i for i in range(self.data.shape[0])]
#                 for idx in idxs:
#                     q.remove(idx)
#
#                 X_train = np.concatenate(self.data[q,0],axis=0)
#                 y_train = np.concatenate(self.data[q,1],axis=0)
#                 z_train = np.concatenate(self.data[q,2],axis=0)
#
#                 print('train set:',q,'\ttest set:',idxs)
#                 return (X_train,y_train,z_train),(X,y,z)
#
#             else:
#
#                 print('dataset:',idxs)
#
#                 return (X,y,z)
#
#         else:
#
#             X  = np.concatenate(self.data[:,0],axis=0)
#             y  = np.concatenate(self.data[:,1],axis=0)
#             z  = np.concatenate(self.data[:,2],axis=0)
#
#             return (X, y, z)
#
#     def data_transform(self,func,axis=0,col=None):
#         """
#         Apply data transformation on each index of data
#
#         func:
#         for axis = 0 : X,y,z = func(X,y,z)
#         for axis = 1 : k = func(k) where k in {X (col=0), y (col=1), z (col=2)}
#         """
#         if axis == 0:
#             for i in range(self.data.shape[0]):
#                 self.data[i,0],self.data[i,1],self.data[i,2] = func(self.data[i,0],self.data[i,1],self.data[i,2])
#         elif axis == 1:
#             for i in range(self.data.shape[0]):
#                 self.data[i,col] = func(self.data[i,col])
#         else:
#             print('No transformation is made')
#         return
#
#     def shape(self):
#         """
#         Print the shape
#         """
#         for i in range(self.data.shape[0]):
#             print(f'index {i} arrays sizes ------ X: ',self.data[i,0].shape,' Y: ',
#                   self.data[i,1].shape,' Z: ',self.data[i,2].shape)
#         print('size of DatasetObject ------ : ',self.data.shape)
#         return


def get_dataloaders(
        train_dir,
        var_dir,
        train_transform=None,
        val_transform=None,
        split=(0.5, 0.5),
        batch_size=32,
        *args, **kwargs):
    """
    This function returns the train, val and test dataloaders.
    """
    # create the datasets
    train_ds = ImageFolder(root=train_dir, transform=train_transform)
    val_ds = ImageFolder(root=var_dir, transform=val_transform)
    # now we want to split the val_ds in validation and test
    lengths = np.array(split) * len(val_ds)
    lengths = lengths.astype(int)
    left = len(val_ds) - lengths.sum()
    # we need to add the different due to float approx to int
    lengths[-1] += left

    val_ds, test_ds = random_split(val_ds, lengths.tolist())
    logging.info(f'Train samples={len(train_ds)}, Validation samples={len(val_ds)}, Test samples={len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, *args, **kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)

    return train_dl, val_dl, test_dl
