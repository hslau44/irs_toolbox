import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.utils import shuffle
from collections import Counter
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder


def major_vote(arr, impurity_threshold=0.01):
    """ find the element that has the majority portion in the array, depending on the threshold

    Args:
    arr: np.ndarray. The target array
    impurity_threshold: float. If array contain a portion of other elemnets and they are higher than the threshold, the function return element with the smallest portion.

    Return:
    result: the element that has the majority portion in the array, depending on the threshold
    """
    counter = Counter(list(arr.reshape(-1)))
    lowest_impurity = float(counter.most_common()[-1][-1]/arr.shape[0])
    if lowest_impurity > impurity_threshold:
        result = counter.most_common()[-1][0]
    else:
        result = counter.most_common()[0][0]
    return result


def slide_augmentation(X, y, z, window_size, slide_size, skip_labels=None):
    """ Data augmentation, create sample by sliding
    Arg:
    features: numpy.ndarray. Must have same size as labels in the first axis
    labels: numpy.ndarray. Must have same size as features in the first axis
    z: numpy.ndarray. data label in Datasetobj
    window_size: int. Create a sample with size equal to window_size
    slide_size: int. Create a sample for each slide_size in the first axis
    skip_labels: list. The labels that should be skipped

    Return:
    X: np.ndarray. Augmented features
    y: np.ndarray. Augmented labels
    z: np.ndarray. z
    """
    assert X.shape[0] == y.shape[0]

    data = {'X':[],'y':[]}

    for i in range(0, X.shape[0] -window_size+1, slide_size):

        label = major_vote(y[i:i+window_size], impurity_threshold=0.01)

        if (skip_labels != None) and (label in skip_labels):

            continue

        else:

            data['X'].append(X[i:i+window_size])

            data['y'].append(label)

    return np.array(data['X']),np.array(data['y']), z


def stacking(x):
    """Increase channel dimension from 1 to 3"""

#     if scale != False:
#         scaler = MinMaxScaler()
#         data_s = scale*scaler.fit_transform(data.reshape(len(data),-1))
#     else:
#         data_s = data
#     data_s = data_s.reshape(data_s.shape[0],-1,90,1) # change axis 1
#     return np.concatenate((data_s,data_s,data_s),axis=3)
    x = x.reshape(*x.shape,1)
    return np.concatenate((x,x,x),axis=3)

def breakpoints(ls):
    points = []
    for i in range(len(ls)-1):
        if ls[i+1] != ls[i]:
            points.append(i)
    return points

def index_resampling(arr,oversampling=True):
    """Return a list of index after resampling from array"""
    series = pd.Series(arr.reshape(-1))
    value_counts = series.value_counts()
    if oversampling == True:
        number_of_sample = value_counts.max()
        replace = True
    else:
        number_of_sample = value_counts.min()
        replace = False
    idx_ls = []
    for item in value_counts.index:
        idx_ls.append([*series[series==item].sample(n=number_of_sample,replace=replace).index])
    idx_ls = np.array(idx_ls).reshape(-1,)
    return idx_ls

def resampling(X,y,z,oversampling=True):
    """Resampling argument"""
    idx_ls = index_resampling(y,oversampling)
    X,y,z = shuffle(X[idx_ls],y[idx_ls],z[idx_ls])
    return X,y,z

def label_encode(label):
    enc = LabelEncoder()
    label = enc.fit_transform(label)
    return label, enc


# class DatasetObject:
#
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

def create_dataloaders(X_train, y_train, X_test, y_test, train_batch_sizes=64, test_batch_sizes=200):
    traindataset = TensorDataset(Tensor(X_train),Tensor(y_train).long())
    testdataset = TensorDataset(Tensor(X_test), Tensor(y_test).long())
    train_loader = DataLoader(traindataset, batch_size=train_batch_sizes, shuffle=True, num_workers=1, drop_last=True)
    test_loader = DataLoader(testdataset, batch_size=test_batch_sizes, shuffle=True, num_workers=1)
    return train_loader, test_loader
