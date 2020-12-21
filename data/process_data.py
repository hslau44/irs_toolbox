import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.utils import shuffle
from collections import Counter
from torch.utils.data import Dataset

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

def slide_augmentation(features,labels,window_size,slide_size,skip_labels=None):
    """ Data augmentation, create sample by sliding
    Arg:
    features: numpy.ndarray. Must have same size as labels in the first axis
    labels: numpy.ndarray. Must have same size as features in the first axis
    window_size: int. Create a sample with size equal to window_size
    slide_size: int. Create a sample for each slide_size in the first axis
    skip_labels: list. The labels that should be skipped

    Reuturn:
    X: numpy.ndarray. Augmented features
    y: np.ndarray. Augmented labels
    """
    assert features.shape[0] == labels.shape[0]
    X,y,z = [],[],[]
    for i in range(0,len(features)-window_size+1,slide_size):
        label = major_vote(labels[i:i+window_size],impurity_threshold=0.01)
        if (skip_labels != None) and (label in skip_labels):
            continue
        else:
            X.append(features[i:i+window_size])
            y.append(label)
    return np.array(X),np.array(y)

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

def seperate_dataframes(df):
    features_ls,labels_ls = [],[]
    for user in df['user'].unique():
        dataframe = df[df['user']==user]
        features = dataframe[[f'amp_{i}' for i in range(1,91)]].to_numpy()
#         features = MinMaxScaler().fit_transform(features)
        features_ls.append(features)
        label = dataframe[['label']].to_numpy()
        labels_ls.append(label)
    return features_ls,labels_ls

# class DatasetObject:
#
#     def __init__(self):
#         self.data = []
#         self.encoders = np.empty(shape=(1,3),dtype=np.ndarray)
#
#     def import_data(self,features_ls,labels_ls,window_size,slide_size,skip_labels=None):
# #         self.data = []
# #         assert len(features_ls) == len(labels_ls)
# #         for item_idx,(features,labels) in enumerate(zip(features_ls,label_ls)):
# #             X,y = slide_augmentation(features,labels,window_size,slide_size,skip_labels)
# #             z = np.full_like(y,item_idx)
# #             assert X.shape[0] == y.shape[0] == z.shape[0]
# #             self.data.append([X,y,z])
# #             print(f'index {item_idx} arrays sizes ------ X: ',X.shape,' Y: ',y.shape,' Z: ',z.shape)
# #         self.data = np.array(self.data,dtype=object)
# #         print('size of DatasetObject ------ : ',self.data.shape)
#
#         assert len(features_ls) == len(labels_ls)
#         self.data = np.empty(shape=(len(features_ls),3),dtype=np.ndarray)
#         for item_idx,(features,labels) in enumerate(zip(features_ls,labels_ls)):
#             X,y = slide_augmentation(features,labels,window_size,slide_size,skip_labels)
#             z = np.full_like(y,item_idx)
#             assert X.shape[0] == y.shape[0] == z.shape[0]
#             self.data[item_idx,0] = X
#             self.data[item_idx,1] = y
#             self.data[item_idx,2] = z
#             print(f'index {item_idx} arrays sizes ------ X: ',X.shape,' Y: ',y.shape,' Z: ',z.shape)
#         print('size of DatasetObject ------ : ',self.data.shape)
#         return
#
#     def __call__(self):
#         X  = np.concatenate(self.data[:,0],axis=0)
#         y  = np.concatenate(self.data[:,1],axis=0)
#         z  = np.concatenate(self.data[:,2],axis=0)
#         return X,y,z
#
#     def data_transform(self,func,axis=0,col=None):
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
#         for i in range(self.data.shape[0]):
#             print(f'index {i} arrays sizes ------ X: ',self.data[i,0].shape,' Y: ',
#                   self.data[i,1].shape,' Z: ',self.data[i,2].shape)
#         print('size of DatasetObject ------ : ',self.data.shape)
#         return
#
#     def query(self,testset):
#         q = [i for i in range(self.data.shape[0])]
#         for t in testset:
#             q.remove(t)
#         print('train set:',q,'\ttest set:',testset)
#         X_train = np.concatenate(self.data[q,0],axis=0)
#         y_train = np.concatenate(self.data[q,1],axis=0)
#         z_train = np.concatenate(self.data[q,2],axis=0)
#         X_test  = np.concatenate(self.data[testset,0],axis=0)
#         y_test  = np.concatenate(self.data[testset,1],axis=0)
#         z_test  = np.concatenate(self.data[testset,2],axis=0)
#         del q
#         return  [X_train,y_train,z_train],[X_test,y_test,z_test]
#
#     def label_encode(self,col=1,encoder=OneHotEncoder()):
#         if self.encoders[0,col] != None:
#             print('array encoded')
#             return
#         self.encoders[0,col] = encoder
#         self.encoders[0,col].fit(np.concatenate(self.data[:,col],axis=0).reshape(-1,1))
#         for i in range(self.data.shape[0]):
#             self.data[i,col] = self.encoders[0,col].transform(self.data[i,col].reshape(-1,1)).toarray()
#         return encoder

class DatasetObject(Dataset):

    def __init__(self):
        self.data = []
        self.encoders = np.empty(shape=(1,3),dtype=np.ndarray)

    def import_data(self,features_ls,labels_ls,window_size,slide_size,skip_labels=None):
#         self.data = []
#         assert len(features_ls) == len(labels_ls)
#         for item_idx,(features,labels) in enumerate(zip(features_ls,label_ls)):
#             X,y = slide_augmentation(features,labels,window_size,slide_size,skip_labels)
#             z = np.full_like(y,item_idx)
#             assert X.shape[0] == y.shape[0] == z.shape[0]
#             self.data.append([X,y,z])
#             print(f'index {item_idx} arrays sizes ------ X: ',X.shape,' Y: ',y.shape,' Z: ',z.shape)
#         self.data = np.array(self.data,dtype=object)
#         print('size of DatasetObject ------ : ',self.data.shape)

        assert len(features_ls) == len(labels_ls)
        self.data = np.empty(shape=(len(features_ls),3),dtype=np.ndarray)
        for item_idx,(features,labels) in enumerate(zip(features_ls,labels_ls)):
            X,y = slide_augmentation(features,labels,window_size,slide_size,skip_labels)
            z = np.full_like(y,item_idx)
            assert X.shape[0] == y.shape[0] == z.shape[0]
            self.data[item_idx,0] = X
            self.data[item_idx,1] = y
            self.data[item_idx,2] = z
            print(f'index {item_idx} arrays sizes ------ X: ',X.shape,' Y: ',y.shape,' Z: ',z.shape)
        print('size of DatasetObject ------ : ',self.data.shape)
        return

    # def __call__(self):
    #     X  = np.concatenate(self.data[:,0],axis=0)
    #     y  = np.concatenate(self.data[:,1],axis=0)
    #     z  = np.concatenate(self.data[:,2],axis=0)
    #     return X,y,z

    def data_transform(self,func,axis=0,col=None):
        if axis == 0:
            for i in range(self.data.shape[0]):
                self.data[i,0],self.data[i,1],self.data[i,2] = func(self.data[i,0],self.data[i,1],self.data[i,2])
        elif axis == 1:
            for i in range(self.data.shape[0]):
                self.data[i,col] = func(self.data[i,col])
        else:
            print('No transformation is made')
        return

    def __len__(self):
        for i in range(self.data.shape[0]):
            print(f'index {i} arrays sizes ------ X: ',self.data[i,0].shape,' Y: ',
                  self.data[i,1].shape,' Z: ',self.data[i,2].shape)
        print('size of DatasetObject ------ : ',self.data.shape)
        return

    def __getitem__(self, idxs, return_sets=False):
        X_test  = np.concatenate(self.data[idxs,0],axis=0)
        y_test  = np.concatenate(self.data[idxs,1],axis=0)
        z_test  = np.concatenate(self.data[idxs,2],axis=0)
        if return_sets == True:
            q = [i for i in range(self.data.shape[0])]
            for idx in idxs:
                q.remove(idx)
            X_train = np.concatenate(self.data[q,0],axis=0)
            y_train = np.concatenate(self.data[q,1],axis=0)
            z_train = np.concatenate(self.data[q,2],axis=0)
            print('train set:',q,'\ttest set:',idxs)
            return (X_train,y_train,z_train),(X_test,y_test,z_test)
        else:
            print('dataset:',idxs)
            return (X_test,y_test,z_test)

    def label_encode(self,col=1,encoder=OneHotEncoder()):
        if self.encoders[0,col] != None:
            print('array encoded')
            return
        self.encoders[0,col] = encoder
        self.encoders[0,col].fit(np.concatenate(self.data[:,col],axis=0).reshape(-1,1))
        for i in range(self.data.shape[0]):
            self.data[i,col] = self.encoders[0,col].transform(self.data[i,col].reshape(-1,1)).toarray()
        return encoder
