from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as T
from torch.utils.data import DataLoader
from data.torchData.custom_data import filepath_dataframe
from data.torchData.utils import DatasetObject
from data.torchData.data_selection import *
from data.torchData.transformation import *


class DataSelection(object):

    def __init__(self,split,test_sub,val_sub=None,nuc=None,room=None,sample_per_class=None):
        """

        Argument:
        split  (str): spliting by either
        test_sub (str/float):
        val_sub   (str/float):
        nuc (str):
        room (str):
        sample_per_class (int/bool): ,default {None}
        """
        self.nuc = nuc
        self.room = room
        self.split = split
        # argument for 'random' split
        if self.split == 'random':
            assert val_sub < 1-test_sub
            self.train_sub = 1-test_sub
            if val_sub:
                self.val_sub = 1-val_sub/self.train_sub
        # argument for 'loov' split
        elif split == 'loov':
            self.test_sub = test_sub
            self.val_sub = val_sub
        # sub_sampling
        self.spc = sample_per_class
        # default argument
        if nuc == None:
            self.nuc = ['NUC1','NUC2']
        if room == None:
            self.room = [1,2]
        self.num_class = 0
        self.encoder = LabelEncoder()

    def __call__(self,df):

        train,val,test = None,None,None

        df['activity'] = self.encoder.fit_transform(df['activity'])

        if isinstance(self.nuc,list):
            df = df[df['nuc'].isin(self.nuc)]
        else:
            df = df[df['nuc'] == self.nuc]

        if isinstance(self.room,list):
            df = df[df['room'].isin(self.room)]
        else:
            df = df[df['room'] == self.room]

        self.num_class = df['activity'].nunique()

        if self.split == 'random':
            train,test = random_split(df,train_size=self.train_sub)
            if self.val_sub:
                train,val = random_split(train,train_size=self.val_sub)
        elif self.split == 'loov':
            if self.val_sub:
                train,val,test = leaveOneOut_split(df,testsub=self.test_sub,valsub=self.val_sub)
            else:
                train,test = leaveOneOut_split(df,testsub=self.test_sub,valsub=None)

        if isinstance(self.spc,int):
            train = resampling(train,'activity',oversampling=False)
            train,_ = random_split(train,train_size=self.num_class*self.spc)
        test = resampling(test,'activity',oversampling=False)


        return train,val,test


def dataSelection_Set1():
    """To test the model under lab environemnt by leave-one-person-out validation"""
    return DataSelection(split='loov',test_sub='One',val_sub='Four',nuc='NUC1',room=1,sample_per_class=None)

def dataSelection_Set2():
    """To test the model under lab environemnt by random-split validation"""
    return DataSelection(split='random',test_sub=0.2,val_sub=0.1,nuc='NUC1',room=1,sample_per_class=None)

def dataSelection_Set3():
    """To test if combine data get better generalization"""
    return DataSelection(split='loov',test_sub='One',val_sub='Four',nuc=None,room=None,sample_per_class=None)

def dataSelection_Set4(spc=5):
    """To test model performance under low number of sample"""
    return DataSelection(split='loov',test_sub='One',val_sub='Four',nuc='NUC1',room=1,sample_per_class=spc)


def dataSelection_Set5():
    """To test the model under unseen environemnt"""
    def func(df):
        lb = LabelEncoder()
        df['activity'] = lb.fit_transform(df['activity'])
        df = df[df['nuc'] == 'NUC1']
        train = df[df['room'] == 1]
        test  = df[df['room'] == 2]

        train,val, _ = leaveOneOut_split(train,testsub='One',valsub='Four')
        _ , _ , test = leaveOneOut_split(test,testsub='One',valsub='Four')

        test = resampling(test,'activity',oversampling=False)
        return train,val,test
    return func



class DataLoading(object):

    def __init__(self,transform,batch_size=64,test_size='batch',shuffle=False,num_workers=0):
        self.transform = transform
        self.batch_size = batch_size
        self.test_size = test_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def __call__(self,train,val=None,test=None):

        train_loader,val_loader,test_loader = None, None, None

        train_obj = DatasetObject(filepaths=train['fullpath'].to_numpy(),
                                  label=train['activity'].to_numpy(),
                                  transform=self.transform)
        train_loader = DataLoader(train_obj, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

        if val:
            val_obj = DatasetObject(filepaths=val['fullpath'].to_numpy(),
                                      label=val['activity'].to_numpy(),
                                      transform=self.transform)
            val_loader = DataLoader(val_obj, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

        if test:
            test_obj = DatasetObject(filepaths=test['fullpath'].to_numpy(),
                                      label=test['activity'].to_numpy(),
                                      transform=self.transform)

            if self.test_size == 'full':
                test_loader = DataLoader(test_obj, batch_size=test.shape[0], shuffle=False, num_workers=self.num_workers)
            else:
                test_loader = DataLoader(test_obj, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader,val_loader,test_loader


def dataLoading_CnnLstmS(df):
    """dataLoading_cnnLstmS"""
    transform = T.Compose([ReduceRes(),Unsqueeze(),ToStackImg(25)])
    return DataLoading(transform,batch_size=64,test_size='full',shuffle=False,num_workers=0)

def dataLoading_CnnS(df):
    """dataLoading_CnnS"""
    transform = T.Compose([ReduceRes(),Unsqueeze()])
    return DataLoading(transform,batch_size=64,test_size='full',shuffle=False,num_workers=0)
