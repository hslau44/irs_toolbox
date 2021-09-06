import pandas as pd
from torch.utils.data import DataLoader
from data.torchData.utils import DatasetObject
from data.transformation import *



class DataLoading(object):
    """
    Load the selected dataset of dataframe (pd.DataFrame) generated by
    torchData.utils.filepath_dataframe() into corresponding torch DataLoader

    Arguments:
    transform (torchvision.transforms.transforms.Compose) - transformation pipeline
    load_data (bool) - if True, it loads all corresponing data into DataLoader *** under testing
    batch_size (int) - batch size of train and validation set
    readtype (str) - currently support 'csv' or 'npy'
    test_size (str) - if test_size = 'full', batch size for test Dataloader equals to size of testset

    Example:
    import torchvision.transforms as T
    from torchData.utils import filepath_dataframe
    from torchData import Selection,DataLoading

    dataselection = Selection(split='loov',
                                  test_sub=0.2,
                                  val_sub=0.1)

    dataloading = DataLoading(transform=T.Compose([ToTensor()]),
                              batch_size=64)

    df = filepath_dataframe(directory)
    train,val,test = dataselection(df)
    train_loader = dataloading(train)
    """

    def __init__(self,transform,batch_size,readtype='npy',load_data=False,**kwargs):
        self.transform = transform
        self.batch_size = batch_size
        self.readtype = readtype
        self.load_data = load_data
        self.kwargs = kwargs

    def __call__(self,df):
        """
        load train-validation-test set into corresponding torch DataLoader

        Arguments:
        train (pd.DataFrame) - train set
        val (pd.DataFrame) - validation set
        test (pd.DataFrame) - test set

        Return:
        train_loader (torch.utils.data.DataLoader) - torch DataLoader of the train set
        val_loader (torch.utils.data.DataLoader) - torch DataLoader of the validation set
        test_loader (torch.utils.data.DataLoader) - torch DataLoader of the test set

        """
        datasetobj = DatasetObject(filepaths=df['fullpath'].to_numpy(),
                                   label=df['activity'].to_numpy(),
                                   transform=self.transform,
                                   readtype=self.readtype)

        if self.load_data: train_obj = datasetobj.load_data()

        data_loader = DataLoader(datasetobj,batch_size=self.batch_size,**self.kwargs)

        return data_loader

class DataLoadings(object):
    """
    Load the selected dataset of dataframe (pd.DataFrame) generated by
    torchData.utils.filepath_dataframe() into corresponding torch DataLoader

    Arguments:
    transform (torchvision.transforms.transforms.Compose) - transformation pipeline
    load_data (bool) - if True, it loads all corresponing data into DataLoader *** under testing
    batch_size (int) - batch size of train and validation set
    test_size (str) - size for test set, currently support 'batch' for batch_size or 'full' for full data in test set
    readtype (str) - currently support 'csv' or 'npy'
    test_size (str) - if test_size = 'full', batch size for test Dataloader equals to size of testset

    Example:
    import torchvision.transforms as T
    from torchData.utils import filepath_dataframe
    from torchData import Selection,DataLoadings

    selection = Selection(split='loov',
                          test_sub=0.2,
                          val_sub=0.1)

    dataloading = DataLoadings(transform=T.Compose([ToTensor()]),
                               batch_size=64,
                               test_size='full',
                               readtype='csv')

    df = filepath_dataframe(directory)
    train,val,test = selection(df)
    train_loader,val_loader,test_loader = dataloading(train,val,test)
    """
    def __init__(self,transform,batch_size,test_size='batch',readtype='npy',load_data=False,**kwargs):

        self.data_loading = DataLoading(transform=transform,
                                          batch_size=batch_size,
                                          readtype=readtype,
                                          load_data=load_data,
                                          **kwargs)

        bs = len(test) if test_size == 'full' else batch_size
        self.test_loading = DataLoading(transform=transform,
                                          batch_size=bs,
                                          readtype=readtype,
                                          load_data=load_data,
                                          **kwargs)

    def __call__(self,train,val=None,test=None):
        """
        load train-validation-test set into corresponding torch DataLoader

        Arguments:
        train (pd.DataFrame) - train set
        val (pd.DataFrame) - validation set
        test (pd.DataFrame) - test set

        Return:
        train_loader (torch.utils.data.DataLoader) - torch DataLoader of the train set
        val_loader (torch.utils.data.DataLoader) - torch DataLoader of the validation set
        test_loader (torch.utils.data.DataLoader) - torch DataLoader of the test set

        """
        train_loader,val_loader,test_loader = None,None,None
        if isinstance(train,pd.core.frame.DataFrame):
            train_loader = self.data_loading(train)
        if isinstance(val,pd.core.frame.DataFrame):
            val_loader   = self.data_loading(val)
        if isinstance(test,pd.core.frame.DataFrame):
            test_loader  = self.test_loading(test)
        return train_loader, val_loader, test_loader


def DataLoadings_CnnLstmS():
    """Default DataLoadings for Resolution-Reduced CNN-LSTM"""
    transform = Transform_CnnLstmS()
    return DataLoadings(transform,batch_size=64,test_size='full',shuffle=False,num_workers=0)

def DataLoadings_CnnS():
    """Default DataLoadings for Resolution-Reduced CNN"""
    transform = Transform_CnnS()
    return DataLoadings(transform,batch_size=64,test_size='full',shuffle=False,num_workers=0)
