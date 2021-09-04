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
    test_size (str) - if test_size = 'full', batch size for test Dataloader equals to size of testset

    Example:
    import torchvision.transforms as T
    from torchData.utils import filepath_dataframe
    from torchData import DataSelection,DataLoading

    dataselection = DataSelection(split='loov',
                                  test_sub=0.2,
                                  val_sub=0.1)

    dataloading = DataLoading(transform=T.Compose([ToTensor()]),
                              batch_size=64)

    df = filepath_dataframe(directory)
    train,val,test = dataselection(df)
    train_loader,val_loader,test_loader = dataloading(train,val,test)
    """

    def __init__(self,transform,load_data=False,batch_size=64,test_size='batch',**kwargs):
        self.transform = transform
        self.load_data = load_data
        self.batch_size = batch_size
        self.test_size = test_size
        self.kwargs = kwargs

        self.readtype = kwargs.get('readtype','npy')

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
        train_loader,val_loader,test_loader = None, None, None

        train_obj = DatasetObject(filepaths=train['fullpath'].to_numpy(),
                                  label=train['activity'].to_numpy(),
                                  transform=self.transform,
                                  readtype=self.readtype)

        if self.load_data: train_obj = train_obj.load_data()

        train_loader = DataLoader(train_obj,batch_size=self.batch_size,**self.kwargs)

        if isinstance(val,pd.DataFrame):
            val_obj = DatasetObject(filepaths=val['fullpath'].to_numpy(),
                                      label=val['activity'].to_numpy(),
                                      transform=self.transform,
                                      readtype=self.readtype)

            if self.load_data: val_obj = val_obj.load_data()

            val_loader = DataLoader(val_obj,batch_size=self.batch_size,**self.kwargs)

        if isinstance(test,pd.DataFrame):
            test_obj = DatasetObject(filepaths=test['fullpath'].to_numpy(),
                                      label=test['activity'].to_numpy(),
                                      transform=self.transform,
                                      readtype=self.readtype)

            if self.load_data: test_obj = test_obj.load_data()

            if self.test_size == 'full':
                batch_size = test.shape[0]
            else:
                batch_size = self.batch_size

            test_loader = DataLoader(test_obj,batch_size=batch_size,**self.kwargs)

        return train_loader,val_loader,test_loader

def DataLoading_CnnLstmS():
    """dataLoading_cnnLstmS"""
    transform = Transform_CnnLstmS()
    return DataLoading(transform,batch_size=64,test_size='full',shuffle=False,num_workers=0)

def DataLoading_CnnS():
    """dataLoading_CnnS"""
    transform = Transform_CnnS()
    return DataLoading(transform,batch_size=64,test_size='full',shuffle=False,num_workers=0)
