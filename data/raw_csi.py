import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.utils import shuffle


import numpy as np
import pandas as pd
import glob

def generate_filepaths(fp):
    return fp+"/input*.csv", fp+"/annotation*.csv"

def list_to_string(ls):
    string = ""
    for i in ls:
        string += i
    return string

def col_generate(exp_no):
    if exp_no == 'exp_3':
        col = []
        for i in range(1,91):
            col.append(str(f'amp_{i}'))
        for i in range(1,91):
            col.append(str(f'theta_{i}'))
    elif exp_no == 'exp_2':
        col = ['time','unknown_1']
        for i in range(1,91):
            col.append(str(f'amp_{i}'))
    elif exp_no == 'exp_1':
        col = ['time','unknown_1']
        for i in range(1,91):
            col.append(str(f'amp_{i}'))
    else:  raise Exception("Error occured in function: col_generate")
    return col


def import_single_file(filepath_x,filepath_y,sample_id,exp_no):
    df = pd.read_csv(filepath_x,names=col_generate(exp_no),header=0)
    df['user'] = sample_id
    df['label'] = pd.read_csv(filepath_y,names=['label'],header=0)
    return df

def import_experimental_data(exp_no,fp):
    searchpaths_x,searchpaths_y = generate_filepaths(fp)
    dataframes = []
    filepaths_x = sorted(glob.glob(searchpaths_x))
    filepaths_y = sorted(glob.glob(searchpaths_y))
    assert len(filepaths_x) == len(filepaths_y)
    print(f"Found {len(filepaths_x)} files.")
    for filepath_x,filepath_y in zip(filepaths_x,filepaths_y):
        if exp_no in ['exp_3','exp_2']:
            user = filepath_x.split('\\')[-1].split("_")[1]
        elif exp_no == 'exp_1':
            user = filepath_x.split('.')[0].split("_")[7:]
            user = list_to_string(user)
        else:
            raise Exception('Error with exp_no')
        dataframes.append(import_single_file(filepath_x,filepath_y,user,exp_no))
        print(filepath_x.split('\\')[-1],filepath_y.split('\\')[-1],user)
    df = pd.concat(dataframes,axis=0)
    return df


def clean_data(name,df):
    """
    Clean specific dataset based on the name. Available dataset are {"exp1",
    "exp2"}
    """
    if name == 'exp_3':
        pass
        # df = df.reset_index(drop=True)
        # df.user = df.user.apply(lambda x: x.split("\\")[0])
    elif name == 'exp_2':
        #  df = df.reset_index(drop=True)
        df.user = df.user.apply(lambda x: x.split('.')[0])
        df = df.iloc[:,2:]
    elif name == 'exp_1':
        #  df = df.reset_index(drop=True)
        df = df.iloc[:,2:]
        df['label'] = df['label'].fillna('noactivity')
    return df

def import_clean_data(name,fp):
    """
    import clean dataset based on the name.

    Arguments:
    name (str): Available dataset are {"exp1","exp2"}
    fp (str): filepath, C:/directory/folder
    """
    if name in ['exp_1','exp_2','exp_3']:
        df = import_experimental_data(name,fp)
        df = clean_data(name,df)
    else:
        return
    return df

# ------------------------------------helper------------------------------------

def seperate_dataframes(df):
    """To be depreciated """
    features_ls,labels_ls = [],[]
    for user in df['user'].unique():
        dataframe = df[df['user']==user]
        features = dataframe[[f'amp_{i}' for i in range(1,91)]].to_numpy()
        features = MinMaxScaler().fit_transform(features)
        features_ls.append(features)
        label = dataframe[['label']].to_numpy()
        labels_ls.append(label)
    return features_ls,labels_ls


def create_datasetobj(X,y):
    """To be depreciated """
    datasetobj = DatasetObject()
    datasetobj.import_data(X, y)
    return datasetobj


def transform_datasetobj(datasetobj, window_size=1000, slide_size=200, txr=1, oversampling=True):
    """To be depreciated """
    # augmentation
    datasetobj.data_transform(lambda x,y,z : process_data.slide_augmentation(x, y, z,window_size=window_size,slide_size=slide_size,skip_labels=['noactivity']),axis=0)
    # txr and channels
    txr_dict = {1:(1,90),3:(3,30)}
    pair,channels =  txr_dict[txr]
    datasetobj.data_transform(lambda arr: arr.reshape(-1,window_size,pair,channels).transpose(0,2,3,1),axis=1, col=0)
    # resample
    datasetobj.data_transform(lambda x,y,z : process_data.resampling(x, y, z, oversampling = oversampling), axis=0)
    # label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(datasetobj()[1])
    datasetobj.data_transform(lambda arr: label_encoder.transform(arr).reshape(arr.shape),axis=1, col=1)
    return datasetobj, label_encoder

def prepare_exp_1(nums=[9], window_size=900,slide_size=200,txr=1,oversampling=True,train_batch_sizes=128):
    """To be depreciated """
    fp = "E:/external_data/Experiment3/csv_files/exp_1"
    df = import_clean_data('exp_1',fp)
    X_ls, y_ls = seperate_dataframes(df)
    del df
    datasetobj = create_datasetobj(X_ls,y_ls)
    datasetobj, label_encoder = transform_datasetobj(datasetobj,window_size,slide_size,txr,oversampling)
    datasetobj.shape()
    del X_ls, y_ls
    (X_train, y_train,_),(X_test, y_test,_) = datasetobj(nums,return_train_sets=True)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test,train_batch_sizes)
    return train_loader, test_loader
