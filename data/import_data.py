import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.utils import shuffle
import utils

def generate_filepaths(fp):
    return fp+"/input*.csv", fp+"/annotation*.csv"

def col_generate(exp_no):
    if exp_no == 'exp_1':
        col = []
        for i in range(1,91):
            col.append(str(f'amp_{i}'))
        for i in range(1,91):
            col.append(str(f'theta_{i}'))
    elif exp_no == 'exp_2':
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

def import_experimental_data(fp):
    searchpaths_x,searchpaths_y = generate_filepaths(fp)
    exp_no = fp.split("/")[-1]
    dataframes = []
    filepaths_x = sorted(glob.glob(searchpaths_x))
    filepaths_y = sorted(glob.glob(searchpaths_y))
    assert len(filepaths_x) == len(filepaths_y)
    print(f"Found {len(filepaths_x)} files.")
    for filepath_x,filepath_y in zip(filepaths_x,filepaths_y):
        user = filepath_x.split('\\')[-1].split("_")[1]
        dataframes.append(import_single_file(filepath_x,filepath_y,user,exp_no))
        print(filepath_x.split('\\')[-1],filepath_y.split('\\')[-1],user)
    df = pd.concat(dataframes,axis=0)
    return df

def clean_data(name,df):
    """
    Clean specific dataset based on the name. Available dataset are {"exp1",
    "exp2"}
    """
    if name == 'exp1':
        df = df.reset_index(drop=True)
        # df.user = df.user.apply(lambda x: x.split("\\")[0])
    if name == 'exp2':
        df = df.reset_index(drop=True)
        df = df.user.apply(lambda x: x.split('.')[0])
        df = df.iloc[:,2:]
    return df

def import_clean_data(name,fp):
    """
    import clean dataset based on the name. Available dataset are {"exp1",
    "exp2"}
    """
    if name in ['exp1','exp2']:
        df = import_experimental_data(fp)
        df = clean_data(name,df)
    else:
        return
    return df
