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
    elif exp_no == 'exp_3':
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
        if exp_no in ['exp_1','exp_2']:
            user = filepath_x.split('\\')[-1].split("_")[1]
        elif exp_no == 'exp_3':
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
    if name == 'exp_1':
        pass
        # df = df.reset_index(drop=True)
        # df.user = df.user.apply(lambda x: x.split("\\")[0])
    elif name == 'exp_2':
        #  df = df.reset_index(drop=True)
        df.user = df.user.apply(lambda x: x.split('.')[0])
        df = df.iloc[:,2:]
    elif name == 'exp_3':
        #  df = df.reset_index(drop=True)
        df = df.iloc[:,2:]
        df['label'] = df['label'].fillna('noactivity')
    return df

def import_clean_data(name,fp):
    """
    import clean dataset based on the name. Available dataset are {"exp1",
    "exp2"}
    """
    if name in ['exp_1','exp_2','exp_3']:
        df = import_experimental_data(name,fp)
        df = clean_data(name,df)
    else:
        return
    return df
