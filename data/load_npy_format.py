import numpy as np
import pandas as pd
import time
import os
import sys
from data.utils import filepath_dataframe

def assure_path_exists(directory,makedir=False):
    if not os.path.exists(directory):
        if makedir == True:
            os.makedirs(directory)
            return True
        else:
            raise FileNotFoundError()
    return True

def helper(df)
    return [col for col in df.columns if 'class_' in col]

def generate_new_folderpath(row,new_root,splitchar,columns):
    """To be tested"""
    new_folderpath = new_root
    available_classes = [col for col in columns if 'class_' in col]
    available_classes.sort()
    for col in available_class:
        new_folderpath = new_folderpath + splitchar + row[col]
    return new_folderpath

def load_npy(df,save_dir='.',splitchar='\\',verbose=True):
    assure_path_exists(save_dir)
    df['filename'] = df['fullpath'].apply(lambda x: x.split(splitchar)[-1].split('.')[0])
    df['new_folderpath'] = df.apply(lambda row: save_dir + splitchar +
                                                row['class_1'] + splitchar +
                                                row['class_2'],
                                                axis=1)
    df['new_filepath'] = df.apply(lambda row: row['new_folderpath'] + splitchar +
                                              row['filename'],axis=1)

    for path in df['new_folderpath'].unique().tolist():
        assure_path_exists(path,makedir=True)

    if verbose: print('Start')
    for idx in range(len(df)):
        arr = pd.read_csv(df.loc[idx,'fullpath'],header=None).to_numpy()
        new_filepath = df.loc[idx,'new_filepath']
        np.save(new_filepath,arr)
        if verbose:
            if (idx+1)%(len(df)//10) == 0:
                print('>', end='')
    return


if __name__ == '__main__':
    directory = '/root/irs_toolbox/data/exp_7_amp_spec_only/spectrogram_multi'
    splitchar='/'

    df = filepath_dataframe(directory,splitchar)
    savepath = '/root/irs_toolbox/data/exp_7_amp_spec_only/npy_format'
    load_npy(df,save_dir=savepath,splitchar=splitchar)
