
import numpy as np
import pandas as pd
import time
import os
import sys
from data.torchData.utils import filepath_dataframe

def assure_path_exists(directory,makedir=False):
    if not os.path.exists(directory):
        if makedir == True:
            os.makedirs(directory)
            return True
        else:
            raise FileNotFoundError(f'{directory} does not exist')
    return True

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
        if (idx+1)%(len(df)/10) == 0: print('>',end='')
    return

if __name__ == '__main__':
    directory = 'E:/external_data/opera_csi/Session_2/experiment_data/experiment_data/exp_7_amp_spec_only/npy_format'
    df = filepath_dataframe(directory)
    savepath = 'E:\\external_data\\opera_csi\\Session_2\\experiment_data\\experiment_data\\exp_7_amp_spec_only\\npy_format'
    load_npy(df,save_dir=savepath)
