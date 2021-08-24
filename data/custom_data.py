import numpy as np
import pandas as pd
from data.utils import list_all_filepaths

def filepath_dataframe(directory):

    filepaths = list_all_filepaths(directory)

    df = pd.DataFrame(data=filepaths,columns=['fullpath'])

    df['filename'] = df['fullpath'].apply(lambda x: x.split('\\')[-1])

    df['exp'] = df['filename'].apply(lambda x: int(x.split('_')[1]))

    df['person'] = df['filename'].apply(lambda x: x.split('_')[3])

    df['room'] = df['filename'].apply(lambda x: int(x.split('_')[5]))

    df['activity'] = df['filename'].apply(lambda x: x.split('_')[6])

    df['index'] = df['filename'].apply(lambda x: int(x.split('_')[8]))

    df['nuc'] = df['filename'].apply(lambda x: x.split('_')[9].split('.')[0])

    df['key'] = df['filename'].apply(lambda x: x[:-9])

    df.pop('filename')

    return df

def nucPaired_fpDataframe(dataframe):

    nuc1 = dataframe[dataframe['nuc'] == 'NUC1'].drop('nuc',axis=1)

    nuc2 = dataframe[dataframe['nuc'] == 'NUC2'][['key','fullpath']]

    return pd.merge(nuc1,nuc2,on='key')
