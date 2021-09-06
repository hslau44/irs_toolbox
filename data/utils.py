import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def list_all_filepaths(directory):
    """
    Return all filepaths in the directory with targeted format

    Arugments:
    directory (str): the directory of the files

    Return
    filepaths (list): all filepaths of the files
    """
    filepaths = []
    for r, d, f in os.walk(directory):
        for item in f:
            filepaths.append(os.path.join(r, item))
    return filepaths

def filepath_dataframe(directory,splitchar='\\'):
    """
    pandas.Dataframe that contains all filepaths in the directory with targeted format,
    the folders in the directory are identified as class based on its levels

    Arguments:
    directory (str)
    format (str): format of the files to be returned, must include '.'

    Return:
    dataframe (pd.DataFrame): a dataframe with filespath as the first columns, number of columns is
    equal to the available level of the folders

    Example:
    for image with the fllowing filepath "directory/cat/black/blackcat.jpg"
    the dataframe would be shown as below:

    fullpath                         | class_1 | class_2
    ---------------------------------+---------+---------
    directory/cat/black/blackcat.jpg | cat     | black

    """

    filepaths = list_all_filepaths(directory)

    df = pd.DataFrame(data=filepaths,columns=['fullpath'])

    start = len(directory.split(splitchar))

    end = df['fullpath'].apply(lambda x: len(x.split(splitchar))).max()-1

    for i in range(start,end):
        df[f'class_{i-start+1}'] = df['fullpath'].apply(lambda x: x.split(splitchar)[i])

    return df


def breakpoints(ls):
    """find the index where element in ls(list) changes"""
    points = []
    for i in range(len(ls)-1):
        if ls[i+1] != ls[i]:
            points.append(i)
    return points

def visual_spectrogram(img,title='spectrogram',figsize=(15,50)):
    """
    visual spectrogram
    """
    if len(img.shape) == 3:

        img.squeeze()

    img = MinMaxScaler(feature_range=(0,1)).fit_transform(img)

    plt.figure(figsize = figsize)
    plt.imshow(img,cmap='jet')
    plt.title(title)
    plt.show()

    return
