import os
from os import listdir
from os.path import isfile, join


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

def filepath_dataframe(directory,split='\\'):
    """
    pandas.Dataframe that contains all filepaths in the directory with targeted format,
    the folders in the directory are identified as class based on its levels

    Arguments:
    directory (str): the directory of the files
    split (str)

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

    start = len(directory.split(split))

    end = df['fullpath'].apply(lambda x: len(x.split(split))).max()-1

    for i in range(start,end):
        df[f'class_{i-start+1}'] = df['fullpath'].apply(lambda x: x.split(split)[i])

    return df




def breakpoints(ls):
    """find the index where element in ls(list) changes"""
    points = []
    for i in range(len(ls)-1):
        if ls[i+1] != ls[i]:
            points.append(i)
    return points
