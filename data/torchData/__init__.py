from data.torchData.transformation import *

def pipeline1(df):
    """To test the model under lab environemnt by leave-one-person-out validation"""
    df = df[df['nuc'] == 'NUC1']
    df = df[df['room'] == 1]
    train,val,test = leaveOnePersonOut_split(df,testsub='One',valsub='Four')
    test = resampling(test,'activity',oversampling=False)
    return train,val,test

def pipeline1b(df):
    """To test the model under lab environemnt by random-split validation"""
    df = df[df['nuc'] == 'NUC1']
    df = df[df['room'] == 1]
    train,test = random_split(df,train_size=0.8)
    train,val = random_split(train,train_size=0.875)
    test = resampling(test,'activity',oversampling=False)
    return train,val,test

def pipeline2(df):
    """To test the model under unseen environemnt"""
    df = df[df['nuc'] == 'NUC1']
    train = df[df['room'] == 1]
    test  = df[df['room'] == 2]

    train,val, _ = leaveOnePersonOut_split(train,testsub='One',valsub='Four')
    _ , _ , test = leaveOnePersonOut_split(test,testsub='One',valsub='Four')

    test = resampling(test,'activity',oversampling=False)
    return train,val,test


def pipeline3(df):
    """ To test if combine data provide better generalizability"""
    nuc = ['NUC1','NUC2']
    rooms = [1,2]

    df = df[df['nuc'].isin(nuc)]
    df = df[df['room'].isin(rooms)]

    train,val,test = leaveOnePersonOut_split(df,testsub='One',valsub='Four')
    test = resampling(test,'activity',oversampling=False)
    return train,val,test


def pipeline4(df,spc=5):
    """To test model performance under low number of sample"""
    df = df[df['nuc'] == 'NUC1']
    df = df[df['room'] == 1]
    num_class = df['activity'].nunique()

    train,val,test = leaveOnePersonOut_split(df,testsub='One',valsub='Four')

    train = resampling(train,'activity',oversampling=False)
    train,_ = random_split(train,train_size=spc*num_class)
    test = resampling(test,'activity',oversampling=False)
    return train,val,test

def generalPipeline(df,split='loov',nuc='NUC1',room=1,spc=None):

    if isinstance(nuc,list):
        df = df[df['nuc'].isin(nuc)]
    else:
        df = df[df['nuc'] == nuc]

    if isinstance(room,list):
        df = df[df['room'].isin(room)]
    else:
        df = df[df['room'] == room]

    num_class = df['activity'].nunique()

    if split == 'random':
        train,test = random_split(df,train_size=0.8)
        train,val = random_split(train,train_size=0.875)
    elif split == 'loov':
        train,val,test = leaveOnePersonOut_split(df,testsub='One',valsub='Four')

    if isinstance(spc,int):
        train = resampling(train,'activity',oversampling=False)
        train,_ = random_split(train,train_size=num_spc*spc)
    test = resampling(test,'activity',oversampling=False)
    return train,val,test
