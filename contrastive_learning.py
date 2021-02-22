import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F

# from torch import Tensor
# from torch.utils.data import DataLoader, TensorDataset
# from torch.nn import functional as F
# from torchsummary import summary
# import torchvision
#
#
#
# from models import ED_module, Classifier
# from losses import SupConLoss, NT_Xent
# from train import train as finetuning
# from train import evaluation,make_directory,save_checkpoint
# from models.self_supervised import Projection_head
# from models.baseline import Encoder_F as Encoder
# from models.cnn import create_vgg16


# random seed
np.random.seed(1024)
torch.manual_seed(1024)

# gpu setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
parallel = True
num_workers = 0

# data setting
loc_dirc = 'E:/external_data/Experiment4/Spectrogram_data_csv_files/CSI_data'
remote_dirc = './data/csi_pwr'
PATH = '.' # './'
EXP_NAME = 'Encoder_vgg16_mode_normal_on_exp4_s_resample_regularize'


DIRC = loc_dirc
mode = 1
p = None
resample = True
pre_train_epochs = 1
fine_tune_epochs = 1
bsz = 64
parallel = True
csi_out_size = (2,3)
pwr_out_size = (3,1)

from data import prepare_single_source,prepare_double_source
from data.spectrogram import import_data, import_pair_data, import_CsiPwr_data
from data.transformation import label_encode,resampling
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def create_criterion(batch_size):
    # External libraries required
    criterion = NT_Xent(batch_size, temperature=0.1, world_size=1)
    return criterion

def create_optimizer(mode,model):
    optimizer = torch.optim.SGD(list(model.parameters()), lr=0.0005)
    # double # optimizer = torch.optim.SGD(list(model1.parameters())+list(model2.parameters()), lr=0.0005)
    return optimizer



def contrastive_pretraining(model,train_loader,optimizer,criterion,end,start=1,model2=None,parallel=True):
    # Check device setting
    if parallel == True:
        print('GPU')
        model = model.to(device)

        if model2:
            model2 = model2.to(device)

    else:
        print('CPU')
        model = model.cpu()

    print('Start Training')
    record = {'train':[]}
    i = start

    #Loop
    while i <= end:

        print(f"Epoch {i}: ", end='')

        for b, (X1, X2) in enumerate(train_loader):

            print(f">", end='')

            optimizer.zero_grad()

            if parallel == True:
                X1 = X1.to(device)

            X1 = model(X1)

            if parallel == True:
                X2 = X2.to(device)

            if model2:
                X2 = model2(X2)
            else:
                X2 = model(X2)

            loss = criterion(X1,X2)

            loss.backward()

            optimizer.step()

        # One epoch completed
        l = loss.tolist()
        record['train'].append(l)
        print(f' loss: {l} ')
        i += 1

        del X1,X2

    model = model.cpu()

    return model,record

def record_log(mode,epochs,record,cmtx=None,cls=None):
    if mode == 'pretraining':
        path = make_directory(EXP_NAME+'_pretrained',epoch=epochs,filepath=PATH+'/record/')
        pd.DataFrame(record['train'],columns=['train_loss']).to_csv(path+'_loss.csv')
    elif mode == 'finetuning':
        path = make_directory(EXP_NAME+'_finetuned',epoch=epochs,filepath=PATH+'/record/')
        pd.DataFrame(record['train'],columns=['train_loss']).to_csv(path+'_loss.csv')
        pd.DataFrame(record['validation'],columns=['validation_accuracy']).to_csv(path+'_accuracy.csv')
        cls.to_csv(path+'_report.csv')
        cmtx.to_csv(path+'_cmtx.csv')
    return

def save(mode,model,optimizer,epochs):
    if mode == 'pretrain':
        path = make_directory(EXP_NAME+'_pretrained',epoch=epochs,filepath=PATH+'/models/saved_models/')
        save_checkpoint(model, optimizer, epochs, path)
    elif mode == 'finetuning':
        path = make_directory(EXP_NAME+'_finetuned',epoch=epochs,filepath=PATH+'/models/saved_models/')
        save_checkpoint(model, optimizer, epochs, path)
    return


def main():
    X,y,lb = prepare_data(dirc)
    return

if __name__ == '__main__':
    main()
