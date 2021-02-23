import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F


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

from data import prepare_double_source
from models import add_SimCLR,add_classifier
from models.baseline import Encoder as Baseline_Encoder
from models.cnn import create_vgg16
from losses import NT_Xent
from train import train


def pretrain(model, train_loader, criterion, optimizer, end, start = 1, device = None):
    # Check device setting
    if device:
        model = model.to(device)
        criterion = criterion.to(device)

    print('Start Training')
    record = {'train':[]}
    i = start
    #Loop
    while i <= end:
        print(f"Epoch {i}: ", end='')
        for b, (items) in enumerate(train_loader):

            if device:
                items = [i.to(device) for i in items]

            print(f">", end='')

            optimizer.zero_grad()

            items = model(*items)

            loss = criterion(*items)

            loss.backward()
            optimizer.step()

        # One epoch completed
        loss = loss.tolist()
        record['train'].append(loss)
        print(f' loss: {loss} ',end='')

        i += 1

    if device:
        items = [i.cpu() for i in items]
        del items
        model = model.cpu()

    return model, record


def create_encoder():
    outsize = 960
    encoder = Baseline_Encoder([32,64,96])
    return encoder, outsize


def main():
    encoder, outsize = create_encoder()
    model = add_SimCLR(enc,out_size)
    pretrain_loader, finetune_loader, validatn_loader, lb, class_weight = prepare_double_source(directory,
                                                                                                modality='single',
                                                                                                axis=1,
                                                                                                train_size=0.8,
                                                                                                joint='first',
                                                                                                p=None,
                                                                                                sampling='weight',
                                                                                                batch_size=64,
                                                                                                num_workers=0)
    criterion = NT_Xent(batch_size, temperature=0.1, world_size=1)
    optimizer = torch.optim.SGD(list(model.parameters()), lr=0.0005)
    model, record = contrastive_pretraining(model,train_loader,optimizer,criterion,end,start=1,model2=None,parallel=True)
    record_log(MAIN_NAME,NUM_EPOCHS,record)
    del criterion,optimizer,pretrain_loader
    # Finetuning
    model = add_classifier(enc,out_size,freeze=True)
    criterion = nn.CrossEntropyLoss(weight=class_weight).to(DEVICE)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0005)
    model, record = train(model=model,
                          train_loader=finetune_loader,
                          criterion=criterion,
                          optimizer=optimizer,
                          end=NUM_EPOCHS,
                          start = 1,
                          test_loader = validatn_loader,
                          device = DEVICE,
                          regularize = None)
    cmtx,cls = evaluation(model,validatn_loader,label_encoder=lb)
    record_log(MAIN_NAME,NUM_EPOCHS,record,cmtx=cmtx,cls=cls)
    save(MAIN_NAME,model,optimizer,NUM_EPOCHS)
    return

if __name__ == '__main__':
    main()
