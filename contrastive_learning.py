import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F

from data import prepare_double_source
from models import add_SimCLR,add_classifier
from models.baseline import Encoder as Baseline_Encoder
from models.cnn import create_vgg16
from losses import NT_Xent
from train import train,record_log,evaluation,save


# random seed
np.random.seed(1024)
torch.manual_seed(1024)

# gpu setting
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = DEVICE
torch.cuda.set_device(DEVICE)

### data setting
DIRC = 'E:/external_data/Experiment4/Spectrogram_data_csv_files/CSI_data_pair'
# DIRC = '.'
MODALITY='dummy'
AXIS=1
TRAIN_SIZE=0.8
JOINT='joint'
PER=None
SAMPLING='weight'

### train setting
BATCH_SIZE=64
NUM_WORKERS = 0
TEMPERATURE=0.1
REGULARIZE = None
PRETRAIN_EPOCHS = 1
FINETUNE_EPOCHS = 10
MAIN_NAME = 'TEST'#'Trainmode_simclr_Network_shallowv2_Data_exp4nuc1'
OUT_PATH = None #'.'
output = OUT_PATH



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

            items = model(items)

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
    model = add_SimCLR(encoder,outsize)
    pretrain_loader, finetune_loader, validatn_loader, lb, class_weight = prepare_double_source(directory=DIRC,
                                                                                                modality=MODALITY,
                                                                                                axis=AXIS,
                                                                                                train_size=TRAIN_SIZE,
                                                                                                joint=JOINT,
                                                                                                p=PER,
                                                                                                sampling=SAMPLING,
                                                                                                batch_size=BATCH_SIZE,
                                                                                                num_workers=NUM_WORKERS)
    criterion = NT_Xent(BATCH_SIZE, temperature=TEMPERATURE, world_size=1)
    optimizer = torch.optim.SGD(list(model.parameters()), lr=0.0005)
    model, record = pretrain(model=model,
                             train_loader=pretrain_loader,
                             criterion=criterion,
                             optimizer=optimizer,
                             end=PRETRAIN_EPOCHS,
                             start=1,
                             device=DEVICE)
    if output:
        record_log(MAIN_NAME,PRETRAIN_EPOCHS,record,filepath=OUT_PATH+'/record/')
    del criterion,optimizer,pretrain_loader
    # Finetuning
    model = add_classifier(model.encoder,outsize,freeze=True)
    criterion = nn.CrossEntropyLoss(weight=class_weight).to(DEVICE)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0005)
    model, record = train(model=model,
                          train_loader=finetune_loader,
                          criterion=criterion,
                          optimizer=optimizer,
                          end=FINETUNE_EPOCHS,
                          start = 1,
                          test_loader = validatn_loader,
                          device = DEVICE,
                          regularize = REGULARIZE)
    cmtx,cls = evaluation(model,validatn_loader,label_encoder=lb)
    if output:
        record_log(MAIN_NAME,FINETUNE_EPOCHS,record,cmtx=cmtx,cls=cls)
        save(MAIN_NAME,model,optimizer,FINETUNE_EPOCHS)
    return

if __name__ == '__main__':
    main()
