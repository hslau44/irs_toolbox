import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F

from data import prepare_double_source
from models import add_SimCLR,add_classifier,add_SimCLR_multi
from models.baseline import Encoder as Baseline_Encoder
from models.cnn import create_vgg16
from losses import NT_Xent
from train import train,record_log,evaluation,save


# random seed
np.random.seed(1024)
torch.manual_seed(1024)

# gpu setting
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = DEVICE
torch.cuda.set_device(DEVICE)

### data setting
# DIRC = 'E:/external_data/Experiment4/Spectrogram_data_csv_files/CSI_data_pair'
DIRC = './data/CSI_CSI'
MODALITY='single'
AXIS=1
TRAIN_SIZE=0.8
JOINT='first'
PER=None
SAMPLING='weight'

### train setting
BATCH_SIZE= 64
NUM_WORKERS = 4
TEMPERATURE= 0.1
REGULARIZE = True
PRETRAIN_EPOCHS = 0
FREEZE=False
FINETUNE_EPOCHS = 300
MAIN_NAME = 'Trainmode-normal_Network-vgg16_Data-exp4csicsi-1_s-l2reg' #'TEST'
OUT_PATH = '.' # None #    


m = MODALITY
output = OUT_PATH

if PRETRAIN_EPOCHS > 0:
    FREEZE=True
    
    
    
print('----------------------EXP ',19,'----------------------')


def pretrain(model, train_loader, criterion, optimizer, end, start = 1, device = None):
    # Check device setting
    if device:
        model = model.to(device)
        criterion = criterion.to(device)

    print('Start Training')
    record = {'loss':[]}
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
        record['loss'].append(loss)
        print(f' loss: {loss} ')

        i += 1

    if device:
        items = [i.cpu() for i in items]
        del items
        model = model.cpu()

    return model, record


def create_encoder():
#     outsize = 960
#     encoder = Baseline_Encoder([32,64,96])
#     outsize = 1920
#     encoder = Baseline_Encoder([64,128,192])
    encoder = create_vgg16((2,2))
    outsize = 512*2*2
    return encoder, outsize

def create_encoders():
    outsize = 512*2*2 
    encoder = create_vgg16((2,2))  
    outsize2 = 512*3*1 
    encoder2 = create_vgg16((3,1))
    return encoder,encoder2,outsize,outsize2


def main():
    
    if m == 'single':
        encoder, outsize = create_encoder()
        model = add_SimCLR(encoder,outsize)
    elif m == 'double':
        encoder,encoder2,outsize,outsize2 = create_encoders()
        model = add_SimCLR_multi(encoder,encoder2,outsize,outsize2)
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
    pretrain_output = PRETRAIN_EPOCHS
    if pretrain_output > 0:
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
    model = add_classifier(model.encoder,outsize,freeze=FREEZE)
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
