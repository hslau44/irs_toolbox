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
from train import train,record_log,evaluation


# # random seed
# np.random.seed(1024)
# torch.manual_seed(1024)

# # gpu setting
# DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = DEVICE
# torch.cuda.set_device(DEVICE)

# ### data setting
# # DIRC = 'E:/external_data/Experiment4/Spectrogram_data_csv_files/CSI_data_pair'
# DIRC = './data/CSI_CSI'
# MODALITY='single'
# AXIS=1
# TRAIN_SIZE=0.8
# JOINT='first'
# PER=None
# SAMPLING='weight'

# ### train setting
# BATCH_SIZE= 64
# NUM_WORKERS = 4
# TEMPERATURE= 0.1
# REGULARIZE = True
# PRETRAIN_EPOCHS = 0
# FREEZE=False
# FINETUNE_EPOCHS = 300
# MAIN_NAME = 'Trainmode-normal_Network-vgg16_Data-exp4csicsi-1_s-l2reg' #'TEST'
# OUT_PATH = '.' # None #    


# m = MODALITY
# output = OUT_PATH

# if PRETRAIN_EPOCHS > 0:
#     FREEZE=True
    
    



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


def phase_lab_pretraining():

    # create dataloader for pretraining  
    print('X1_train: ',X1_train.shape,'\tX2_train: ',X2_train.shape)
    pretrain_loader = create_dataloader(X1_train,X2_train,batch_size=pre_batch_size,num_workers=num_workers)
    encoder, outsize = create_encoder(network,'nuc2')
    encoder2, outsize2 = None,None

    # the pairing data use the same encoder if 'joint', else we use two encoders
    if joint == 'joint':
        simclr = add_SimCLR(encoder, outsize)
    else:
        encoder2, outsize2 = create_encoder(network,pairing)
        simclr = add_SimCLR_multi(enc1=encoder,enc2=encoder2,out_size1=outsize,out_size2=outsize2)

    # (optional) pretraining with simclr  
    phase = 'pretrain'
    if trainmode == 'simclr':
        criterion = NT_Xent(pre_batch_size, temperature, world_size=1)
        optimizer = torch.optim.SGD(list(simclr.parameters()), lr=0.0005)
        simclr, record = pretrain(model=simclr,
                                  train_loader=pretrain_loader,
                                  criterion=criterion,
                                  optimizer=optimizer,
                                  end=pretrain_epochs,
                                  device=device)
        record_log(record_outpath,exp_name,phase,record=record)

    # save
    encoder_fp = save_model(model_outpath,exp_name,phase,simclr.encoder)
    del simclr,encoder,outsize, encoder2,outsize2
    torch.cuda.empty_cache()
    return encoder_fp
    

def phase_lab_finetuning():
    # sampling condition
    samplings = [1,5,10,'weight','undersampling','oversampling',]
    inital = {'lab':True} 
    
    for sampling in samplings:

        # decided whether the dataset should be joined together
        X_train, X_test, y_train, y_test = select_train_test_dataset(X1_train, X1_test, X2_train, X2_test, y_train_, y_test_, joint)
        X_train, X_test, y_train, y_test, lb = filtering_activities_and_label_encoding(X_train, X_test, y_train, y_test, activities)
        lab_finetune_loader, lab_validatn_loader, class_weight = combine1(X_train, X_test, y_train, y_test, 
                                                                          sampling, lb, batch_size, num_workers, 
                                                                          y_sampling=y_sampling)
        print("class: ",lb.classes_)
        print("class_size: ",1-class_weight)

        # load the encoder
        encoder, outsize = load_encoder(network, encoder_fp)
        model = add_classifier(encoder,in_size=outsize,out_size=len(lb.classes_),freeze=freeze)

        # evaluate the untrained network  
        phase = 'lab-initial'
        if inital['lab']:
            cmtx,cls = evaluation(model,lab_finetune_loader,label_encoder=lb)
            record_log(record_outpath,exp_name,phase,cmtx=cmtx,cls=cls)
            inital['lab'] = False

        # finetuning 
        phase = 'lab-finetune'+'-'+str(sampling)
        criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)
        optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0005)
        model, record = train(model=model,
                              train_loader= lab_finetune_loader,
                              criterion=criterion,
                              optimizer=optimizer,
                              end= lab_finetune_epochs,
                              test_loader = lab_validatn_loader,
                              device = device,
                              regularize = regularize)

        # test and record
        cmtx,cls = evaluation(model,lab_validatn_loader,label_encoder=lb)
        record_log(record_outpath,exp_name,phase,record=record,cmtx=cmtx,cls=cls,acc_rec=True)

        # every loop the model (including encoder),criterion,optimizer are to be deleted 
        if sampling != 'weight':
            del encoder,model,criterion,optimizer,record,cmtx,cls
            torch.cuda.empty_cache()
        elif sampling == 'weight':
            # model_fp = save_model(model_outpath,exp_name,phase,model)
            del encoder,model,criterion,optimizer,record,cmtx,cls
            torch.cuda.empty_cache()
    
        return 
    
    

def main():
    return

if __name__ == '__main__':
    main()
