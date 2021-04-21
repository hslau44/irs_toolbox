import sys

import numpy as np
import pandas as pd
import seaborn as sns  # for heatmaps
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
import torchvision

from data.spectrogram import import_data, import_pair_data
from data import *
from models.baseline import *
from models.cnn import *
from models.self_supervised import *
from models.utils import *
from models import *
from losses import NT_Xent
from train import record_log,evaluation,pretrain,train,load_checkpoint,save_model



PRETRAIN_SKIP = ['waving'] # list of name of activities to be skiped in pretraining
FINETUNE_SKIP = ['waving'] # list of name of activities to be skiped in finetuning/training

EXTRA = None   # add extra string at the end of file for notation
VAL = 'random'   # validation method; option: 'id', 'random'

PRE_BATCH_SIZE = 64          # batch size for simclr
BATCH_SIZE = 64
NUM_WORKERS = 0
PRETRAIN_EPOCHS = 750
FINETUNE_EPOCHS = 200
VAL_ID = 1                   # participant; option: 1,2,3,4,5
SAMPLING = 'weight'
REGULARIZE = False
JOINT = 'first'

fp2 = './data/experiment_data/exp_2/spectrogram'
fp3 = './data/experiment_data/exp_3/spectrogram'
fp4 = 'E://external_data/experiment_data/exp_4/spectrogram_multi'

DATAPATH = 'E://external_data/opera_csi/experiment_data/exp_4/spectrogram_multi'
OUT_PATH = 'E://external_data/opera_csi/laboratory/experiment_e'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(DEVICE)
np.random.seed(1024)
torch.manual_seed(1024)


def main():

    trainmode,network,pairing,temperature = sys.argv[1:]

    exp_name = f'Trainmode-{trainmode}-{t}_Network-{network}_Data-exp4csi{pairing}{('-j' if JOINT=='joint')}'

    model_outpath = record_outpath = OUT_PATH

    data = import_pair_data(filepath=DATAPATH,modal=['csi',pairing],return_id=True)

    X1,X2,y,id_list = initial_filtering_activities(data,activities=PRETRAIN_SKIP)

    X1_train, X1_test, X2_train, X2_test, y_train, y_test = split_datasets(X1,X2,y,split=0.8,stratify=y)

    ##### Pretraining  #####
    pretrain_loader = create_dataloader(X1_train,X2_train,
                                        batch_size=PRE_BATCH_SIZE,num_workers=NUM_WORKERS)
    # create enocder
    encoder, outsize = create_encoder(network,'nuc2')

    encoder2, outsize2 = None,None

    if JOINT == 'joint':
        pretrain_model = add_SimCLR(encoder, outsize)
    else:
        encoder2, outsize2 = create_encoder(network,pairing)
        pretrain_model = add_SimCLR_multi(enc1=encoder,enc2=encoder2,out_size1=outsize,out_size2=outsize2)

    # pretraining with simclr
    phase = 'pretrain'

    if trainmode == 'simclr':

        criterion = NT_Xent(temperature=temperature, batch_size=PRE_BATCH_SIZE)

        optimizer = torch.optim.SGD(list(pretrain_model.parameters()), lr=0.0005)

        pretrain_model, record = pretrain(model=pretrain_model,
                                          train_loader=pretrain_loader,
                                          criterion=criterion,
                                          optimizer=optimizer,
                                          end=PRETRAIN_EPOCHS,
                                          device=DEVICE)

        record_log(record_outpath,exp_name,phase,record=record)
    # save

    encoder_fp = save_model(model_outpath,exp_name,phase,simclr.encoder)

    del simclr,encoder,outsize, encoder2,outsize2,criterion,optimizer

    torch.cuda.empty_cache()

    ##### Finetuning #####
    X_train, X_test, y_train, y_test = select_train_test_dataset(X1_train, X1_test, X2_train, X2_test, y_train, y_test,
                                                                 joint=JOINT)

    X_train, X_test, y_train, y_test, label_encoder = filtering_activities_and_label_encoding(X_train, X_test, y_train, y_test,
                                                                                              activities=FINETUNE_SKIP)

    X_train, X_test, y_train, y_test = apply_sampling(X_train, X_test, y_train, y_test, lb=label_encoder,
                                                      sampling=SAMPLING,y_sampling='weight')

    finetune_loader,validatn_loader = prepare_dataloaders(X_train, X_test,y_train, y_test,
                                                          sampling=SAMPLING, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    class_weight = return_class_weight(y_train)
    # load encoder
    encoder, outsize = load_encoder(network, encoder_fp)

    if trainmode == 'simclr':
        model = add_classifier(encoder,in_size=outsize,out_size=len(label_encoder.classes_),freeze=True)
    else:
        model = add_classifier(encoder,in_size=outsize,out_size=len(label_encoder.classes_),freeze=False)

    phase = 'lab-finetune'+'-'+str(SAMPLING)

    criterion = nn.CrossEntropyLoss(weight=class_weight).to(DEVICE)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0005)

    model, record = train(model=model,
                          train_loader= finetune_loader,
                          test_loader = validatn_loader,
                          criterion=criterion,
                          optimizer=optimizer,
                          end= FINETUNE_EPOCHS,
                          regularize = REGULARIZE,
                          device = DEVICE)

    cmtx,report = evaluation(model,validatn_loader,label_encoder)

    record_log(record_outpath,exp_name,phase,record=record,cmtx=cmtx,cls=report,acc_rec=True)

    model_fp = save_model(model_outpath,exp_name,phase,model)

    del encoder,model,criterion,optimizer# ,record,cmtx,cls

    torch.cuda.empty_cache()

    return




if __name__ == '__main__':
    main()
