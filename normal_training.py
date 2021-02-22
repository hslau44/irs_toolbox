import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
import torchvision

from data.spectrogram import import_data
from data.process_data import label_encode, create_dataloaders
from models import Lambda, ED_module
from losses import SupConLoss
from train import train, make_directory,save_checkpoint
from train import evaluation

# Random seed
np.random.seed(1024)
torch.manual_seed(1024)

# GPU setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
parallel = True
num_workers = 0

# data setting
columns = [f"col_{i+1}" for i in range(501)] # 65*501
window_size=501
slide_size=501


# Data main folder
dirc = "E:/external_data/Experiment4/Spectrogram_data_csv_files/CSI_data"

# Saving path
PATH = 'C://Users/Creator/Script/Python/Project/irs_toolbox' # '.'

# Training setting
bsz = 128
pre_train_epochs = 500
fine_tune_epochs = 300

exp_name = 'Encoder_64-128-256-512-128-7_mode_clr_on_exp4csi'



def prepare_dataloader():
    X,y  = import_data(dirc,columns=columns,window_size=window_size,slide_size=slide_size)
    X = X.reshape(*X.shape,1).transpose(0,3,1,2)
    y,lb = label_encode(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, train_batch_sizes=bsz, test_batch_sizes=200, num_workers=num_workers)
    return train_loader, test_loader,lb


class Encoder(nn.Module):
    """
    Encoder for spectrogram (1,65,65), 3 layer
    """
    def __init__(self,num_filters):
        super(Encoder, self).__init__()
        l1,l2,l3,l4 = num_filters
        ### 1st ###
        self.conv1 = nn.Conv2d(1,l1,kernel_size=(5,5),stride=(2,2))
        self.norm1 = nn.BatchNorm2d(l1)
        self.actv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((1,2))
        ### 2nd ###
        self.conv2 = nn.Conv2d(l1,l2,kernel_size=(4,4),stride=(2,2))
        self.norm2 = nn.BatchNorm2d(l2)
        self.actv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((1,2))
        ### 3rd ###
        self.conv3 = nn.Conv2d(l2,l3,kernel_size=(3,3),stride=(2,2))
        self.norm3 = nn.BatchNorm2d(l3)
        self.actv3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d((1,2)) # nn.AdaptiveAvgPool2d((2,2))
        ### 4th ###
        self.conv4 = nn.Conv2d(l3,l4,kernel_size=(2,2),stride=(2,2))
        self.norm4 = Lambda(lambda x:x)
        self.actv4 = nn.Tanh()
        self.pool4 = nn.AdaptiveAvgPool2d((1,2))

    def forward(self,X):
        X = self.pool1(self.actv1(self.norm1(self.conv1(X))))
        X = self.pool2(self.actv2(self.norm2(self.conv2(X))))
        X = self.pool3(self.actv3(self.norm3(self.conv3(X))))
        X = self.pool4(self.actv4(self.norm4(self.conv4(X))))
        X = torch.flatten(X, 1)

        return X


class Classifier(nn.Module):
    """
    Linear Classifier (Double Layer)

    Hint: user nn.Linear as Classifier (Single Layer)
    """
    def __init__(self,input_shape,hidden_shape,output_shape,clr=False):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(input_shape,hidden_shape)
        self.linear2 = nn.Linear(hidden_shape,output_shape)
        if clr == True:
            self.norm = Lambda(lambda x: F.normalize(x,dim=1))
        else:
            self.norm = Lambda(lambda x:x)

    def forward(self,X):
        X = F.leaky_relu(self.linear1(X),inplace=True)
        X = F.dropout(X,0.2)
        X = self.linear2(X)
        X = self.norm(X)
        return X

def create_model(enc=None):
    enc = Encoder([64,128,256,512])
    clf = Classifier(1024,128,7,True)
    model = ED_module(encoder=enc,decoder=clf)
    return model


def record_log(mode,epochs,record,cmtx=None,cls=None):
    if mode == 'pretrain':
        path = make_directory(exp_name+'_pretrain',epoch=epochs,filepath=PATH+'/record/')
        pd.DataFrame(record['train'],columns=['train_loss']).to_csv(path+'_loss.csv')
    elif mode == 'finetuning':
        path = make_directory(exp_name+'_finetuning',epoch=epochs,filepath=PATH+'/record/')
        pd.DataFrame(record['train'],columns=['train_loss']).to_csv(path+'_loss.csv')
        pd.DataFrame(record['validation'],columns=['validation_accuracy']).to_csv(path+'_accuracy.csv')
        cls.to_csv(path+'_report.csv')
        cmtx.to_csv(path+'_cmtx.csv')
    return

def save(mode,model,optimizer,epochs):
    if mode == 'pretrain':
        path = make_directory(exp_name+'_pretrain',epoch=epochs,filepath=PATH+'/models/saved_models/')
        save_checkpoint(model, optimizer, epochs, path)
    elif mode == 'finetuning':
        path = make_directory(exp_name+'_finetuning',epoch=epochs,filepath=PATH+'/models/saved_models/')
        save_checkpoint(model, optimizer, epochs, path)
    return

def main():
    model = create_model()
    criterion = SupConLoss(temperature=0.1,stack=True) # Contrastive loss
    # criterion = nn.CrossEntropyLoss() # Cross entropy
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0005)
    train_loader, test_loader, lb = prepare_dataloader()
    # torch.cuda.empty_cache()
    model, record = train(model,train_loader,criterion,optimizer,fine_tune_epochs,1,test_loader,parallel)
    cmtx,cls = evaluation(model,test_loader,label_encoder=lb)
    record_log('finetuning',fine_tune_epochs,record,cmtx,cls)
    save('finetuning',model,optimizer,fine_tune_epochs)
    return

if __name__ == '__main__':
    main()
