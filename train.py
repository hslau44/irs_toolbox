import os
import sys
from time import gmtime, strftime
import numpy as np
import pandas as pd
import seaborn as sns  # for heatmaps
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor, nn
from torch.nn import functional as F

# root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(root_folder)


device = torch.device("cuda:0")
torch.cuda.set_device(device)

np.random.seed(1024)
torch.manual_seed(1024)



def train(model, train_loader, criterion, optimizer, end, start = 1, test_loader = None, parallel = None, **kwargs):

    # Check device setting
    if parallel == True:
        print('GPU')
        model = model.cuda()
    else:
        print('CPU')

    print('Start Training')
    record = {'train':[],'validation':[]}
    i = start
    #Loop
    while i <= end:
        print(f"Epoch {i}: ", end='')
        for b, (X_train, y_train) in enumerate(train_loader):

            if parallel == True:
                X_train = X_train.cuda() #.to(device)

            print(f">", end='')
            optimizer.zero_grad()
            y_pred = model(X_train)

            if parallel == True:
                X_train = X_train.cpu()
                del X_train
                y_train = y_train.cuda()

            loss   = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            if parallel == True:
                y_pred = y_pred.cpu()
                y_train = y_train.cpu()
                del y_pred,y_train

        # One epoch completed
        loss = loss.tolist()
        record['train'].append(loss)
        print(f' loss: {loss} ',end='')
        if (test_loader != None) and i%10 ==0 :
            acc = short_evaluation(model,test_loader,parallel)
            record['validation'].append(acc)
            print(f' accuracy: {acc}')
        else:
            print('')
        i += 1

    model = model.cpu()
    return model, record



def short_evaluation(model,test_loader,parallel):
    # copy the model to cpu
    if parallel == True:
        model = model.cpu()
    with torch.no_grad():
        for X_test, y_test in test_loader:
            y_val = model(X_test)
            predicted = torch.max(y_val, 1)[1]
            acc = accuracy_score(y_test.view(-1), predicted.view(-1))
    # send model back to gpu
    if parallel == True:
        model = model.cuda()
    return acc

def cmtx_table(cmtx,label_encoder=None):
    if label_encoder != None:
        if type(label_encoder) == sklearn.preprocessing._label.LabelEncoder:
            cmtx = pd.DataFrame(cmtx,
                                index=[f"actual: {i}"for i in label_encoder.classes_.tolist()],
                                columns=[f"predict : {i}"for i in label_encoder.classes_.tolist()])
        else:
            cmtx = pd.DataFrame(cmtx,
                                index=[f"actual: {i}"for i in label_encoder.categories_[0].tolist()],
                                columns=[f"predict : {i}"for i in label_encoder.categories_[0].tolist()])
    else:
        cmtx = pd.DataFrame(cmtx)
    return cmtx

def evaluation(model,test_loader,label_encoder=None):
    model = model.cpu()
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test, y_test
            y_val = model(X_test)
            predicted = torch.max(y_val, 1)[1]
    cls = classification_report(y_test.view(-1), predicted.view(-1), output_dict=True)
    cls = pd.DataFrame(cls)
    print(cls)
    cmtx = confusion_matrix(y_test.view(-1), predicted.view(-1))
    cmtx = cmtx_table(cmtx,label_encoder)
    return cmtx,cls



def make_directory(name, epoch=None, filepath='./models/saved_models/'):
    time = strftime("%Y_%m_%d_%H_%M", gmtime())
    directory = filepath + name + '_checkpoint_' + str(epoch) + '__' + time
    return directory



def save_checkpoint(model,optimizer,epoch,directory):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, directory)
    print(f"save checkpoint in : {directory}")
    return

def load_checkpoint(model,optimizer,filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model,optimizer




# -----------------------------------------helper---------------------------------------











def main():
    pass


if __name__ == '__main__':
    main()
