#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sn  # for heatmaps
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # import data

# In[2]:


# from scipy.io import loadmat

# link = r"D:\external_data\Experiment4"
# filename = r"\Dataset_PWR_WiFi.mat"
# directory = link + filename

# mat = loadmat(directory)


# In[3]:


from data.import_data import import_experimental_data
# from data.MyDataset import DatasetObject

# import experimental dataset 1
folderpath1 = "D:/external_data/Experiment3/csv_files/exp_1"  # CHANGE THIS IF THE PATH CHANGED
df_exp1 = import_experimental_data(folderpath1) # import_clean_data('exp1',
df_exp1.head()


# In[4]:


# # import experimental dataset 2
# folderpath2 = "D:/external_data/Experiment3/csv_files/exp_2"  # CHANGE THIS IF THE PATH CHANGED
# df_exp2 = import_clean_data('exp2',folderpath2)


# # process data

# In[5]:


def seperate_dataframes(df):
    features_ls,labels_ls = [],[]
    for user in df['user'].unique():
        dataframe = df[df['user']==user]
        features = dataframe[[f'amp_{i}' for i in range(1,91)]].to_numpy()
#         features = MinMaxScaler().fit_transform(features)
        features_ls.append(features)
        label = dataframe[['label']].to_numpy()
        labels_ls.append(label)
    return features_ls,labels_ls

X_ls, y_ls = seperate_dataframes(df_exp1)
del df_exp1


# In[6]:


from data.process_data import DatasetObject
from data import process_data
from sklearn.preprocessing import LabelEncoder

exp_1 = DatasetObject()
exp_1.import_data(X_ls, y_ls,window_size=800,slide_size=100,skip_labels=['noactivity'])


# In[7]:


exp_1.data_transform(lambda arr: arr.reshape(*arr.shape,1),axis=1,col=0)
exp_1.data_transform(lambda x,y,z : process_data.resampling(x,y,z,True),axis=0,col=0)
exp_1.__len__()
# exp_1.label_encode(1,LabelEncoder())
# exp_1.label_encode(2)
# exp_1.__len__()


# In[8]:


del X_ls,y_ls


# In[ ]:





# In[9]:


def transform_pipeline(X_train,y_train,X_test,y_test):
    X_train = X_train.transpose(0,3,1,2)
    X_test = X_test.transpose(0,3,1,2)
    
    lb = LabelEncoder()
    y_train = lb.fit_transform(y_train)
    y_test  = lb.transform(y_test)
    return X_train,y_train,X_test,y_test

(X_train,y_train,_),(X_test,y_test,_) = exp_1.__getitem__([0],return_sets=True)
X_train,y_train,X_test,y_test = transform_pipeline(X_train,y_train,X_test,y_test)


# In[10]:


# print(X.shape)
# print(y.shape)


# ## dataloader

# In[11]:


import numpy as np
import pandas as pd


# In[12]:


import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor


# In[13]:


# traindataset = TensorDataset(Tensor(X_train),Tensor(y_train).long()) 
# testdataset = TensorDataset(Tensor(X_test), Tensor(y_test).long())
# rand_x = torch.zeros_like(Tensor(X_train))
# rand_y = torch.zeros_like(Tensor(y_train))


# In[14]:


# from sklearn.preprocessing import OneHotEncoder

# rand_x = torch.rand((128*16, 1, 800, 90))
# rand_y = torch.randint(0,8,(128*16,))
# traindataset = TensorDataset(rand_x,rand_y)


# In[15]:


# train_loader = DataLoader(traindataset, batch_size=128,shuffle=True, num_workers=0)
# test_loader = DataLoader(testdataset, batch_size=128,shuffle=True, num_workers=0)


# In[30]:


def create_dataloaders(X_train,y_train,X_test,y_test):
    traindataset = TensorDataset(Tensor(X_train),Tensor(y_train).long()) 
    testdataset = TensorDataset(Tensor(X_test), Tensor(y_test).long())
    train_loader = DataLoader(traindataset, batch_size=128,shuffle=True, num_workers=0)
    test_loader = DataLoader(testdataset, batch_size=1024, shuffle=True, num_workers=0) 
    return train_loader, test_loader
    
train_loader, test_loader = create_dataloaders(X_train,y_train,X_test,y_test)


# # model

# In[17]:


import torch
from torch import nn
from torch.nn import functional as F


# In[ ]:





# In[18]:


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


# In[19]:


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        ### 1st ###
        self.conv1 = nn.Conv2d(1,64,kernel_size=5,stride=5)
        self.actv1 = nn.ReLU()
        self.norm1 = Lambda(lambda x:x)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,1))
        ### 2nd ###
        self.conv2 = nn.Conv2d(64,128,kernel_size=3,stride=3)
        self.actv2 = nn.ReLU()
        self.norm2 = Lambda(lambda x:x)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,1))
        ### 3rd ###
        self.conv3 = nn.Conv2d(128,256,kernel_size=2,stride=2)
        self.actv3 = nn.ReLU()
        self.norm3 = Lambda(lambda x:x)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,1))
    
    def forward(self,X):
        X = self.pool1(self.norm1(self.actv1(self.conv1(X))))
        X = self.pool2(self.norm2(self.actv2(self.conv2(X))))
        X = self.pool3(self.norm3(self.actv3(self.conv3(X))))
        X = torch.flatten(X, 1)
        return X
    
    
    
class Classifier(nn.Module):
    def __init__(self,input_shape):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(input_shape,128) 
        self.linear2 = nn.Linear(128,8)
    
    def forward(self,X):
        X = F.dropout(F.leaky_relu(self.linear1(X)))
        X = self.linear2(X)
        return F.log_softmax(X,dim=0)
    
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        ### 1st ###
        self.conv1 = nn.Conv2d(1,64,kernel_size=5,stride=5)
        self.actv1 = nn.ReLU()
        self.norm1 = Lambda(lambda x:x)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,1))
        ### 2nd ###
        self.conv2 = nn.Conv2d(64,128,kernel_size=3,stride=3)
        self.actv2 = nn.ReLU()
        self.norm2 = Lambda(lambda x:x)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,1))
        ### 3rd ###
        self.conv3 = nn.Conv2d(128,256,kernel_size=2,stride=2)
        self.actv3 = nn.ReLU()
        self.norm3 = Lambda(lambda x:x)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,1))
        ### Classifier ####
        self.linear1 = nn.Linear(768,128) 
        self.linear2 = nn.Linear(128,8)
    
    def forward(self,X):
        X = self.pool1(self.norm1(self.actv1(self.conv1(X))))
        X = self.pool2(self.norm2(self.actv2(self.conv2(X))))
        X = self.pool3(self.norm3(self.actv3(self.conv3(X))))
        X = torch.flatten(X, 1)
        X = F.dropout(F.leaky_relu(self.linear1(X)))
        X = self.linear2(X)
        return F.log_softmax(X,dim=0)
    
model = CNN()


# In[20]:


# encoder = Encoder()
# z  = encoder.forward(rand_x)  
# z.shape


# In[21]:


# classifier = Classifier(z.shape[1])
# out = classifier.forward(z)
# out.shape


# In[22]:


# from torchsummary import summary

# summary(encoder,input_size=rand_x.shape[1:])


# ## criterion, optimizer

# In[23]:


def setting(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)
    return criterion, optimizer

criterion, optimizer = setting(model)


# In[24]:


# criterion = nn.CrossEntropyLoss()
# optimizer_e = torch.optim.Adam(encoder.parameters(), lr=0.001)
# optimizer_c = torch.optim.Adam(classifier.parameters(), lr=0.001)
# optimizer   = torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()), lr=0.001)


# ## train

# In[25]:


def train(model, train_loader, criterion, optimizer, epoch):
    
    for i in range(epochs):
        
        for b, (X_train, y_train) in enumerate(train_loader):
            
            model.zero_grad()
            
            y_pred = model(X_train)
            loss   = criterion(y_pred, y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model

epochs = 100
model  = train(model, train_loader, criterion, optimizer, epochs)


# In[26]:


# import time
# start_time = time.time()

# epochs = 100
# train_losses = []
# test_losses = []
# train_correct = []
# test_correct = []

# for i in range(epochs):
#     trn_corr = 0
#     tst_corr = 0
    
#     # Run the training batches
#     for b, (X_train, y_train) in enumerate(train_loader):
#         b+=1
        
#         # Encoder
#         encoder.zero_grad()
#         z_train = encoder(X_train)
#         # Classifier
#         classifier.zero_grad()
#         y_pred = classifier(z_train)
#         # Calculate loss
#         loss = criterion(y_pred, y_train)
 
#         # Tally the number of correct predictions
#         predicted = torch.max(y_pred.data, 1)[1]
#         batch_corr = (predicted == y_train).sum()
#         trn_corr += batch_corr
        
#         # Update parameters
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # Print interim results
#         print(f'epoch: {i:2}  batch: {b:4}  loss: {loss.item():10.8f} accuracy: {trn_corr.item()*100/(10*b):7.3f}%')
        
#     train_losses.append(loss)
#     train_correct.append(trn_corr)
        


# In[31]:


def evalaute(model, test_loader):
    
    with torch.no_grad():
        
        for X_test, y_test in test_loader:
            
            y_val = model(X_test)
            predicted = torch.max(y_val,1)[1] 
    
    arr = confusion_matrix(y_test.view(-1), predicted.view(-1))
    return arr

arr = evalaute(model, test_loader)


# In[32]:


arr


# In[ ]:



# with torch.no_grad():
#     correct = 0
#     for X_test, y_test in test_loader:
#         y_val = classifier(encoder(X_test))
#         predicted = torch.max(y_val,1)[1]
#         correct += (predicted == y_test).sum()

# arr = confusion_matrix(y_test.view(-1), predicted.view(-1))
# arr
# df_cm = pd.DataFrame(arr, class_names, class_names)
# plt.figure(figsize = (9,6))
# sn.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
# plt.xlabel("prediction")
# plt.ylabel("label (ground truth)")
# plt.show()


# In[ ]:




