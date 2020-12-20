from torch.utils.data import Dataset

class DatasetObject(Dataset):

    def __init__(self):
        self.data = []
        self.encoders = np.empty(shape=(1,3),dtype=np.ndarray)

    def import_data(self,features_ls,labels_ls,window_size,slide_size,skip_labels=None):
#         self.data = []
#         assert len(features_ls) == len(labels_ls)
#         for item_idx,(features,labels) in enumerate(zip(features_ls,label_ls)):
#             X,y = slide_augmentation(features,labels,window_size,slide_size,skip_labels)
#             z = np.full_like(y,item_idx)
#             assert X.shape[0] == y.shape[0] == z.shape[0]
#             self.data.append([X,y,z])
#             print(f'index {item_idx} arrays sizes ------ X: ',X.shape,' Y: ',y.shape,' Z: ',z.shape)
#         self.data = np.array(self.data,dtype=object)
#         print('size of DatasetObject ------ : ',self.data.shape)

        assert len(features_ls) == len(labels_ls)
        self.data = np.empty(shape=(len(features_ls),3),dtype=np.ndarray)
        for item_idx,(features,labels) in enumerate(zip(features_ls,labels_ls)):
            X,y = slide_augmentation(features,labels,window_size,slide_size,skip_labels)
            z = np.full_like(y,item_idx)
            assert X.shape[0] == y.shape[0] == z.shape[0]
            self.data[item_idx,0] = X
            self.data[item_idx,1] = y
            self.data[item_idx,2] = z
            print(f'index {item_idx} arrays sizes ------ X: ',X.shape,' Y: ',y.shape,' Z: ',z.shape)
        print('size of DatasetObject ------ : ',self.data.shape)
        return

    # def __call__(self):
    #     X  = np.concatenate(self.data[:,0],axis=0)
    #     y  = np.concatenate(self.data[:,1],axis=0)
    #     z  = np.concatenate(self.data[:,2],axis=0)
    #     return X,y,z

    def data_transform(self,func,axis=0,col=None):
        if axis == 0:
            for i in range(self.data.shape[0]):
                self.data[i,0],self.data[i,1],self.data[i,2] = func(self.data[i,0],self.data[i,1],self.data[i,2])
        elif axis == 1:
            for i in range(self.data.shape[0]):
                self.data[i,col] = func(self.data[i,col])
        else:
            print('No transformation is made')
        return

    def __len__(self):
        for i in range(self.data.shape[0]):
            print(f'index {i} arrays sizes ------ X: ',self.data[i,0].shape,' Y: ',
                  self.data[i,1].shape,' Z: ',self.data[i,2].shape)
        print('size of DatasetObject ------ : ',self.data.shape)
        return

    def __getitem__(self, idxs, return_sets=False):
        X_test  = np.concatenate(self.data[idxs,0],axis=0)
        y_test  = np.concatenate(self.data[idxs,1],axis=0)
        z_test  = np.concatenate(self.data[idxs,2],axis=0)
        if return_sets == True:
            q = [i for i in range(self.data.shape[0])]
            for idx in idxs:
                q.remove(idx)
            X_train = np.concatenate(self.data[q,0],axis=0)
            y_train = np.concatenate(self.data[q,1],axis=0)
            z_train = np.concatenate(self.data[q,2],axis=0)
            print('train set:',q,'\ttest set:',idxs)
            return (X_train,y_train,z_train),(X_test,y_test,z_test)
        else:
            print('dataset:',idxs)
            return (X_test,y_test,z_test)

    def label_encode(self,col=1,encoder=OneHotEncoder()):
        if self.encoders[0,col] != None:
            print('array encoded')
            return
        self.encoders[0,col] = encoder
        self.encoders[0,col].fit(np.concatenate(self.data[:,col],axis=0).reshape(-1,1))
        for i in range(self.data.shape[0]):
            self.data[i,col] = self.encoders[0,col].transform(self.data[i,col].reshape(-1,1)).toarray()
        return encoder
