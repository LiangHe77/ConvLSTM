#!/usr/bin/env python
# coding: utf-8

# In[35]:


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# In[36]:


class MovingMNIST(Dataset):
    def __init__(self,path='E:\数据集\Moving MNIST\mnist_test_seq.npy',transforms=None,train=True):
        self.path = path
        self.content = np.load(self.path)
        #self.content = (self.content-np.min(self.content))/(np.max(self.content)-np.min(self.content))
        #self.content = np.resize(self.content,(8,7500,1,64,64))
        self.content_ = self.content.transpose(1,0,2,3)  #(batch_size,channels,seq_nums,height,width)
        self.transforms = transforms
        self.train = train
        if train:
            self.Total = len(self.content_[:40,:,:,:])
        else:
            self.Total = len(self.content_[8000:10000,:,:,:])
        
    def __getitem__(self,index):
        if self.train:
            self.train_dataset = self.content_[0:40,:,:,:]
            self.sample_ = self.train_dataset[index,...]/255.0
            self.sample = torch.from_numpy(np.expand_dims(self.sample_,axis=0)).float()
            return self.sample
        else:
            self.test_dataset = self.content_[8000:10000,:,:,:]
            self.sample_ =  self.test_dataset[index,...]/255.0
            self.sample = torch.from_numpy(np.expand_dims(self.sample_,axis=0)).float()
            return self.sample
    def __len__(self):
        return self.Total


# In[ ]:


transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45)
    ])


# In[37]:


train_mnistdata = MovingMNIST(train=True,transforms=transforms)
traindata_loader = DataLoader(dataset=train_mnistdata,batch_size=2,shuffle=True)
test_mnistdata = MovingMNIST(train=False)
testdata_loader = DataLoader(dataset=test_mnistdata,batch_size=2,shuffle=False)


# In[ ]:



                  


# In[ ]:




