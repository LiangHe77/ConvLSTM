#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from ConvLSTM import ConvLSTM
from Dataset import traindata_loader,testdata_loader
from torch.optim import lr_scheduler


# In[2]:


def train(epoch):
    criterion = nn.MSELoss()
    net = ConvLSTM(1,16,3,2)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.7)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    epo_losses = []
    save_path = os.getcwd()
    for i in range(epoch):
        epo_loss = 0
        for batch,data in enumerate(traindata_loader):
            input = data[:,:,0:7,:,:]
            label = data[:,:,7,:,:]
            optimizer.zero_grad()
            predict_images = net(input)
            loss = criterion(label,predict_images)
            loss.backward()
            iter_loss = loss.item()
            epo_loss += iter_loss
            optimizer.step()
            print("epoch:{}/{},batch:{},loss:{:.6f}".format(i+1,epoch,batch,loss.item()))
        if (i+1)%20==0:
            torch.save(net, save_path+"/checkpoints/trained_model_{}.pkt".format(i+1))
            epo_losses.append(epo_loss)    
        scheduler.step(epo_loss)
    plt.plot(epo_losses,color=(0,0,1),label='loss') 
    min_index = np.argmin(epo_losses)
    min_data = np.min(epo_losses)
    plt.plot(min_index,min_data,"ks")
    show_min = "[{0},{1}]".format(min_index,min_data)
    plt.title("Train_Loss")
    plt.annotate(show_min,xytext=(min_index,min_data),xy=(min_index,min_data))
    plt.savefig('train_loss.jpg')
    plt.close()


# In[3]:


def test(num):
    
    file_path = os.getcwd() + '/checkpoints/trained_model_{}.pkt'.format(num)
    testnet = torch.load(file_path)
    testnet.eval()
    
    for index,data in enumerate(testdata_loader):
        input = data[:,:,0:7,:,:]
        label = data[:,:,7:14,:,:]
        buf = input
        for i in range(7):
            pred = testnet(buf)
            pred = torch.unsqueeze(pred,2)
            buf = torch.cat((buf[:,:,1:7,:,:],pred),axis=2)
        input = input.detach().numpy()
        buf = buf.detach().numpy()
        label = label.detach().numpy()
        
        #print(input.shape)
        
        for batch in range(2):
            plt.figure(batch)
            fig,axs = plt.subplots(3,7)

            for i in range(7):
                axs[0,i].imshow(input[batch][0][i]*255)
                axs[1,i].imshow(label[batch][0][i]*255)
                axs[2,i].imshow(buf[batch][0][i]*255)
            plt.savefig('E:/Code/TCN/result/result_{}.jpg'.format(index))
            plt.close(fig)
                
    np.save(os.getcwd()+'/input',input.detach().numpy())
    np.save(os.getcwd()+'/label',label.detach().numpy())
    np.save(os.getcwd()+'/pred',buf.detach().numpy())


# In[4]:


if __name__=='__main__':
    train(100)
    test(100)






