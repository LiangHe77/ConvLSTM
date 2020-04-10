#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


# In[ ]:


class ConvLSTMCell(nn.Module):
    def __init__(self,in_channels,hidden_channels,kernel_size):
        super(ConvLSTMCell,self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(self.in_channels+self.hidden_channels,4*hidden_channels,kernel_size,1,                             padding=kernel_size//2)
        
    def forward(self,input,hidden_states):
        hx,cx = hidden_states
        combined = torch.cat((input,hx),1)
        gates = self.conv(combined)
        
        ingate,forgetgate,cellgate,outgate = torch.split(gates,self.hidden_channels,1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        
        cy = (forgetgate*cx)+(ingate*cellgate)
        hy = outgate*torch.tanh(cy)
        
        return hy,cy
    
    def init_hidden(self,batch_size,shape):
        h,w = shape
        return (Variable(torch.zeros(batch_size,self.hidden_channels,h,w)),
               Variable(torch.zeros(batch_size,self.hidden_channels,h,w)))


# In[ ]:


class ConvLSTM(nn.Module):
    def __init__(self,in_channels,hidden_channels,kernel_size,num_layers):
        super(ConvLSTM,self).__init__()
       
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        cell_list = []
        cell_list.append(ConvLSTMCell(self.in_channels,self.hidden_channels,self.kernel_size))
        for idlayer in range(1,num_layers):
            cell_list.append(ConvLSTMCell(self.hidden_channels,self.hidden_channels,                                         self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv = nn.Conv2d(self.hidden_channels,1,self.kernel_size,1,self.kernel_size//2)
        
    def forward(self,input,hidden_states=None):
        
        cur_input = input
        next_hidden = []
        (batch_size,channels,seq_num,height,width) = cur_input.size()
        
        if hidden_states is not None:
            raise NotImplementedError
        else:
            hidden_states = self.init_hidden(batch_size,(height,width))
            
        layer_output_list = []
        last_state_list = []

        for idlayer in range(self.num_layers):
            hidden_c = hidden_states[idlayer]
            out_inner = []
            
            for t in range(seq_num):
                hidden_c = self.cell_list[idlayer](cur_input[:,:,t,:,:],hidden_c)
                out_inner.append(hidden_c[0])
            
            next_hidden.append(hidden_c)
            cur_input = torch.stack(out_inner,dim=2)
            
            layer_output_list.append(cur_input)
            last_state_list.append(hidden_c)
            
        out = self.conv(layer_output_list[-1][:,:,-1,:,:])
        
        return out
    
    def init_hidden(self,batch_size,image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size,image_size))
        return init_states        

