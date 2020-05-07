import torch
import torch.nn as nn
import numpy as np
import itertools
import os
import torch.optim as optim
from torch.autograd import Variable
from Dataset import traindata_loader,testdata_loader


class ConvLSTMCell(nn.Module):
    def __init__(self,in_channels,hidden_channels,kernel_size):
        super(ConvLSTMCell,self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(self.in_channels+self.hidden_channels,4*hidden_channels,kernel_size,1,\
                             padding=kernel_size//2)
        
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
        return (Variable(torch.zeros(batch_size,self.hidden_channels,h,w)).cuda(),
               Variable(torch.zeros(batch_size,self.hidden_channels,h,w)).cuda())

class Model(nn.Module):
    def __init__(self,in_channels,hidden_channels,kernel_size,num_layers):
        super(Model,self).__init__()
       
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        cell_list = []
        cell_list.append(ConvLSTMCell(self.in_channels,self.hidden_channels[0],self.kernel_size).cuda())
        for idlayer in range(1,num_layers):
            cell_list.append(ConvLSTMCell(self.hidden_channels[idlayer-1],self.hidden_channels[idlayer],\
                                         self.kernel_size).cuda())
        self.cell_list = nn.ModuleList(cell_list)
        self.conv = nn.Conv3d(self.hidden_channels[-1],1,(1,self.kernel_size,self.kernel_size),1,\
                              (0,self.kernel_size//2,self.kernel_size//2))
        
    def forward(self,input,hidden_states=None):
        
        cur_input = input
        next_hidden = []
        (batch_size,channels,seq_num,height,width) = cur_input.size()
        
        #if hidden_states is not None:
        #    raise NotImplementedError
        #else:
        if hidden_states is None:
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
         
        out = self.conv(layer_output_list[-1][:,:,-1:,:,:])
       
        return out,next_hidden
    
    def init_hidden(self,batch_size,image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size,image_size))
        return init_states   

    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        
        self.Encoder = Model(1,[8,16],3,2)
        
    def forward(self,input):
        
        out,next_hidden = self.Encoder(input)
        
        return out,next_hidden



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        
        self.Decoder = Model(1,[8,16],3,2)
        
    def forward(self,input,hidden_states=None):
        
        output,hidden_states = self.Decoder(input,hidden_states)

        return output,hidden_states


if __name__=='__main__':
    criterion = nn.MSELoss().cuda()
    encoder = Encoder().cuda()
    decoder = Decoder().cuda()
    
    optimizer = optim.Adam(itertools.chain(encoder.parameters(),decoder.parameters()),lr=0.001)
    epo_losses = []
    epochs = []    
    save_path = os.getcwd()
    for i in range(1000):
        epo_loss = 0
        for batch,data in enumerate(traindata_loader):
	    iter_loss = 0
	    result_ = []
	    data=data.cuda()
            input = data[:,:,0:10,:,:]
            label = data[:,:,10:20,:,:]
	    out,hidden = encoder(input)
            optimizer.zero_grad()
	    for _ in range(input.shape[2]):
		out,hidden = decoder(out,hidden)
		result_.append(out)
	    result = torch.cat(result_,2)
            loss = criterion(label,result)
            loss.backward()
            iter_loss = loss.item()
            epo_loss += iter_loss
            optimizer.step()
            print("epoch:{}/{},batch:{},loss:{}".format(i+1,10,batch,loss.item()))
    
    torch.save(encoder,save_path+"/checkpoints/encoder.pkl")
    torch.save(decoder,save_path+"/checkpoints/decoder.pkl")
