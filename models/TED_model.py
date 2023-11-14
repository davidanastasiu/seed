#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import time,os,sys
import math

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
from utils.utils2 import *
from sklearn.metrics import mean_absolute_percentage_error
import logging
logging.basicConfig(filename = "TED_model.log", filemode='w', level = logging.DEBUG)
random.seed('a')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderLSTM(nn.Module):
    def __init__(self, opt):
        super(EncoderLSTM, self).__init__()        
        self.hidden_dim = opt.hidden_dim
        self.layer_dim = opt.layer
        self.cnn_dim = opt.cnn_dim
        self.interval = opt.inf_interval
        cnn_d = [5,3,3,2,1] #[15,7,5,3] #[21,15,7,3] #[31,21,15,7][7,5,3,2]#
        cnn0_d = cnn_d[0]
        cnn1_d = cnn_d[1]
        cnn2_d = cnn_d[2]
        cnn3_d = cnn_d[3]
        cnn4_d = cnn_d[4]
        
        if(opt.watershed == 1):
            dim_in = 2
        else:
            dim_in = 1
     
        self.cnn0 = nn.Sequential(            
            nn.Conv1d(dim_in, self.cnn_dim, cnn0_d, stride=cnn0_d, padding=0),
            nn.ELU(True),
        )        
        self.cnn1 = nn.Sequential(            
            nn.Conv1d(dim_in, self.cnn_dim, cnn1_d, stride=cnn1_d, padding=0),
            nn.ELU(True),
        ) 
        
        self.cnn2 = nn.Sequential(            
            nn.Conv1d(dim_in, self.cnn_dim, cnn2_d, stride=cnn2_d, padding=0),
            nn.ELU(True),
        )
            
        self.cnn3 = nn.Sequential(            
            nn.Conv1d(dim_in, self.cnn_dim, cnn3_d, stride=cnn3_d, padding=0),
            nn.ELU(True),
        ) 
   
        self.cnn4 = nn.Sequential(            
            nn.Conv1d(dim_in, self.cnn_dim, cnn4_d, stride=cnn4_d, padding=0),
            nn.ELU(True),
        ) 
    
        self.lstm0 = nn.LSTM(self.cnn_dim, self.hidden_dim, self.layer_dim, dropout=0, bidirectional=True, batch_first=True)
        self.lstm1 = nn.LSTM(self.cnn_dim, self.hidden_dim, self.layer_dim, dropout=0, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(self.cnn_dim, self.hidden_dim, self.layer_dim*2, dropout=0, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(self.cnn_dim, self.hidden_dim, self.layer_dim*2, dropout=0, bidirectional=True, batch_first=True)
        self.lstm4 = nn.LSTM(self.cnn_dim, self.hidden_dim, self.layer_dim*3, dropout=0, bidirectional=True, batch_first=True)

        
    def forward(self, x, h, c):
        # Initialize hidden and cell state with zeros
        h0 = h
        c0 = c
        
        h0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).to(device)    
        h2 = torch.zeros(self.layer_dim*4, x.size(0), self.hidden_dim).to(device)
        c2 = torch.zeros(self.layer_dim*4, x.size(0), self.hidden_dim).to(device)           
        h3 = torch.zeros(self.layer_dim*6, x.size(0), self.hidden_dim).to(device)
        c3 = torch.zeros(self.layer_dim*6, x.size(0), self.hidden_dim).to(device)    
        
        x = x.permute(0,2,1)
        
        cnn_out0 = torch.tanh(self.cnn0(x))
        cnn_out0 = cnn_out0.permute(0,2,1)  
        
        cnn_out1 = torch.tanh(self.cnn1(x))
        cnn_out1 = cnn_out1.permute(0,2,1)  
        
        cnn_out2 = torch.tanh(self.cnn2(x))
        cnn_out2 = cnn_out2.permute(0,2,1)  
        
        cnn_out3 = torch.tanh(self.cnn3(x))
        cnn_out3 = cnn_out3.permute(0,2,1) 
        
        cnn_out4 = torch.tanh(self.cnn4(x))
        cnn_out4 = cnn_out4.permute(0,2,1) 
       
        hn = []
        cn = []
        out, (hn0, cn0) = self.lstm0(cnn_out0, (h0,c0))
        out, (hn1, cn1) = self.lstm1(cnn_out1, (h0,c0))
        out, (hn2, cn2) = self.lstm2(cnn_out2, (h2,c2))
        out, (hn3, cn3) = self.lstm3(cnn_out3, (h2,c2))
        out, (hn4, cn4) = self.lstm4(cnn_out4, (h3,c3))

        hn.append(hn0)
        hn.append(hn1)
        hn.append(hn2)
        hn.append(hn3)
        hn.append(hn4)

        cn.append(cn0)
        cn.append(cn1)
        cn.append(cn2)
        cn.append(cn3)
        cn.append(cn4)
        
        return hn, cn


class DecoderLSTM(nn.Module):
    def __init__(self, opt):
        super(DecoderLSTM, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.layer_dim = opt.layer
        
        self.lstm00 = nn.LSTM(2, self.hidden_dim, self.layer_dim, dropout=0, bidirectional=True, batch_first=True) #4
        self.lstm01 = nn.LSTM(self.hidden_dim*2, self.hidden_dim, self.layer_dim, dropout=0, bidirectional=True, batch_first=True) #16
        self.lstm02 = nn.LSTM(self.hidden_dim*2, self.hidden_dim, self.layer_dim*2, dropout=0, bidirectional=True, batch_first=True) #32
        self.lstm03 = nn.LSTM(self.hidden_dim*2, self.hidden_dim, self.layer_dim*2, dropout=0, bidirectional=True, batch_first=True) #96
        self.lstm04 = nn.LSTM(self.hidden_dim*2, self.hidden_dim, self.layer_dim*3, dropout=0, bidirectional=True, batch_first=True) #288

        
        self.L_out00 = nn.Linear(self.hidden_dim*2, 1)
        self.L_out01 = nn.Linear(self.hidden_dim*2, 1)
        self.L_out02 = nn.Linear(self.hidden_dim*2, 1)
        self.L_out03 = nn.Linear(self.hidden_dim*2, 1)
        self.L_out04 = nn.Linear(self.hidden_dim*2, 1)


    def forward(self, x1, x3, encoder_h, encoder_c):  # x1: time sin & cos; x3: input sequence
        # Initialize hidden and cell state with zeros
        h0 = encoder_h
        c0 = encoder_c

        aa = torch.cat([x3[:,-1:,:], x3[:,-1:,:]], dim = 1)
        aa = torch.cat([aa, aa], dim=1)  
        bb = torch.cat([x1[:,0:1,:], x1[:,72:73,:]], dim = 1)
        bb = torch.cat([bb, x1[:,144:145,:]], dim = 1)
        bb = torch.cat([bb, x1[:,216:217,:]], dim = 1)
        aa = torch.cat([aa, bb], dim=2)  
          
        # segment predict with 4 width      
        out, (hn, cn) = self.lstm00(bb, (h0[0],c0[0]))
        out00 = self.L_out00(out)
        out0 = torch.squeeze(out00) # output level0
        seg_label_p0 = out00 # aggr level 0
        
        # expand seg0 to 16 width
        for i in range(16):
            if i == 0:
                temp0 = out[:,0:1,:]
                temp1 = seg_label_p0[:,0:1]
            else:
                temp0 = torch.cat([temp0, out[:, int(i/4):int(i/4)+1, :]], dim=1)    
                temp1 = torch.cat([temp1, seg_label_p0[:, int(i/4):int(i/4)+1]], dim=1)  
         
        out, (hn, cn) = self.lstm01(temp0, (h0[1],c0[1]))
        out = temp0 + out
        out01 = self.L_out01(out)
        out1 = torch.squeeze(out01)  # output level1
        seg_label_p1 = out01 # aggr level 1

        # expand seg1 to 32 width
        for i in range(32):
            if i == 0:
                temp0 = out[:,0:1,:]
                temp1 = seg_label_p1[:,0:1]
            else:
                temp0 = torch.cat([temp0, out[:, int(i/2):int(i/2)+1, :]], dim=1)   
                temp1 = torch.cat([temp1, seg_label_p1[:, int(i/2):int(i/2)+1]], dim=1)
       
        out, (hn, cn) = self.lstm02(temp0, (h0[2],c0[2]))
        out = temp0 + out
        out02 = self.L_out02(out)
        out2 = torch.squeeze(out02)  # output level 2 
        seg_label_p2 = out02  # aggr level 2

        # expand seg1 to 96 width
        for i in range(96):
            if i == 0:
                temp0 = out[:,0:1,:]
                temp1 = seg_label_p2[:,0:1]
            else:
                temp0 = torch.cat([temp0, out[:, int(i/3):int(i/3)+1, :]], dim=1)  
                temp1 = torch.cat([temp1, seg_label_p2[:, int(i/3):int(i/3)+1]], dim=1)
     
        out, (hn, cn) = self.lstm03(temp0, (h0[3],c0[3]))
        out = temp0+out
        out03 = self.L_out03(out)
        out3 = torch.squeeze(out03) # output level 3
        seg_label_p3 = out03  # aggr level 3      
       
        # expand seg_label_p to 288 width
        for i in range(288):
            if i == 0:
                temp0 = out[:,0:1]
                temp1 = seg_label_p3[:,0:1]
            else:
                temp0 = torch.cat([temp0, out[:, int(i/3):int(i/3)+1,:]], dim=1) 
                temp1 = torch.cat([temp1, seg_label_p3[:, int(i/3):int(i/3)+1]], dim=1)   

        o4, (hn, cn) = self.lstm04(temp0, (h0[4],c0[4]))
        o4 = temp0+o4
        out04 = self.L_out04(o4)
        out4 = torch.squeeze(out04) # output level 4, hinter


        return out0, out1, out2, out3, out4

