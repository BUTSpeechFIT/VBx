#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Authors: Shuai Wang, Federico Landini, Johan Rohdin
# @Emails: wsstriving@gmail.com, landini@fit.vutbr.cz, rohdin@fit.vutbr.cz

# Code for loading the model described in 
# F. Landini, S. Wang, M. Diez, L. Burget et al.
# BUT System for the Second DIHARD Speech Diarization Challenge, ICASSP 2020

import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TDNN(nn.Module):
    def __init__(self, feat_dim=40):
        super(TDNN, self).__init__()
        self.conv_1 = nn.Conv1d(feat_dim,1024, 5)
        self.bn_1 = nn.BatchNorm1d(1024, affine=False)
        self.conv_2 = nn.Conv1d(1024,1024, 1)
        self.bn_2 = nn.BatchNorm1d(1024, affine=False)
        self.conv_3 = nn.Conv1d(1024, 1024, 5, dilation=2)
        self.bn_3 = nn.BatchNorm1d(1024, affine=False)
        self.conv_4 = nn.Conv1d(1024,1024, 1)
        self.bn_4 = nn.BatchNorm1d(1024, affine=False)
        self.conv_5 = nn.Conv1d(1024, 1024, 3, dilation=3)
        self.bn_5 = nn.BatchNorm1d(1024, affine=False)
        self.conv_6 = nn.Conv1d(1024,1024, 1)
        self.bn_6 = nn.BatchNorm1d(1024, affine=False)
        self.conv_7 = nn.Conv1d(1024, 1024, 3, dilation=4)
        self.bn_7 = nn.BatchNorm1d(1024, affine=False)
        self.conv_8 = nn.Conv1d(1024,1024, 1)
        self.bn_8 = nn.BatchNorm1d(1024, affine=False)
        self.conv_9 = nn.Conv1d(1024, 2000, 1)
        self.bn_9 = nn.BatchNorm1d(2000, affine=False)

        self.dense_10 = nn.Linear(6048, 512)
        self.bn_10 = nn.BatchNorm1d(512, affine=False)
        self.dense_11 = nn.Linear(512, 512)
        self.bn_11 = nn.BatchNorm1d(512, affine=False)
        self.dense_12 = nn.Linear(512, 7146)


    def forward(self, x):
        # Repeat first and last frames to allow an xvector to be computed even for short segments
        # 13 is obtained by summing all the time delays: (5/2)*1 + (5/2)*2 + (3/2)*3 + (3/2)*4
        x = torch.cat([x[:,:,0].resize(1,40,1).repeat(1,1,13), x, x[:,:,-1].resize(1,40,1).repeat(1,1,13)], 2)

        out = self.conv_1(x)
        out = F.relu(out)
        out = self.bn_1(out)

        out = self.conv_2(out)
        out = F.relu(out)
        out = self.bn_2(out)

        out = self.conv_3(out)
        out = F.relu(out)
        out = self.bn_3(out)

        out = self.conv_4(out)
        out = F.relu(out)
        out = self.bn_4(out)
        
        out = self.conv_5(out)
        out = F.relu(out)
        out = self.bn_5(out)

        out = self.conv_6(out)
        out = F.relu(out)
        out = self.bn_6(out)
        
        out = self.conv_7(out)
        out = F.relu(out)
        out = self.bn_7(out)
        out7 = out

        out = self.conv_8(out)
        out = F.relu(out)
        out = self.bn_8(out)
        
        out = self.conv_9(out)
        out = F.relu(out)
        out = self.bn_9(out)
        out9 = out

        pooling_mean7 = torch.mean(out7, dim=2)
        pooling_std7 = torch.std(out7, dim=2, unbiased=True)

        pooling_mean9 = torch.mean(out9, dim=2)
        pooling_std9 = torch.std(out9, dim=2, unbiased=True)

        stats = torch.cat((pooling_mean7, pooling_std7, pooling_mean9, pooling_std9), 1)

        embedding_1 = self.dense_10(stats)
        out = F.relu(embedding_1)
        out = self.bn_10(out)

        embedding_2 = self.dense_11(out)
        out = F.relu(embedding_2)
        out = self.bn_11(out)

        out = self.dense_12(out)

        return out, embedding_1, embedding_2


def load_kaldi_model(model_file):
    with open (model_file) as f:
        weights         = []
        biases          = []
        batchnorm_means = []
        batchnorm_vars  = []

        r1 = re.compile('\<ComponentName\> (tdnn[1-9](|a).affine) .*\[')
        r2 = re.compile('\<ComponentName\> (tdnn[1-9](|a).batchnorm) .*\[')
        r3 = re.compile('\<ComponentName\> (output.affine) .*\[')

        l = f.readline()
        while l:
            m1 = r1.match( l )
            m3 = r3.match( l )
            if (m1 != None ) or (m3 != None ):
                if (m1 != None ):
                    m = m1
                else:
                    m=  m3

                print ("Loading component " + m.group(1))
                w = []
                while True:
                    l = f.readline()
                    a = l.rsplit(' ')
                    w.append(np.array(a[2:-1],dtype='float32'))

                    if ( a[-1] == ']\n' ):
                        weights.append(np.array(w))
                        print (" Weights loaded")
                        break

                l = f.readline()
                a = l.rsplit(' ')
                biases.append(np.array(a[3:-1], dtype='float32'))
                print (" Bias loaded")

            l = f.readline()

            m2 = r2.match( l )
            if (m2 != None ):
                print ("Loading batchnorm " + m.group(1))
                a = l.rsplit(' ')
                batchnorm_means.append(np.array(a[18:-1], dtype='float32'))
                print (" Batchnorm mean loaded")

                l = f.readline()
                a = l.rsplit(' ')
                batchnorm_vars.append(np.array(a[3:-1], dtype='float32'))
                print (" Batchnorm variance loaded")


    tdnn = TDNN(feat_dim=40)

    tdnn.conv_1.weight.data = torch.from_numpy(weights[0]).reshape(1024, -1, 40).transpose(1,2)
    tdnn.conv_1.bias.data = torch.from_numpy(biases[0])
    tdnn.bn_1.running_mean = torch.from_numpy(batchnorm_means[0])
    tdnn.bn_1.running_var = torch.from_numpy(batchnorm_vars[0])
    tdnn.bn_1.eps=1e-3

    weight_mat = torch.from_numpy(weights[1])
    tdnn.conv_2.weight.data = weight_mat.reshape(weight_mat.size(0), -1, tdnn.conv_2.weight.data.size(1)).transpose(1,2)
    tdnn.conv_2.bias.data = torch.from_numpy(biases[1])
    tdnn.bn_2.running_mean = torch.from_numpy(batchnorm_means[1])
    tdnn.bn_2.running_var = torch.from_numpy(batchnorm_vars[1])
    tdnn.bn_2.eps=1e-3

    weight_mat = torch.from_numpy(weights[2])
    tdnn.conv_3.weight.data = weight_mat.reshape(weight_mat.size(0), -1, tdnn.conv_3.weight.data.size(1)).transpose(1,2)
    tdnn.conv_3.bias.data = torch.from_numpy(biases[2])
    tdnn.bn_3.running_mean = torch.from_numpy(batchnorm_means[2])
    tdnn.bn_3.running_var = torch.from_numpy(batchnorm_vars[2])
    tdnn.bn_3.eps=1e-3

    weight_mat = torch.from_numpy(weights[3])
    tdnn.conv_4.weight.data = weight_mat.reshape(weight_mat.size(0), -1, tdnn.conv_4.weight.data.size(1)).transpose(1,2)
    tdnn.conv_4.bias.data = torch.from_numpy(biases[3])
    tdnn.bn_4.running_mean = torch.from_numpy(batchnorm_means[3])
    tdnn.bn_4.running_var = torch.from_numpy(batchnorm_vars[3])
    tdnn.bn_4.eps=1e-3

    weight_mat = torch.from_numpy(weights[4])
    tdnn.conv_5.weight.data = weight_mat.reshape(weight_mat.size(0), -1, tdnn.conv_5.weight.data.size(1)).transpose(1,2)
    tdnn.conv_5.bias.data = torch.from_numpy(biases[4])
    tdnn.bn_5.running_mean = torch.from_numpy(batchnorm_means[4])
    tdnn.bn_5.running_var = torch.from_numpy(batchnorm_vars[4])
    tdnn.bn_5.eps=1e-3

    weight_mat = torch.from_numpy(weights[5])
    tdnn.conv_6.weight.data = weight_mat.reshape(weight_mat.size(0), -1, tdnn.conv_6.weight.data.size(1)).transpose(1,2)
    tdnn.conv_6.bias.data = torch.from_numpy(biases[5])
    tdnn.bn_6.running_mean = torch.from_numpy(batchnorm_means[5])
    tdnn.bn_6.running_var = torch.from_numpy(batchnorm_vars[5])
    tdnn.bn_6.eps=1e-3

    weight_mat = torch.from_numpy(weights[6])
    tdnn.conv_7.weight.data = weight_mat.reshape(weight_mat.size(0), -1, tdnn.conv_7.weight.data.size(1)).transpose(1,2)
    tdnn.conv_7.bias.data = torch.from_numpy(biases[6])
    tdnn.bn_7.running_mean = torch.from_numpy(batchnorm_means[6])
    tdnn.bn_7.running_var = torch.from_numpy(batchnorm_vars[6])
    tdnn.bn_7.eps=1e-3

    weight_mat = torch.from_numpy(weights[7])
    tdnn.conv_8.weight.data = weight_mat.reshape(weight_mat.size(0), -1, tdnn.conv_8.weight.data.size(1)).transpose(1,2)
    tdnn.conv_8.bias.data = torch.from_numpy(biases[7])
    tdnn.bn_8.running_mean = torch.from_numpy(batchnorm_means[7])
    tdnn.bn_8.running_var = torch.from_numpy(batchnorm_vars[7])
    tdnn.bn_8.eps=1e-3

    weight_mat = torch.from_numpy(weights[8])
    tdnn.conv_9.weight.data = weight_mat.reshape(weight_mat.size(0), -1, tdnn.conv_9.weight.data.size(1)).transpose(1,2)
    tdnn.conv_9.bias.data = torch.from_numpy(biases[8])
    tdnn.bn_9.running_mean = torch.from_numpy(batchnorm_means[8])
    tdnn.bn_9.running_var = torch.from_numpy(batchnorm_vars[8])
    tdnn.bn_9.eps=1e-3

    tdnn.dense_10.weight.data = torch.from_numpy(weights[9])
    tdnn.dense_10.bias.data = torch.from_numpy(biases[9])
    tdnn.bn_10.running_mean = torch.from_numpy(batchnorm_means[9])
    tdnn.bn_10.running_var = torch.from_numpy(batchnorm_vars[9])
    tdnn.bn_10.eps=1e-3
    tdnn.dense_11.weight.data = torch.from_numpy(weights[10])
    tdnn.dense_11.bias.data = torch.from_numpy(biases[10])
    tdnn.bn_11.running_mean = torch.from_numpy(batchnorm_means[10])
    tdnn.bn_11.running_var = torch.from_numpy(batchnorm_vars[10])
    tdnn.bn_11.eps=1e-3

    tdnn.dense_12.weight.data = torch.from_numpy(weights[11])
    tdnn.dense_12.bias.data = torch.from_numpy(biases[11])

    return tdnn