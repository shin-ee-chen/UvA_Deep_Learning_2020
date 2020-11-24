"""
This module implements a LSTM with peephole connections in PyTorch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class peepLSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(peepLSTM, self).__init__()

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        embedding_dim = 6
        self.embeddings = nn.Embedding(3, embedding_dim)

        # define parameters
        self.W_fx = nn.Parameter(torch.Tensor(hidden_dim, embedding_dim))
        self.W_fh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.Tensor(hidden_dim, 1))

        self.W_ix = nn.Parameter(torch.Tensor(hidden_dim, embedding_dim))
        self.W_ih = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.Tensor(hidden_dim, 1))

        self.W_ox = nn.Parameter(torch.Tensor(hidden_dim, embedding_dim))
        self.W_oh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.Tensor(hidden_dim, 1))

        self.W_cx = nn.Parameter(torch.Tensor(hidden_dim, embedding_dim))
        self.b_c = nn.Parameter(torch.Tensor(hidden_dim, 1))

        self.W_ph = nn.Parameter(torch.Tensor(num_classes, hidden_dim))
        self.b_p = nn.Parameter(torch.Tensor(num_classes, 1))
        
        self.c_init = nn.Parameter(torch.Tensor(hidden_dim, batch_size),requires_grad=False)

        self.device = device
        self.seq_length = seq_length
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=-1)
        # init
        for p in self.parameters():
            nn.init.kaiming_normal_(p, nonlinearity='linear')
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # h_prev = self.h_init
        c_prev = self.c_init

        x = self.embeddings(torch.squeeze(x.long()))
        # print(x[:, 0].shape)
        for t in range(self.seq_length - 1):
            f_t = self.sigmoid(self.W_fx @ x[:, t].t() + self.W_fh @ c_prev + self.b_f)
            i_t = self.sigmoid(self.W_ix @ x[:, t].t() + self.W_ih @ c_prev + self.b_i)
            o_t = self.sigmoid(self.W_ox @ x[:, t].t() + self.W_oh @ c_prev + self.b_o)

            c_t = self.sigmoid(self.W_cx @ x[:, t].t() + self.b_c) * i_t + c_prev * f_t
            h_t = self.tanh(c_t) * o_t
            c_prev = c_t
        
        p_t = self.W_ph @ h_t + self.b_p
        out = self.softmax(p_t.t())
        return out
        ########################
        # END OF YOUR CODE    #
        #######################

