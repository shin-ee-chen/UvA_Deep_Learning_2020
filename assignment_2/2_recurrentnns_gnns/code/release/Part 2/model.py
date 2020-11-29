# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0', dropout_keep_prob = 0):

        super(TextGenerationModel, self).__init__()
        # Initialization here...

        embedding_dim = lstm_num_hidden
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_num_hidden, lstm_num_layers, 
                            dropout= 1 - dropout_keep_prob)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)

        self.device = device
        # self.init_state = (torch.zeros(lstm_num_layers, batch_size, lstm_num_hidden).to(device),
        #                    torch.zeros(lstm_num_layers, batch_size, lstm_num_hidden).to(device))
        self.prev_state = None
        self.lstm_num_layers = lstm_num_layers
        self.lstm_num_hidden = lstm_num_hidden

    def forward(self, x):
        # Implementation here...
        x = self.embedding(x)
        if self.prev_state == None:
            self.prev_state = (torch.zeros(self.lstm_num_layers, x.shape[1], 
                                           self.lstm_num_hidden).to(self.device),
                               torch.zeros(self.lstm_num_layers, x.shape[1], 
                                           self.lstm_num_hidden).to(self.device))

        out, state = self.lstm(x, self.prev_state)
        out = self.linear(out)
        return out, state
