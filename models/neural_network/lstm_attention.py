# -*- coding: utf-8 -*-
"""
Author: MengQiu Wang 
Email: wangmengqiu@ainnovation.com
Date: 03/01/2020

Description:
    Class for implementing the Lstm+Attention
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import (
pack_padded_sequence,
pad_packed_sequence
)

from .base_config import BaseConfig


class Config(BaseConfig):
    def __init__(
            self,
            dropout=0.5,
            required_improvement=1000,
            num_epochs=20,
            batch_size=128,
            learning_rate=1e-3,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.required_improvement = required_improvement
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embed_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embed_pretrained, freeze=True)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_dim, padding_idx=config.n_vocab - 1)
       
        self.lstm = nn.LSTM(config.embed_dim, config.hidden_size, config.num_layers, batch_first=True,                                     bidirectional=config.bidirectional)
        self.attention_layer = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                             nn.ReLU(inplace=True))
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes)
        )
        

    def attention_net_with_w(self, lstm_out, lstm_hidden):
        """
        cal the attention between all hidden state and the last hidden state
        """
        # lstm_hidden - [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        lstm_hidden = lstm_hidden.unsqueeze(1)
        
        # atten_w - [batch_size, 1, n_hidden]
        atten_w = self.attention_layer(lstm_hidden)
        m = nn.Tanh()(lstm_out)
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        softmax_w = F.softmax(atten_context, dim=-1)
        context = torch.bmm(softmax_w, lstm_out)
        res = context.squeeze(1)
        return res

    def forward(self, x):
        batch_size = x[0].size(0)
        h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(self.num_layers,batch_size, self.hidden_size).cuda())
        output = self.embedding(x[0])
        output  = pack_padded_sequence(output, x[1],enforce_sorted=False, batch_first=True)
        output, (final_hidden_state, final_cell_state) = self.lstm(output, (h_0, c_0))
        padded_out, input_sizes = pad_packed_sequence(output, batch_first=True)
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        atten_out = self.attention_net_with_w(padded_out, final_hidden_state)
        return self.fc(atten_out)
