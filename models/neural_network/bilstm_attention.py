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
import torch.nn.functional as F
from torch.autograd import Variable
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
#             attention_size=256,
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
#         self.attention_size = attention_size
        
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embed_pretrained1 is not None:
            self.embedding1 = nn.Embedding.from_pretrained(
                config.embed_pretrained1, 
                freeze=True
            )
        
        if config.embed_pretrained2 is not None:
            self.embedding2 = nn.Embedding.from_pretrained(
                config.embed_pretrained2, 
                freeze=True
            )
        
        self.attention_layer1 = nn.Sequential(
         nn.Linear(config.hidden_size, config.hidden_size),
         nn.ReLU(inplace=True)
        )
        
        self.attention_layer2 = nn.Sequential(
         nn.Linear(config.hidden_size, config.hidden_size),
         nn.ReLU(inplace=True)
        )
        
        self.lstm1 = nn.LSTM(
            config.embed_dim1,
            config.hidden_size, 
            config.num_layers, 
            dropout=config.dropout,                                 
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        self.lstm2 = nn.LSTM(
            config.embed_dim2,
            config.hidden_size, 
            config.num_layers, 
            dropout=config.dropout,                                 
            bidirectional=config.bidirectional,
            batch_first=True
        )
  
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size*2, config.hidden_size*2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size*2, config.num_classes)
        )

    def attention_net(self, lstm_out, lstm_hidden, attention_layer):
        
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        lstm_hidden = lstm_hidden.unsqueeze(1)
        atten_w = attention_layer(lstm_hidden)
        m = nn.Tanh()(h)
        atten_context = torch.bmm(atten_w, m.transpose(1,2))
        softmax_w = F.softmax(atten_context, dim=-1)
        context = torch.bmm(softmax_w, h)
        attn_output = context.squeeze(1)
        
        return attn_output

    def forward(self, x, batch_size=None):
        out1 = self.embedding1(x[0])
        out2 = self.embedding2(x[2])
#         batch_max_len = x[1].max()
        out1  = pack_padded_sequence(out1, x[1],enforce_sorted=False, batch_first=True)
        out2  = pack_padded_sequence(out2, x[3],enforce_sorted=False, batch_first=True)
#         print(out.data.shape)
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        out1, (final_hidden_state1, final_cell_state1) = self.lstm1(out1)
        out2, (final_hidden_state2, final_cell_state2) = self.lstm2(out2)
#         print(out.data.shape)
#         padded_out, input_sizes = pad_packed_sequence(out,batch_first=True)
#         print(padded_out.shape)
#         padded_out = padded_out.permute(1, 0, 2)
#         print(padded_out.shape)
#         if padded_out.shape[0] < self.max_seq_len:
#             dummy_tensor = torch.autograd.Variable(torch.zeros(self.max_seq_len - padded_out.shape[0], batch_size,                                                                self.hidden_size)).cuda()
#             padded_out = torch.cat([padded_out, dummy_tensor],0)
#         print(padded_out.shape)
        out1, input_sizes1 = pad_packed_sequence(out1, batch_first=True)
        out2, input_sizes2 = pad_packed_sequence(out2, batch_first=True)
#         print(out.data.shape, final_hidden_state.shape)
        final_hidden_state1 = final_hidden_state1.permute(1,0,2)
        final_hidden_state2 = final_hidden_state2.permute(1,0,2)
        
        attn_output1 = self.attention_net(out1, final_hidden_state1, self.attention_layer1)
        attn_output2 = self.attention_net(out2, final_hidden_state2, self.attention_layer2)
        
        attn_output = torch.cat((attn_output1, attn_output2),1)
#         print(attn_output.shape)
        logits = self.fc(attn_output)
        return logits
    