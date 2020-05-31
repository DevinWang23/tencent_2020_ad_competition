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
       
        if config.embed_pretrained_list[0] is not None:
            self.embedding1 = nn.Embedding.from_pretrained(
                config.embed_pretrained_list[0], 
                freeze=True
            )
        
        if config.embed_pretrained_list[1] is not None:
            self.embedding2 = nn.Embedding.from_pretrained(
                config.embed_pretrained_list[1], 
                freeze=True
            )
        
        if config.embed_pretrained_list[2] is not None:
            self.embedding3 = nn.Embedding.from_pretrained(
                config.embed_pretrained_list[2], 
                freeze=True
            )
        
#         if config.embed_pretrained_list[3] is not None:
#             self.embedding4 = nn.Embedding.from_pretrained(
#                 config.embed_pretrained_list[3], 
#                 freeze=True
#             )
       
    #         self.lstm = nn.LSTM(
    #             config.embed_dim, config.hidden_size, config.num_layers, batch_first=True,                                     bidirectional=config.bidirectional)

        
        self.lstm1 = nn.LSTM(
            config.embed_dim_list[0],
            config.hidden_size, 
            config.num_layers,                              
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        self.lstm2 = nn.LSTM(
            config.embed_dim_list[1],
            config.hidden_size, 
            config.num_layers,                         
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        self.lstm3 = nn.LSTM(
            config.embed_dim_list[2],
            config.hidden_size, 
            config.num_layers,                             
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        self.attention_layer1 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(inplace=True)
        )
        
        self.attention_layer2 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(inplace=True)
        )
        
        self.attention_layer3 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(inplace=True)
        )
        
#         self.num_layers = config.num_layers
#         self.hidden_size = config.hidden_size
        
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size*config.num_sparse_feat, config.hidden_size*config.num_sparse_feat),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size*config.num_sparse_feat, config.num_classes)
        )

    def attention_net_with_w(self, lstm_out, lstm_hidden, attention_layer):
        """
        cal the attention between all hidden state and the last hidden state
        """
        # lstm_hidden - [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        lstm_hidden = lstm_hidden.unsqueeze(1)
        
        # atten_w - [batch_size, 1, n_hidden]
        atten_w = attention_layer(lstm_hidden)
        m = nn.Tanh()(lstm_out)
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        softmax_w = F.softmax(atten_context, dim=-1)
        context = torch.bmm(softmax_w, lstm_out)
        res = context.squeeze(1)
        return res

    def forward(self, x):
#         batch_size = x[0].size(0)
#         h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda())
#         c_0 = Variable(torch.zeros(self.num_layers,batch_size, self.hidden_size).cuda())

        out1 = self.embedding1(x[0])
        out2 = self.embedding2(x[2])
        out3 = self.embedding3(x[4])
#         out4 = self.embedding4(x[6])

        out1  = pack_padded_sequence(out1, x[1],enforce_sorted=False, batch_first=True)
        out2  = pack_padded_sequence(out2, x[3],enforce_sorted=False, batch_first=True)
        out3  = pack_padded_sequence(out3, x[5],enforce_sorted=False, batch_first=True)
#         out4  = pack_padded_sequence(out4, x[7],enforce_sorted=False, batch_first=True)

        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        self.lstm3.flatten_parameters()
#         self.lstm4.flatten_parameters()
        
        out1, (final_hidden_state1, final_cell_state1) = self.lstm1(out1)
        out2, (final_hidden_state2, final_cell_state2) = self.lstm2(out2)
        out3, (final_hidden_state3, final_cell_state3) = self.lstm3(out3)
#         out4, (final_hidden_state4, final_cell_state4) = self.lstm3(out4)

        padded_out1, input_sizes1 = pad_packed_sequence(out1, batch_first=True)
        padded_out2, input_sizes2 = pad_packed_sequence(out2, batch_first=True)
        padded_out3, input_sizes3 = pad_packed_sequence(out3, batch_first=True)
        
        final_hidden_state1 = final_hidden_state1.permute(1, 0, 2)
        final_hidden_state2 = final_hidden_state2.permute(1, 0, 2)
        final_hidden_state3 = final_hidden_state3.permute(1, 0, 2)
      
        atten_out1 = self.attention_net_with_w(padded_out1, final_hidden_state1, self.attention_layer1)
        atten_out2 = self.attention_net_with_w(padded_out2, final_hidden_state2, self.attention_layer2)
        atten_out3 = self.attention_net_with_w(padded_out3, final_hidden_state3, self.attention_layer3)
        
        atten_out = torch.cat((atten_out1,atten_out2,atten_out3), dim=1)
        return self.fc(atten_out)
