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
            
#         if config.embed_pretrained_list[4] is not None:
#             self.embedding5 = nn.Embedding.from_pretrained(
#                 config.embed_pretrained_list[4], 
#                 freeze=True
#             )    
            
#         if config.embed_pretrained_list[5] is not None:
#             self.embedding6 = nn.Embedding.from_pretrained(
#                 config.embed_pretrained_list[5], 
#                 freeze=True
#             ) 
        
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
        
#         self.attention_layer4 = nn.Sequential(
#          nn.Linear(config.hidden_size, config.hidden_size),
#          nn.ReLU(inplace=True)
#         )
        
#         self.attention_layer5 = nn.Sequential(
#          nn.Linear(config.hidden_size, config.hidden_size),
#          nn.ReLU(inplace=True)
#         )
        
#         self.attention_layer6 = nn.Sequential(
#          nn.Linear(config.hidden_size, config.hidden_size),
#          nn.ReLU(inplace=True)
#         )
        
        self.lstm1 = nn.LSTM(
            config.embed_dim_list[0],
            config.hidden_size, 
            config.num_layers, 
            dropout=config.dropout,                                 
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        self.lstm2 = nn.LSTM(
            config.embed_dim_list[1],
            config.hidden_size, 
            config.num_layers, 
            dropout=config.dropout,                                 
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        self.lstm3 = nn.LSTM(
            config.embed_dim_list[2],
            config.hidden_size, 
            config.num_layers, 
            dropout=config.dropout,                                 
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
#         self.lstm4 = nn.LSTM(
#             config.embed_dim_list[3],
#             config.hidden_size, 
#             config.num_layers, 
#             dropout=config.dropout,                                 
#             bidirectional=config.bidirectional,
#             batch_first=True
#         )
        
#         self.lstm5 = nn.LSTM(
#             config.embed_dim_list[4],
#             config.hidden_size, 
#             config.num_layers, 
#             dropout=config.dropout,                                 
#             bidirectional=config.bidirectional,
#             batch_first=True
#         )
        
#         self.lstm6 = nn.LSTM(
#             config.embed_dim_list[5],
#             config.hidden_size, 
#             config.num_layers, 
#             dropout=config.dropout,                                 
#             bidirectional=config.bidirectional,
#             batch_first=True
#         )
        
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size*config.num_sparse_feat, config.hidden_size*config.num_sparse_feat),
#             nn.Linear(config.hidden_size*config.num_sparse_feat, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size*config.num_sparse_feat, config.num_classes)
#             nn.Linear(config.hidden_size, config.num_classes)
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
        out3 = self.embedding3(x[4])
#         out4 = self.embedding4(x[6])
#         out5 = self.embedding5(x[8])
#         out6 = self.embedding6(x[10])

        out1  = pack_padded_sequence(out1, x[1],enforce_sorted=False, batch_first=True)
        out2  = pack_padded_sequence(out2, x[3],enforce_sorted=False, batch_first=True)
        out3  = pack_padded_sequence(out3, x[5],enforce_sorted=False, batch_first=True)
#         out4  = pack_padded_sequence(out4, x[7],enforce_sorted=False, batch_first=True)
#         out5  = pack_padded_sequence(out5, x[9],enforce_sorted=False, batch_first=True)
#         out6  = pack_padded_sequence(out6, x[11],enforce_sorted=False, batch_first=True)

        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        self.lstm3.flatten_parameters()
#         self.lstm4.flatten_parameters()
#         self.lstm5.flatten_parameters()
#         self.lstm6.flatten_parameters()
        
        out1, (final_hidden_state1, final_cell_state1) = self.lstm1(out1)
        out2, (final_hidden_state2, final_cell_state2) = self.lstm2(out2)
        out3, (final_hidden_state3, final_cell_state3) = self.lstm3(out3)
#         out4, (final_hidden_state4, final_cell_state4) = self.lstm4(out4)
#         out5, (final_hidden_state5, final_cell_state5) = self.lstm5(out5)
#         out6, (final_hidden_state6, final_cell_state6) = self.lstm6(out6)
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
        out3, input_sizes3 = pad_packed_sequence(out3, batch_first=True)
#         out4, input_sizes4 = pad_packed_sequence(out4, batch_first=True)
#         out5, input_sizes5 = pad_packed_sequence(out5, batch_first=True)
#         out6, input_sizes6 = pad_packed_sequence(out6, batch_first=True)
        
#         print(out.data.shape, final_hidden_state.shape)
        final_hidden_state1 = final_hidden_state1.permute(1,0,2)
        final_hidden_state2 = final_hidden_state2.permute(1,0,2)
        final_hidden_state3 = final_hidden_state3.permute(1,0,2)
#         final_hidden_state4 = final_hidden_state4.permute(1,0,2)
#         final_hidden_state5 = final_hidden_state5.permute(1,0,2)
#         final_hidden_state6 = final_hidden_state6.permute(1,0,2)
        
        attn_output1 = self.attention_net(out1, final_hidden_state1, self.attention_layer1)
        attn_output2 = self.attention_net(out2, final_hidden_state2, self.attention_layer2)
        attn_output3 = self.attention_net(out3, final_hidden_state3, self.attention_layer3)
#         attn_output4 = self.attention_net(out4, final_hidden_state4, self.attention_layer4)
#         attn_output5 = self.attention_net(out5, final_hidden_state5, self.attention_layer5)
#         attn_output6 = self.attention_net(out6, final_hidden_state6, self.attention_layer6)
        
#         attn_output = torch.cat((attn_output1, attn_output2, attn_output3, attn_output4),1)
#         attn_output = torch.cat((attn_output1, attn_output2, attn_output3, attn_output4, attn_output5, attn_output6),1)
        attn_output = torch.cat((attn_output1, attn_output2, attn_output3),1)
#         print(attn_output.shape)
        logits = self.fc(attn_output)
        return logits
    