import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
    def __init__(
        self, 
        config
    ):
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
            
#         self.dropout = nn.Dropout(config.dropout)
        self.lstm1 = nn.LSTM(
            config.embed_dim_list[0], 
            config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            batch_first=True,
            
        )
        
        self.lstm2 = nn.LSTM(
            config.embed_dim_list[1], 
            config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        self.lstm3 = nn.LSTM(
            config.embed_dim_list[2], 
            config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
#         self.lstm4 = nn.LSTM(
#             config.embed_dim_list[3], 
#             config.hidden_size,
#             num_layers=config.num_layers,
#             bidirectional=config.bidirectional,
#         )
        self.fc = nn.Linear(config.hidden_size * config.num_layers * 2 * config.num_sparse_feat, config.num_classes) if                           config.bidirectional else nn.Linear(config.hidden_size * config.num_layers * config.num_sparse_feat,                           config.num_classes)
    
    def forward(self,x):
        out1 = self.embedding1(x[0])
        out2 = self.embedding2(x[2])
        out3 = self.embedding3(x[4])
#         out4 = self.embedding4(x[6])
#         out = self.dropout(out)
        
        out1  = pack_padded_sequence(out1, x[1],enforce_sorted=False, batch_first=True)
        out2  = pack_padded_sequence(out2, x[3],enforce_sorted=False, batch_first=True)
        out3  = pack_padded_sequence(out3, x[5],enforce_sorted=False, batch_first=True)
#         out4  = pack_padded_sequence(out4, x[7],enforce_sorted=False, batch_first=False)

        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        self.lstm3.flatten_parameters()
#         self.lstm4.flatten_parameters()
        
        out1, (final_hidden_state1, final_cell_state1) = self.lstm1(out1)
        out2, (final_hidden_state2, final_cell_state2) = self.lstm2(out2)
        out3, (final_hidden_state3, final_cell_state3) = self.lstm3(out3)
        
#         out4, (final_hidden_state4, final_cell_state4) = self.lstm3(out4)
#         print(final_hidden_state1[-1].shape)
        out = torch.cat((final_hidden_state1[-1], final_hidden_state2[-1], final_hidden_state3[-1]),dim=1)
        
        logit = self.fc(out)
#         out = self.fc(self.dropout(torch.cat([ct[i,:,:] for i in range(ct.shape[0])], dim=1)))
        return logit