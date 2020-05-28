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
        if config.embed_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embed_pretrained, freeze=True)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_dim, padding_idx=config.n_vocab - 1)
            
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(
            config.embed_dim, 
            config.hidden_size,
            num_layers=config.num_layers,
#             dropout=config.dropout,
            bidirectional=config.bidirectional,
        )
#             batch_first=True)
        self.fc = nn.Linear(config.hidden_size * config.num_layers * 2, config.num_classes) if config.bidirectional else                   nn.Linear(config.hidden_size * config.num_layers, config.num_classes)
    
    def forward(self,x):
        out = self.embedding(x[0])
#         out = self.dropout(out)
        out = torch.transpose(out, dim0=1,dim1=0)
        out = pack_padded_sequence(out, x[1], batch_first=False, enforce_sorted=False)
        out, (ht, ct) = self.lstm(out)
        out = self.fc(ht[-1])
#         out = self.fc(self.dropout(torch.cat([ct[i,:,:] for i in range(ct.shape[0])], dim=1)))
        return out