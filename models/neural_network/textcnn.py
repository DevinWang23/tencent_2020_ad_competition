import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_config import BaseConfig


class Config(BaseConfig):
    def __init__(
            self,
            dropout=0.5,
            required_improvement=1000,
            num_epochs=20,
            batch_size=128,
            learning_rate=1e-3,
            filter_size=(2, 3, 4),
            num_filters=256,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.required_improvement = required_improvement
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.pad_size = pad_size
        self.learning_rate = learning_rate
        self.filter_sizes = filter_size
        self.num_filters = num_filters

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embed_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embed_pretrained, freeze=True)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_dim, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed_dim)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out