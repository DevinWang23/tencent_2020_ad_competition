# -*- coding: utf-8 -*-
"""
Author:  MengQiu Wang
Email: wangmengqiu@ainnovation.com
Date: 14/02/2020

Description: 
   Class for transformer Model
    
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base_config import BaseConfig


class Config(BaseConfig):
    def __init__(
            self,
            dropout=0.2,
            required_improvement=1000,
            num_epochs=20,
            batch_size=128,
            learning_rate=5e-4,
            dim_model=300,
            hidden=1024,
            last_hidden=512,
            num_head=5,
            num_encoder=2,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.required_improvement = required_improvement
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dim_model = dim_model
        self.hidden = hidden
        self.last_hidden = last_hidden
        self.num_head = num_head
        self.num_encoder = num_encode


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embed_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embed_pretrained, freeze=True)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_dim, padding_idx=config.n_vocab - 1)

        self.position_embedding = PositionalEncoding(config.embed_dim, config.pad_size, config.dropout, config.device)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])

        self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)

    def forward(self, x):
        out = self.embedding(x[0])
        out = self.position_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(dim_model, num_head, dropout)
        self.feed_forward = PositionWiseFeedForward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, scale=None, atten_mask=None):
        """"
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
            atten_mask: mask for padding token
        Return:
            self-attention后的张量，以及attention张量
        """
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # [batch_size * num_head, seq_len, seq_len]
        if scale:
            attention = attention * scale
        if atten_mask:
            attention = attention.masked_fill_(atten_mask, -1e9)
        attention = F.softmax(attention, dim=-1)  # normalized by softmax
        context = torch.matmul(attention, V)  # [batch_size*num_head, seq_len, dim_head]
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0, 'dim model must be divided by num_head'
        self.dim_head = dim_model // self.num_head

        # query, key and value are [batch_size, seq_len, dim_model]
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)

        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x, atten_mask=None):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)

        # [batch size * num_head, seq_len, dim_head]
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)

        # mask for padding token
        if atten_mask:
            atten_mask = atten_mask.repeat(self.num_head, 1, 1)

        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale, atten_mask)  # 通过query和key的相似度来决定value的权重分布
        context = context.view(batch_size, -1, self.dim_head * self.num_head)  # [batch_size, seq_len, dim_model], 相当于concat
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # short connection
        out = self.layer_norm(out)
        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # short connection
        out = self.layer_norm(out)
        return out
