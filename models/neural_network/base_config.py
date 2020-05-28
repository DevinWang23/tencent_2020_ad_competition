import os
import sys

import torch
import numpy as np

sys.path.append('../../')
import conf

DEFAULT_EMBEDDING_SIZE = 300
DEFAULT_INITIAL_VALUE = []

class BaseConfig(object):
    r""" Base class for all configuration classes.
         Handles a few parameters common to all models' configurations.
    """

    def __init__(self, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = kwargs['model_name']

        # Attributes about input dataset
        self.train_path = os.path.join(conf.DATA_DIR, 'neural_train_fe_df.feather')
        self.test_path = os.path.join(conf.DATA_DIR, 'neural_test_fe_df.feather')
        self.num_classes = kwargs['num_classes']
        self.sparse_feat = kwargs['sparse_feat']
        self.max_seq_len = kwargs['max_seq_len']
        self.use_pad = kwargs['use_pad']
        self.init_method = kwargs['init_method']
        self.seed = kwargs['seed']
        self.vocab_paths = [os.path.join(conf.DATA_DIR, vocab_path) for vocab_path in kwargs['vocab_paths']]
        self.save_path = os.path.join(conf.TRAINED_MODEL_DIR, 'checkpoints/%s.ckpt' % self.model_name)
        self.log_path = os.path.join(conf.TRAINED_MODEL_DIR, 'torch_log/%s.log' % self.model_name)
#         self.embed_size = kwargs['']

        # Use embedding or not
        self.embed = kwargs.pop('embed', [])
        if self.embed != []:
            self.embed_pretrained1 = torch.tensor(
                np.load(os.path.join(conf.DATA_DIR, self.embed[0])).astype('float32')) \
                if self.embed[0] != '' else None
            self.embed_dim1 = self.embed_pretrained1.size(1) \
                if self.embed_pretrained1 is not None else DEFAULT_EMBEDDING_SIZE
            
            self.embed_pretrained2 = torch.tensor(
                np.load(os.path.join(conf.DATA_DIR, self.embed[1])).astype('float32')) \
                if self.embed[1] != '' else None
            self.embed_dim2 = self.embed_pretrained2.size(1) \
                if self.embed_pretrained2 is not None else DEFAULT_EMBEDDING_SIZE
            
        # Size of vocab, assigning value when train
        self.n_vocab_list = DEFAULT_INITIAL_VALUE