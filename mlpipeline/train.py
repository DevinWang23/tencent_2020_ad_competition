# -*- coding: utf-8 -*-
"""
Author:  MengQiu Wang
Email: wangmengqiu@ainnovation.com
Date: 10/04/2020

Description: 
   Train and eval models
    
"""
import os
import sys

import pandas as pd

sys.path.append("../")
import conf
from utils import (
    LogManager,

)

# global setting
LogManager.created_filename = os.path.join(conf.LOG_DIR, 'train.log')
# LogManager.log_handle = 'file'
logger = LogManager.get_logger(__name__)


def train_pipeline_ensemble():
    raise NotImplementedError


def train_pipeline_stacking():
    raise NotImplementedError


def train_pipeline_neural():
    raise NotImplementedError


def train(
        fe_filename,
        is_eval
):
    fe_df = pd.read_feather(os.path.join(conf.DATA_DIR, fe_filename))


if __name__ == "__main__":
    params = {
        'fe_filename': 'fe_df.feather',
        'is_eval': True,
    }
    train(**params)
