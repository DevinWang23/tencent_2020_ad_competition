# -*- coding: utf-8 -*-
"""
Author:  MengQiu Wang
Email: wangmengqiu@ainnovation.com
Date: 10/04/2020

Description: 
   Global configurations

"""
import os


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
LOG_DIR = os.path.join(ROOT_DIR, 'log')
# DATA_DIR = os.path.join(ROOT_DIR, 'sample_data')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
ROUND_ONE_TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train_preliminary')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')
SUBMISSION_DIR = os.path.join(ROOT_DIR, 'submission')
LIB_DIR = os.path.join(ROOT_DIR, 'lib')
FIGS_DIR = os.path.join(ROOT_DIR, 'figs')
TRAINED_MODEL_DIR = os.path.join(ROOT_DIR, 'trained_models')

