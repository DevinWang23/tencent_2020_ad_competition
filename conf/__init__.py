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
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
LIB_DIR = os.path.join(ROOT_DIR, 'lib')
FIGS_DIR = os.path.join(ROOT_DIR, 'figs')
ROUND_ONE_DATA_DIR = os.path.join(DATA_DIR, 'tencent_2019_ad_competition_data/round1_data')
ROUND_TWO_DATA_DIR = os.path.join(DATA_DIR, 'tencent_2019_ad_competition_data/round2_data')
