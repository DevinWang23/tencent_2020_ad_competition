# -*- coding: utf-8 -*-
"""
Author:  MengQiu Wang
Email: wangmengqiu@ainnovation.com
Date: 10/04/2020

Description: 
   Doing feature engineering
    
"""
import sys
import os

sys.path.append('../')
import conf
from utils import (
    LogManager,
    timer,
)


# global setting
LogManager.created_filename = os.path.join(conf.LOG_DIR, 'feature_engineering.log')
logger = LogManager.get_logger(__name__)


@timer(logger)
def feature_engineering(filename='',
                        fe_save_filename='fe_df.feather',
                        is_train=True,):
    """

    :return:
    """


if __name__ == "__main__":
    pass