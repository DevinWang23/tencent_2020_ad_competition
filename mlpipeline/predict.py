# -*- coding: utf-8 -*-
"""
Author:  MengQiu Wang
Email: wangmengqiu@ainnovation.com
Date: 10/04/2020

Description: 
   Using trained model for prediction or classification
    
"""
import sys
import os
import argparse

sys.path.append('../')
import conf
from utils import (
    get_latest_model,
    LogManager,
    timer,
)

# global setting
LogManager.created_filename = os.path.join(conf.LOG_DIR, 'predict.log')
logger = LogManager.get_logger(__name__)


@timer(logger)
def inference_pipeline_ensemble(
        fe_df,
        model_save_path=None,
        model_name=None,
):
    raise NotImplementedError


@timer(logger)
def inference_pipeline_neural(
        fe_df,
        model_save_path=None,
        model_name=None,
):
    raise NotImplementedError


@timer(logger)
def inference_pipeline_stacking(
        fe_df,
        model_save_path=None,
        model_name=None,
):
    raise NotImplementedError


@timer(logger)
def predict(
        model_type,
        model_name,
        is_train=False,
):
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, type=str, help='ensemble, stacking and neural')
    parser.add_argument('--model_name', required=True, type=str, help='lgb')
    parser.add_argument('--is_train', required=True, type=lambda x: (str(x).lower() == 'true'),
                        help='flag for identifying train or predict')
    args = parser.parse_args()

    predict(
            model_type=args.model_type,
            model_name=args.model_name,
            is_train=args.is_train,
    )
