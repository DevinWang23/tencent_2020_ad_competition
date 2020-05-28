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
from datetime import datetime
from importlib import import_module
import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .train import (
    label_map_dict,
    use_label_col
)
sys.path.append('../')
import conf
from utils import (
    get_latest_model,
    LogManager,
    timer,
    load_model,
    log_scale,
    build_iterater,
    build_dataset,
    get_time_diff,
)


# global setting
LogManager.created_filename = os.path.join(conf.LOG_DIR, 'predict.log')
logger = LogManager.get_logger(__name__)

@timer(logger)
def _generate_submission_file(submission_df):
    if use_label_col == ['y']:
        submission_df.loc[:,'predicted_gender'] = submission_df['pred'].apply(lambda x: label_map_dict[x][0])
        submission_df.loc[:,'predicted_age'] = submission_df['pred'].apply(lambda x: label_map_dict[x][1])
    elif use_label_col == ['gender']:
        submission_df.loc[:,'pred'] = submission_df['pred'] + 1
        submission_df.rename(columns={'pred':'predicted_gender'}, inplace=True)
    elif use_label_col == ['age']:
        submission_df.loc[:,'pred'] = submission_df['pred'] + 1
        submission_df.rename(columns={'pred':'predicted_age'}, inplace=True)
    else:
        raise ValueError('input use_label_col out of range')
        
    submission_save_path = os.path.join(conf.SUBMISSION_DIR,'submission_%s_%s.csv'%(use_label_col[0], datetime.now().isoformat()))
    if 'pred' in submission_df.columns:
        submission_df = submission_df.drop(columns=['pred'])
    submission_df.to_csv(submission_save_path,index=False)
    logger.info('submission file has been stored in %s'%submission_save_path)
    return submission_df
            
@timer(logger)
def inference_pipeline_ensemble_and_linear(
        fe_df,
        use_log,
        use_std=True,
        scaler=None,
        model_save_path=None,
        model_name=None,
):
    index_cols, cate_cols, cont_cols, label_cols, features, model = load_model(model_save_path)
    assert cate_cols is not None or cont_cols is not None, 'feature columns are empty' 
    
    if cate_cols and not cont_cols:
            test_features = fe_df[cate_cols]
    elif not cate_cols and cont_cols:
            if use_std:
                fe_df.loc[:,cont_cols] = scaler.transform(fe_df[cont_cols])
            if use_log:
                fe_df,_ = log_scale(cont_cols, sub_fe_df)
            test_features = fe_df[cont_cols]
    else:
            if use_std:
                fe_df.loc[:,cont_cols] = scaler.transform(fe_df[cont_cols])
            if use_log:
                fe_df,_ = log_scale(cont_cols, sub_fe_df)
            test_features = pd.concat([fe_df[cate_cols], 
                                       fe_df[cont_cols]],
                                       axis=1)
            
    X_test = test_features[features]
    pred = [np.argmax(pred_arr) for pred_arr in model.predict(X_test)]
    submission_df = fe_df[['user_id']]
    submission_df.loc[:,'pred'] = pred
    
    return submission_df
    

@timer(logger)
def inference_pipeline_neural(
        fe_df,
        model_module,
        model_params,
):  
    
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True 
    config = model_module.Config(**model_params)
    vocab_dict, test_data,  = build_dataset(
                                           config, 
                                           X_test=fe_df, 
                                           is_train=False,
            )
    config.n_vocab = len(vocab_dict)

    start_time = time.time()
    logger.info("Loading data...")
    test_iter = build_iterater(test_data, config, is_train=False)
    end_time = time.time()
    time_diff = get_time_diff(start_time, end_time)
    logger.info("Time usage:%s" % time_diff)
    
    model = model_module.Model(config).to(config.device)
    state_dict = torch.load(config.save_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(config.device)
    del state_dict
    torch.cuda.empty_cache()
#     model.load_state_dict(torch.load(config.save_path))
    model.eval()
    pred = np.array([], dtype=int)
    with torch.no_grad():
        for seq in test_iter:
            outputs = model(seq)
            sub_pred = torch.max(outputs.data, 1)[1].cpu().numpy()
            pred = np.append(pred, sub_pred)
            
    submission_df = fe_df[['user_id']]
    submission_df.loc[:,'pred'] = pred
    return submission_df

@timer(logger)
def inference_pipeline_stacking(
        fe_df,
        model_save_path=None,
        model_name=None,
):
    raise NotImplementedError


@timer(logger)
def predict(
        test_fe_filename,
        use_log=False,
        use_std=False,
        model_type='neural',
        model_name='lstm',
        model_params={},
        scaler='',
):  
    logger.info('test_fe_filename: %s, use_log: %s, use_std: %s, model_type: %s, model_name: %s'%(
                                                                                            test_fe_filename,
                                                                                            use_log,
                                                                                            use_std,
                                                                                            model_type,
                                                                                            model_name, 
    ))
    test_fe_df = pd.read_feather(os.path.join(conf.DATA_DIR, test_fe_filename))
    if model_type == 'linear' or model_type=='ensemble':
        model_save_path = get_latest_model(conf.TRAINED_MODEL_DIR, '%s.model' % model_name)
        submission_df = inference_pipeline_ensemble_and_linear(
                                                              test_fe_df, 
                                                              use_log,
                                                              use_std,
                                                              scaler,
                                                              model_save_path=model_save_path,
                                                              model_name=model_name
        )
    elif model_type == 'neural':
        model_module = import_module('models.neural_network.' + model_name)
        submission_df = inference_pipeline_neural(
                                                  test_fe_df, 
                                                  model_module,
                                                  model_params
         )
        torch.cuda.empty_cache()
    else:
        raise NotImplementedError('model type %s out of range'%model_type)
    
    submission_df = _generate_submission_file(submission_df)
    
    return  submission_df

if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_type', required=True, type=str, help='ensemble, stacking and neural')
#     parser.add_argument('--model_name', required=True, type=str, help='lgb')
#     parser.add_argument('--is_train', required=True, type=lambda x: (str(x).lower() == 'true'),
#                         help='flag for identifying train or predict')
#     args = parser.parse_args()

#     predict(
#             model_type=args.model_type,
#             model_name=args.model_name,
#             is_train=args.is_train,
#     )
      pass
