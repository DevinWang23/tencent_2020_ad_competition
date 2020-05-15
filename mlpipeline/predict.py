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

import pandas as pd
import numpy as np

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
)


# global setting
LogManager.created_filename = os.path.join(conf.LOG_DIR, 'predict.log')
logger = LogManager.get_logger(__name__)

@timer(logger)
def _generate_submission_file(submission_df):
    if use_label_col == ['y']:
        submission_df['predicted_gender'] = submission_df['pred'].apply(lambda x: label_map_dict[x][0])
        submission_df['predicted_age'] = submission_df['pred'].apply(lambda x: label_map_dict[x][1])
    elif use_label_col == ['gender']:
        submission_df.loc[:,'pred'] = submission_df['pred'] + 1
        submission_df.rename(columns={'pred':'predicted_gender'}, inplace=True)
    elif use_label_col == ['age']:
        submission_df.loc[:,'pred'] = submission_df['pred'] + 1
        submission_df.rename(columns={'pred':'predicted_age'}, inplace=True)
    else:
        raise ValueError('input use_label_col out of range')
    submission_save_path = os.path.join(conf.SUBMISSION_DIR,'submission_%s_%s.csv'%(use_label_col[0],                                                                                                               datetime.now().isoformat()))
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
    submission_df = _generate_submission_file(submission_df)
    
    return submission_df
    

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
        test_fe_filename,
        use_log,
        use_std,
        model_type,
        model_name,
        scaler,
):  
    logger.info('test_fe_filename: %s, use_log: %s, use_std: %s, model_type: %s, model_name: %s'%(
                                                                                            test_fe_filename,
                                                                                            use_log,
                                                                                            use_std,
                                                                                            model_type,
                                                                                            model_name, 
    ))
    test_fe_df = pd.read_feather(os.path.join(conf.DATA_DIR, test_fe_filename))
    model_save_path = get_latest_model(conf.TRAINED_MODEL_DIR, '%s.model' % model_name)
    if model_type == 'linear' or model_type=='ensemble':
        submission_df = inference_pipeline_ensemble_and_linear(
                                                              test_fe_df, 
                                                              use_log,
                                                              use_std,
                                                              scaler,
                                                              model_save_path=model_save_path,
                                                              model_name=model_name)
    
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
