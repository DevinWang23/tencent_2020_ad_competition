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
from datetime import datetime
import time
from importlib import import_module
import gc

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import ( 
                             accuracy_score,
                             classification_report,
                             confusion_matrix,   
)
import numpy as np
import lightgbm as lgb
import xgboost as xgb 
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("../")
import conf
from utils import (
    LogManager,
    log_scale,
    standard_scale,
    timer,
    check_columns,
    save_model,
    neural_train,
    build_iterater,
    build_dataset,
    get_time_diff,
    init_network
)



# global setting
LogManager.created_filename = os.path.join(conf.LOG_DIR, 'train.log')
# LogManager.log_handle = 'file'
logger = LogManager.get_logger(__name__)

# global varirable
use_label_col = ['y']  # ['age'], ['gender'], ['y']
ensemble_early_stopping_rounds = 10
label_map_dict = {0: (1, 1),
                  1: (1, 2),
                  2: (1, 3),
                  3: (1, 4),
                  4: (1, 5),
                  5: (1, 6),
                  6: (1, 7),
                  7: (1, 8),
                  8: (1, 9),
                  9: (1, 10),
                  10: (2, 1),
                  11: (2, 2),
                  12: (2, 3),
                  13: (2, 4),
                  14: (2, 5),
                  15: (2, 6),
                  16: (2, 7),
                  17: (2, 8),
                  18: (2, 9),
                  19: (2, 10)}
DEFAULT_MISSING_FOLAT = -1.234

@timer(logger)
def _get_cv_folds(
                    X,
                    y,
                    n_splits,
):
    
    skf = StratifiedKFold(
                          n_splits=n_splits,
                          random_state=10,
                          shuffle=False,
    )
    for train_idx, valid_idx in skf.split(X, y[use_label_col]):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        yield X_train, X_valid, y_train, y_valid

def _generate_train_data_from_fe_df(
                                   fe_df,
                                   model_type='linear',
                                   use_log=False
):
    
    index_cols, cate_cols, cont_cols, label_cols = check_columns(fe_df.dtypes.to_dict())
    assert cate_cols is not None or cont_cols is not None, 'feature columns are empty'
    if model_type !='neural' and model_type !='stacking' :
        logger.info('连续性特征数量: %s' % len(cont_cols))
        logger.info('离散性特征数量: %s' % len(cate_cols))
    
    if cate_cols and not cont_cols:
         train_x_df = fe_df[index_cols + cate_cols]
    elif not cate_cols and cont_cols:
        if use_log:
            fe_df = log_scale(
                              fe_df,
                              cont_cols,
                              )
        train_x_df = fe_df[index_cols + cont_cols]
    else:
        if use_log:
            fe_df = log_scale(
                              fe_df,
                              cont_cols,
                              )
        train_x_df = fe_df[index_cols + cont_cols + cate_cols]
        
    train_y_df = fe_df[label_cols]
    return  index_cols, cate_cols, cont_cols, label_cols, train_x_df, train_y_df
            
def _eval(
          model,
          model_type,
          dev_iter=[],
          X_valid=pd.DataFrame(),
          y_valid=pd.DataFrame(),
          index_cols=[],
          cate_cols=[],
          cont_cols=[],
          config={}
):  
    def __eval_metrics(ground_truth, pred, label_name):
        acc = accuracy_score(ground_truth, pred)
        report = classification_report(ground_truth, pred, digits=4)
        confusion = confusion_matrix(ground_truth, pred)
        msg = label_name + '_Val Acc: {0:>6.2%}'
        logger.info(msg.format(acc))
        logger.info(label_name + "_Precision, Recall and F1-Score...")
        logger.info(report)
        logger.info(label_name + "_Confusion Matrix...")
        logger.info(confusion)
        return acc
    
    eval_df = pd.concat([X_valid, y_valid], axis=1)
    if model_type=='neural':
        state_dict = torch.load(config.save_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.to(config.device)
        del state_dict
        torch.cuda.empty_cache()
        model.eval()
        loss_total = 0
        pred = np.array([], dtype=int)
        with torch.no_grad():
            for seq, labels in dev_iter:
                outputs = model(seq)
                loss = F.cross_entropy(outputs, labels)
                loss_total += loss
                sub_predict = torch.max(outputs.data, 1)[1].cpu().numpy()
                pred = np.append(pred, sub_predict)
                
        del outputs, loss, model
        gc.collect()
        torch.cuda.empty_cache()
            
    else:    
        pred = [np.argmax(pred_arr) for pred_arr in model.predict(eval_df[cate_cols + cont_cols])]
    
    if use_label_col == ['y']:
        eval_df.loc[:,'pred'] = pred
        eval_df['pred_gender'] = eval_df['pred'].apply(lambda x :label_map_dict[x][0])
        eval_df['pred_age'] = eval_df['pred'].apply(lambda x :label_map_dict[x][1])
        acc_gender = __eval_metrics(eval_df['gender'],eval_df['pred_gender'], 'gender')
        acc_age = __eval_metrics(eval_df['age'], eval_df['pred_age'],'age')
        return acc_gender, acc_age
    elif use_label_col == ['age']:
        eval_df.loc[:,'pred_age'] = pred
        acc_age = __eval_metrics(eval_df['age'], eval_df['pred_age'],'age')
        return DEFAULT_MISSING_FOLAT,acc_age
    elif use_label_col == ['gender']:
        eval_df.loc[:,'pred_gender'] = pred
        acc_gender = __eval_metrics(eval_df['gender'], eval_df['pred_gender'],'gender')
        return acc_gender, DEFAULT_MISSING_FOLAT, 
    else:
        raise NotImplementedError('input label %s out of range'%use_label_col[0])

def _log_best_round_of_model(model, 
                             evals_result,
                             valid_index,
                             metric
                            ):
        assert hasattr(model, 'best_iteration'), 'just can logger object that has best_iteration attribute'
        n_estimators = model.best_iteration
        logger.info('eval最优轮数: %s, eval最优%s: %s' %(
                                                       n_estimators, 
                                                       metric,            
                                                       evals_result[valid_index][metric][n_estimators - 1]))
        return n_estimators
    
@timer(logger)
def _train_pipeline_ensemble_and_linear(
                                          fe_df,
                                          model_params,
                                          use_cv,
                                          use_log,
                                          use_std,
                                          model_name,
                                          n_splits,
                                          model_save_path,
                                          is_eval
):
        index_cols, cate_cols, cont_cols, label_cols, train_x_df, train_y_df = _generate_train_data_from_fe_df(fe_df, use_log)
        
        feature_name = cate_cols + cont_cols
        
        # transfomer gender value range from [1,2] to [0,1] and age value range from [1,10] to [0,9]
        if use_label_col == ['gender']:
            train_y_df.loc[:,'gender'] = train_y_df['gender'] - 1
        if use_label_col == ['age']:
            train_y_df.loc[:,'age'] = train_y_df['age'] - 1
            
        if is_eval:
            logger.info('eval参数: %s' % (model_params))
            gender_result_list = []
            age_result_list = []
            cv_folds_generator = _get_cv_folds(
                                               train_x_df, 
                                               train_y_df, 
                                               n_splits
            )
            
            for _ in range(n_splits):
                X_train, X_valid, y_train, y_valid = next(cv_folds_generator)
                if use_std:
                    X_train, X_valid = standard_scale(cont_cols, X_train, X_valid)
                if model_name == 'lgb':
                    evals_result = {}
                    train_set = lgb.Dataset(data=X_train[cate_cols + cont_cols], label=y_train[use_label_col])
                    val_set = lgb.Dataset(data=X_valid[cate_cols + cont_cols], label=y_valid[use_label_col],                                                             reference=train_set)
                    model = lgb.train(
                                      params=model_params, 
                                      train_set=train_set, 
                                      valid_sets=[train_set, val_set],
                                      evals_result = evals_result,
                                      early_stopping_rounds=ensemble_early_stopping_rounds
                    )
                    _ = _log_best_round_of_model( 
                                                 model,
                                                 evals_result,
                                                 'valid_1',
                                                 model_params['metric'][0]
                    )
                elif model_name == 'xgb':
                    model = xgb.XGBClassifier(**model_params)
                    val_set = [(X_train[cate_cols + cont_cols], y_train[use_label_col].values.ravel()),(X_valid[cate_cols +                         cont_cols], y_valid[use_label_col].values.ravel())]
                    model.fit(
                        X_train[cate_cols + cont_cols], 
                        y_train[use_label_col].values.ravel(),
                        early_stopping_rounds=ensemble_early_stopping_rounds,
                        eval_set=val_set,
                        verbose=True
                    )
                    logger.info(model.get_xgb_params())
                elif model_name == 'lr':
                    pass
                else:
                    raise NotImplementedError('input model name %s out of range'%model_name)
      
                acc_gender, acc_age = _eval(
                                              model,
                                              X_valid,
                                              y_valid,
                                              index_cols,
                                              cate_cols,
                                              cont_cols
                )
                gender_result_list += [acc_gender]
                age_result_list += [acc_age]
                
            mean_age_score =  np.mean(age_result_list)
            mean_gender_score = np.mean(gender_result_list)
            logger.info('age result list: %s, cv mean: %s'%(age_result_list, mean_age_score))
            logger.info('gender result list: %s, cv mean: %s'% (gender_result_list, mean_gender_score))
            logger.info('total score cv mean: %s'%(mean_age_score + mean_gender_score))
            return _, _
            
        # train on whole data        
        else:
            logger.info('train参数:%s' % model_params)
            if use_std:
                train_x_df, scaler = standard_scale(cont_cols, train_x_df)
            if model_name == 'lgb':
                train_set = lgb.Dataset(data=train_x_df[cate_cols + cont_cols], label=train_y_df[use_label_col])
                model = lgb.train(
                              params=model_params,
                              train_set=train_set,
                              valid_sets=[train_set],
    #                           evals_result=evals_result,
                              early_stopping_rounds=ensemble_early_stopping_rounds
                )

            save_model(model_save_path, 
                       (index_cols, 
                        cate_cols, 
                        cont_cols,
                        label_cols, 
                        feature_name,
                        model)
                             )
            return (model, scaler) if use_std else (model, '')
      
        
def _train_pipeline_stacking():
    raise NotImplementedError

@timer(logger)
def _train_pipeline_neural(
    fe_df,
    model_params,
    model_name,
    model_module,
    is_eval,
    n_splits,
    model_type
):      
    index_cols, cate_cols, cont_cols, label_cols, train_x_df, train_y_df = _generate_train_data_from_fe_df(fe_df,                   model_type=model_type)
    
    # transfomer gender value range from [1,2] to [0,1] and age value range from [1,10] to [0,9]
    if use_label_col == ['gender']:
        train_y_df.loc[:,'gender'] = train_y_df['gender'] - 1
    if use_label_col == ['age']:
        train_y_df.loc[:,'age'] = train_y_df['age'] - 1
    
    # fix all random seed
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True 
    config = model_module.Config(**model_params)
    logger.info('模型参数: %s' % (model_params))
    if is_eval:
        gender_result_list = []
        age_result_list = []
        cv_folds_generator = _get_cv_folds(
                                           train_x_df,
                                           train_y_df,
                                           n_splits
            )
        for _ in range(n_splits):
            X_train, X_valid, y_train, y_valid = next(cv_folds_generator)
            X_train = pd.concat([X_train, y_train],axis=1)
            X_valid = pd.concat([X_valid, y_valid],axis=1)
#             print(X_train.columns)
            vocab_dict_list, train_data, dev_data = build_dataset(
                                                   config, 
                                                   use_label_col, 
                                                   X_train, 
                                                   X_valid,
                                                   is_eval=is_eval,
            )
            config.n_vocab_list = [len(vocab_dict) for vocab_dict in vocab_dict_list]
            
            start_time = time.time()
            logger.info("Loading data...")
            train_iter = build_iterater(train_data, config,is_train=True)
            dev_iter = build_iterater(dev_data, config,is_train=True)
            end_time = time.time()
            time_diff = get_time_diff(start_time, end_time)
            logger.info("Time usage:%s" % time_diff)
            model = model_module.Model(config).to(config.device)
#             model.embedding.cpu()
#             model = model_module.Model(config)
#             model = nn.DataParallel(model)
#             model.to(config.device)
            init_network(model,method=config.init_method, seed=config.seed)
            model = neural_train(config, model, train_iter, dev_iter)
            acc_gender, acc_age  = _eval(
                                          model,
                                          model_type,
                                          dev_iter=dev_iter,
                                          X_valid=X_valid,
                                          config=config
                                    )
            gender_result_list += [acc_gender]
            age_result_list += [acc_age]
            del model
            gc.collect()
            torch.cuda.empty_cache()
        mean_age_score =  np.mean(age_result_list)
        mean_gender_score = np.mean(gender_result_list)
        logger.info('age result list: %s, cv mean: %s'%(age_result_list, mean_age_score))
        logger.info('gender result list: %s, cv mean: %s'% (gender_result_list, mean_gender_score))
        logger.info('total score list: %s, total score cv mean: %s'%(np.asarray(age_result_list) +                                     np.asarray(gender_result_list), mean_age_score + mean_gender_score))
        return _
    else:
        X_train = pd.concat([train_x_df, train_y_df],axis=1) 
        vocab_dict, train_data = build_dataset(
                                               config, 
                                               use_label_col, 
                                               X_train, 
                                               is_eval=is_eval,
        )
        config.n_vocab = len(vocab_dict)
        
        start_time = time.time()
        logger.info("Loading data...")
        train_iter = build_iterater(train_data, config, is_train=True)
        end_time = time.time()
        time_diff = get_time_diff(start_time, end_time)
        logger.info("Time usage:%s" % time_diff)
        model = model_module.Model(config).to(config.device)
        init_network(model, method=config.init_method, seed=config.seed)
        model = neural_train(config, model, train_iter, is_eval=is_eval)
        
        return model
            
@timer(logger)
def train(
            fe_filename,
            is_eval,
            model_type,
            model_name='',
            model_params={},
            use_log=False,
            use_std=True,
            scaler='',
            use_cv=False,
            n_splits=2,
            random_state=1
):
    logger.info("using_fe_df: %s, use_label: %s, is_eval: %s, model_type: %s, model_name: %s, use_log: %s, use_std: %s, use_cv: %s, n_splits: %s"%(
                 fe_filename,
                 use_label_col[0],                                                                                             
                 is_eval,
                 model_type,
                 model_name,
                 use_log,
                 use_std,
                 use_cv,
                 n_splits                                                                                                               
    ))
    fe_df = pd.read_feather(os.path.join(conf.DATA_DIR, fe_filename))
    model_save_path = os.path.join(conf.TRAINED_MODEL_DIR, "%s.model.%s" % (model_name, datetime.now().isoformat()))
    if model_type =='linear' or model_type =='ensemble':
        model, scaler = _train_pipeline_ensemble_and_linear(
                                          fe_df,
                                          model_params,
                                          use_cv,
                                          use_log,
                                          use_std,
                                          model_name,
                                          n_splits,
                                          model_save_path,
                                          is_eval
        )
        
    elif model_type == 'neural':
            model_module = import_module('models.neural_network.' + model_name)
            model = _train_pipeline_neural(
                                            fe_df,
                                            model_params,
                                            model_name,
                                            model_module,
                                            is_eval,
                                            n_splits,
                                            model_type
            )
    elif model_type == 'stacking':
        raise  NotImplementedError
    else:
        raise NotImplementedError('input model type %s out of range'%model_type)
    logger.info("%s模型训练完成!模型保存至:%s" % (model_name, model_save_path)) if not is_eval else logger.info(
                "%s模型训练完成!" % model_name)
    return model if model_type == 'neural' else model, scaler

if __name__ == "__main__":
#     params = {
#         'fe_filename': 'fe_df.feather',
#         'is_eval': True,
#     }
#     train(**params)
    pass 
