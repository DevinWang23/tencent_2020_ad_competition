# -*- coding: utf-8 -*-
"""
Author:  MengQiu Wang
Email: wangmengqiu@ainnovation.com
Date: 07/03/2020

Description: 
   General utils 
    
"""
import os
import sys
from time import time
from datetime import timedelta
from functools import wraps
import joblib

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .log_manager import LogManager
sys.path.append('../')
import conf

# global setting
LogManager.created_filename = os.path.join(conf.LOG_DIR, 'utils.log')
logger = LogManager.get_logger(__name__)

# global variables
SELECTED_INDEX = [
    'user_id',
    'time',
    'creative_id',
    'ad_id',
    'product_id',
    'advertiser_id']
SELECTED_LABEL = ['age', 'gender', 'y']


def get_time_diff(start_time, end_time):
    """cal the time func consumes"""
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def timer(logger_):
    def real_timer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time()
            logger_.info('%s开始' % func.__name__)
            result = func(*args, **kwargs)
            end_time = time()
            logger_.info('%s已完成，共用时%s' % (
                func.__name__,
                get_time_diff(start_time, end_time)))
            return result

        return wrapper

    return real_timer


def save_model(path, model):
    joblib.dump(model, path)


def load_model(path):
    model = joblib.load(path)
    return model


def keyword_only(func):
    """
    A decorator that forces keyword arguments in the wrapped method
    and saves actual input keyword arguments in `_input_kwargs`.

    .. note:: Should only be used to wrap a method where first arg is `self`
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if len(args) > 0:
            raise TypeError("Method %s forces keyword arguments." % func.__name__)
        self._input_kwargs = kwargs
        return func(self, **kwargs)

    return wrapper


def overrides(interface_class):
    """
    overrides decorate for readability
    :param interface_class:
    :return:
    """

    def overrider(method):
        assert method.__name__ in dir(interface_class), '%s is not in %s' % (method.__name__, interface_class.__name__)
        return method

    return overrider


def get_latest_model(dir_path, file_prefix=None):
    files = sorted(os.listdir(dir_path))
    if file_prefix is not None:
        files = [x for x in files if x.startswith(file_prefix)]
    return os.path.join(dir_path, files[-1])


@timer(logger)
def correct_column_type_by_value_range(
    df, 
    use_float16=False,
):
    index_cols, cate_cols, cont_cols, label_cols = check_columns(df.dtypes.to_dict())

    def __reduce_cont_cols_mem_by_max_min_value():
        for col in cont_cols:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(df[col].dtypes)[:3] == "int":  # judge col_type by type prefix
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:

                # space and accuracy trade-off
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    __reduce_cont_cols_mem_by_max_min_value()

    ## some user-defined data types
    columns = df.columns
    if 'time' in columns:
        df['time'] = df['time'].astype(np.int8)
    if 'y' in columns:
        df['y'] = df['y'].astype(np.int8)
    if 'ad_id' in columns:
        df['ad_id'] = df['ad_id'].astype(np.int32)
#     if 'product_id' in columns:
#         df['product_id'] = df['product_id'].astype(np.int32)
    if 'advertiser_id' in columns:
        df['advertiser_id'] = df['advertiser_id'].astype(np.int32)
    if 'product_category' in columns:
        df['product_category'] = df['product_category'].astype('category')
    if 'industry' in columns:
        df['industry'] = df['industry'].astype('category')
    if 'age' in columns:
        df['age'] = df['age'].astype(np.int8)
    if 'gender' in columns:
        df['gender'] = df['gender'].astype(np.int8)
    df.sort_values(by='time', inplace=True)
    logger.info('col_types: %s' % df.dtypes)


# @timer(logger)
# def check_category_column(fe_df, cate_cols, num_cates_threshold=5):
#     cate_transform_dict = {}
#     total_samples = fe_df.shape[0]
#     ret_cate_cols = []
#     for cate in cate_cols:
#         if fe_df[[cate]].drop_duplicates().shape[0] >= num_cates_threshold:
#             cate_stat = fe_df.groupby(cate, as_index=False)[['date']].count()
#             cate_stat['date'] = cate_stat['date'].apply(lambda x: round(x / total_samples, 3))
#             select_cates = set(cate_stat[cate_stat['date'] > 0.005][cate])  # 至少占比0.5%的类别特征才会被选择
#             cate_transform_dict[cate] = select_cates
#             ret_cate_cols += [cate]
#
#     return cate_transform_dict, ret_cate_cols

@timer(logger)
def remove_cont_cols_with_small_std(fe_df, cont_cols, threshold=1):
    assert not fe_df.empty and len(cont_cols) > 0, 'fe_df and cont_cols cannot be empty'
    small_std_cols = []
    for col in cont_cols:
        col_std = round(fe_df[col].std(), 2)
        logger.info('%s - %s ' % (col, col_std))
        if col_std <= threshold:
            small_std_cols += [col]
    return small_std_cols


@timer(logger)
def remove_cont_cols_with_unique_value(fe_df, cont_cols, threshold=3):
    assert not fe_df.empty and len(cont_cols) > 0, 'fe_df and cont_cols cannot be empty'
    unique_cols = []
    for col in cont_cols:
        num_unique = len(fe_df[col].unique())
        logger.info('%s - %s ' % (col, num_unique))
        if num_unique <= threshold:
            unique_cols += [col]
    logger.info('drop cols: %s' % unique_cols)
    return unique_cols


@timer(logger)
def check_nan_value(df, threshold=30):
    nan_cols = []
    for col in df.columns:
        num_null = df[col].isnull().sum()
        missing_ratio = round((num_null / df.shape[0]) * 100, 2)
        logger.info("%s - missing_number: %s, missing_ratio:%s%%" % (
                                                                        col, 
                                                                        num_null, 
                                                                        missing_ratio,
                                                                    ))
        if missing_ratio >= threshold:
            nan_cols += [col]
    logger.info('drop cols: %s' % nan_cols)
    return nan_cols


def check_columns(col_dict):
    """Check columns type"""
    index_cols, cate_cols, cont_cols, label_cols = [], [], [], []
    for col in col_dict:
        if col in SELECTED_INDEX:
            index_cols.append(col)
        elif col in SELECTED_LABEL:
            label_cols.append(col)
        # judge cont cols type by its type prefix
        elif str(col_dict[col])[:5] == 'float' or str(col_dict[col])[:3] == 'int' or str(col_dict[col])[:6]=='Sparse':
            cont_cols.append(col)
        else:
            cate_cols.append(col)
    return index_cols, cate_cols, cont_cols, label_cols

@timer(logger)
def log_scale(
              fe_df,
              cont_cols,
             ):
    for cont_col in tqdm(cont_cols):
        fe_df.loc[:, cont_col] = np.log2(fe_df[cont_col] + 1e-8)
    return fe_df

@timer(logger)
def standard_scale(
                   cont_cols,
                   X_train,
                   X_valid=pd.DataFrame()):
    scaler = StandardScaler().fit(X_train[cont_cols])
    X_train.loc[:, cont_cols] = scaler.transform(X_train[cont_cols])
    if not X_valid.empty:
        X_valid.loc[:, cont_cols] = scaler.transform(X_valid[cont_cols])
        return X_train, X_valid
    return X_train, scaler