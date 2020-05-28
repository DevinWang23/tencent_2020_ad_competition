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
import time
from datetime import timedelta
from functools import wraps
import joblib
import pickle as pkl
import gc

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from tensorboardX import SummaryWriter

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
UNK, PAD = '<UNK>', '<PAD>' 

def get_time_diff(start_time, end_time):
    """cal the time func consumes"""
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def timer(logger_):
    def real_timer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger_.info('%s开始' % func.__name__)
            result = func(*args, **kwargs)
            end_time = time.time()
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

################################Neural Network Utilities############################### 
class DatasetIterater(object):
    def __init__(self, **kwargs):
        self.batch_size = kwargs['batch_size']
        self.batches = kwargs['batches']
        self.n_batches = len(self.batches) // self.batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(self.batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = kwargs['device']
        self.is_train = kwargs['is_train']
    
    def _to_tensor(self, data):
        x1 = torch.LongTensor([_[0] for _ in data]).to(self.device)
        x2 = torch.LongTensor([_[2] for _ in data]).to(self.device)
        if self.is_train:
            y = torch.LongTensor([_[-1] for _ in data]).to(self.device)
            # pad前的长度(超过pad_size的设为pad_size)
            seq_len1 = torch.LongTensor([_[1] for _ in data]).to(self.device)
            seq_len2 = torch.LongTensor([_[3] for _ in data]).to(self.device)
            return (x1, seq_len1,x2,seq_len2), y
        else:
            seq_len1 = torch.LongTensor([_[1] for _ in data]).to(self.device)
            seq_len2 = torch.LongTensor([_[3] for _ in data]).to(self.device)
            return (x1, seq_len1,x2,seq_len2)
       
    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches
        
@timer(logger)
def build_iterater(
                   dataset,
                   config,
                   is_train,
                   ):
    iter_ = DatasetIterater(batches=dataset, batch_size=config.batch_size, device=config.device, is_train=is_train)
    return iter_

@timer(logger)
def build_dataset(
#     sparse_feat_list,
    config,
    use_label_cols=['y'],
    X_train=[],
    X_valid=[],
    X_test=[],
    is_train=True,
    is_eval=False
    ):
    
    vocab_dict_list = []
    for index, vocab_path in enumerate(config.vocab_paths):
        if os.path.exists(vocab_path):
            vocab_dict = pkl.load(open(vocab_path,'rb'))
            vocab_dict_list += [vocab_dict]
            logger.info('%s has been loaded'%vocab_path)
        else:
            raise FileNotFoundError('build %s first'%vocab_path)
        
#     def __load_dataset(fe_df):
#         for index, sparse_feat in enumerate(config.sparse_feat):
#         fe_df.loc[:,config.sparse_feat] = fe_df[config.sparse_feat].apply(lambda x: x.split(' '))
#         fe_df.loc[:,config.sparse_feat] = fe_df[config.sparse_feat].apply(lambda x: x[:config.max_seq_len])
#         fe_df.loc[:,'len_%s'%config.sparse_feat] = fe_df[config.sparse_feat].apply(len)
# #         logger.info(max(fe_df['len_%s'%config.sparse_feat].values))
#         if config.use_pad:
#             fe_df.loc[:,config.sparse_feat] = fe_df[config.sparse_feat].apply(lambda x: x + ([PAD]*(config.max_seq_len -                                 len(x))) if len(x)< config.max_seq_len else x[:config.max_seq_len])
# #         fe_df.loc[:,'len_%s'%config.sparse_feat] = fe_df[config.sparse_feat].apply(len)
#         fe_df.loc[:,'%s_to_idx'%config.sparse_feat] = fe_df[config.sparse_feat].apply(lambda x :[vocab_dict.get(i,                         vocab_dict.get(UNK)) for i in x])
# #         fe_df.loc[:,'%s_to_idx'%config.sparse_feat] = fe_df[config.sparse_feat].apply(lambda x :[vocab_dict[i] for i in x])
#         ret = list(zip(fe_df['%s_to_idx'%config.sparse_feat].tolist(),fe_df[use_label_cols[0]].tolist(),
#                   fe_df['len_%s'%config.sparse_feat].tolist())) if is_train else                                                                 list(zip(fe_df['%s_to_idx'%config.sparse_feat].tolist(), fe_df['len_%s'%config.sparse_feat].tolist()))
#         return ret
    def __load_dataset(fe_df):
        result_list = []
#         print(config.sparse_feat)
        for i in range(len(config.sparse_feat)):
            fe_df.loc[:,config.sparse_feat[i]] = fe_df[config.sparse_feat[i]].apply(lambda x: x.split(' '))
            fe_df.loc[:,config.sparse_feat[i]] = fe_df[config.sparse_feat[i]].apply(lambda x: x[:config.max_seq_len])
            fe_df.loc[:,'len_%s'%config.sparse_feat[i]] = fe_df[config.sparse_feat[i]].apply(len)
    #         logger.info(max(fe_df['len_%s'%config.sparse_feat].values))
            if config.use_pad:
                fe_df.loc[:,config.sparse_feat[i]] = fe_df[config.sparse_feat[i]].apply(lambda x: x + ([PAD]*                                   (config.max_seq_len - len(x))) if len(x)< config.max_seq_len else x[:config.max_seq_len])
    #         fe_df.loc[:,'len_%s'%config.sparse_feat] = fe_df[config.sparse_feat].apply(len)
            fe_df.loc[:,'%s_to_idx'%config.sparse_feat[i]] = fe_df[config.sparse_feat[i]].apply(lambda x :                                          [vocab_dict_list[i].get(word,vocab_dict_list[i].get(UNK)) for word in x])
#         fe_df.loc[:,'%s_to_idx'%config.sparse_feat] = fe_df[config.sparse_feat].apply(lambda x :[vocab_dict[i] for i in x])
            result_list += [fe_df['%s_to_idx'%config.sparse_feat[i]].tolist(), fe_df['len_%s'%config.sparse_feat[i]].tolist()]
    
        if is_train:   
            result_list += [fe_df[use_label_cols[0]].tolist()]
        
        ret = list(zip(*result_list))
        return ret
    
    if is_train:
        if is_eval:
            train = __load_dataset(X_train)
            valid = __load_dataset(X_valid)
            return vocab_dict_list, train, valid
        else:
            train = __load_dataset(X_train)
            return vocab_dict_list, train
    else:
        test = __load_dataset(X_test)
        return vocab_dict_list, test

@timer(logger)
def neural_train(
    config, 
    model, 
    train_iter, 
    dev_iter=[], 
    is_eval=True,
    ):
    
    start_time = time.time()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    writer = SummaryWriter(log_dir=os.path.join(config.log_path, time.strftime('%m-%d_%H.%M', time.localtime())))
    model.train()
    if is_eval:
        for epoch in range(config.num_epochs):
            logger.info('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
            for i, (trains, labels) in enumerate(train_iter):
                outputs = model(trains)
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels) 
                loss.backward()
                optimizer.step()
                if total_batch % 100 == 0:
                    true = labels.data.cpu()
                    pred = torch.max(outputs.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, pred)
                    dev_acc, dev_loss = _neural_eval(config, model, dev_iter)
                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        torch.save(model.state_dict(), config.save_path)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    end_time = time.time()
                    time_diff = get_time_diff(start_time, end_time)
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  ' \
                          'Val Acc: {4:>6.2%},  Time: {5} {6}'
                    logger.info(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_diff, improve))
#                     msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' \
#                           'Time: {3}'
#                     logger.info(msg.format(total_batch, loss.item(), train_acc, time_diff))
                    writer.add_scalar("loss/train", loss.item(), total_batch)
                    writer.add_scalar("loss/dev", dev_loss, total_batch)
                    writer.add_scalar("acc/train", train_acc, total_batch)
                    writer.add_scalar("acc/dev", dev_acc, total_batch)
                    model.train()
                total_batch += 1
                if total_batch - last_improve > config.required_improvement:
                    logger.info("No optimization for a long time, auto-stopping...")
                    flag = True
                    break

            if flag:
                break
#         torch.save(model.state_dict(), config.save_path)
        writer.close()
        
        return model 
    else:
        for epoch in range(config.num_epochs):
            logger.info('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
            for i, (trains, labels) in enumerate(train_iter):
                outputs = model(trains)
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels) 
                loss.backward()
                optimizer.step()
                if total_batch % 100 == 0:
                    true = labels.data.cpu()
                    pred = torch.max(outputs.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, pred)
                    end_time = time.time()
                    time_diff = get_time_diff(start_time, end_time)
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' \
                          'Time: {3}'
                    logger.info(msg.format(total_batch, loss.item(), train_acc, time_diff))
                    writer.add_scalar("loss/train", loss.item(), total_batch)
                    writer.add_scalar("acc/train", train_acc, total_batch)
                total_batch += 1
        torch.save(model.state_dict(), config.save_path)
        writer.close()
        return model 
        

def _neural_eval(config, model, data_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
        
    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, loss_total / len(data_iter)

@timer(logger)
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
#         print(name)
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass