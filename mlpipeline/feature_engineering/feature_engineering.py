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
import gc
import functools

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
# from gensim.test.utils import get_tmpfile
import pandas as pd 
import numpy as np

sys.path.append('../')
import conf
from utils import (
    LogManager,
    timer,
    check_columns, 
)


# global setting
LogManager.created_filename = os.path.join(conf.LOG_DIR, 'feature_engineering.log')
logger = LogManager.get_logger(__name__)

@timer(logger)
def _load_preprocessed_data(
    train_preprocessed_data_path,
    test_preprocessed_data_path
):
    
    if os.path.exists(train_preprocessed_data_path) and os.path.exists(test_preprocessed_data_path):
        train_preprocessed_df = pd.read_feather(train_preprocessed_data_path)
        test_preprocessed_df = pd.read_feather(test_preprocessed_data_path)
    else:
        raise FileNotFoundError('run preprocess first')
    
    return train_preprocessed_df, test_preprocessed_df

@timer(logger)
def _generate_emb_for_sparse_feat(
    df,
    sparse_feat,
    method,
    max_df=1.0,
    min_df=1,
    emb_dim=100,
    ngram_range = (1,1),
    window=5,
    min_count=5,
    sample=6e-5,
    negative=0,
    hs=0,
    alpha=0.03,
    min_alpha=0.0007,
    workers=4,
    sg=1
):  
    sparse_feat_seq_df = df[['user_id',sparse_feat]]
    sparse_feat_seq_df[sparse_feat] = sparse_feat_seq_df[sparse_feat].astype(str)
    sparse_feat_seq_df = sparse_feat_seq_df.groupby(['user_id'])[sparse_feat].apply(list).reset_index()
#     logger.info(sparse_feat_seq_df.shape)
    
#     logger.info(sparse_feat_seq_df.shape)
    
    if method == 'tf_idf':
        logger.info('sparse_feat: %s , method: %s, emb_dim: %s, max_df: %s, min_df: %s, ngram_range: %s'%(
                                                                                  sparse_feat,
                                                                                  method,
                                                                                  emb_dim,
                                                                                  max_df,
                                                                                  min_df,
                                                                                  ngram_range,
    ))  
        sparse_feat_seq_df[sparse_feat] = sparse_feat_seq_df[sparse_feat].apply(lambda x:' '.join(x))
        tf = TfidfVectorizer(max_features=emb_dim, max_df=max_df, min_df=min_df, sublinear_tf=True,                                     ngram_range=ngram_range)
        sparse_feat_seq_arr = tf.fit_transform(sparse_feat_seq_df[sparse_feat].to_list()).toarray()
#         logger.info(sparse_feat_seq_arr.shape)
#         csr_df = pd.DataFrame.sparse.from_spmatrix(sparse_feat_seq_tf_csr)
#         csr_df.columns = ['tf_'+ str(col) for col in csr_df.columns]
#         sparse_feat_seq_df = pd.concat([sparse_feat_seq_df[['user_id']], csr_df])
    elif method == 'matrix_factorization':
        raise NotImplementedError
    elif method == 'w2v':
        logger.info('sparse_feat: %s, method: %s, emd_dim: %s, window: %s, min_count: %s, workers: %s, sg: %s, hs: %s, smaple: %s, negative: %s, alpha: %s, min_alpha: %s'
                    %(
                                                                                  sparse_feat,
                                                                                  method,
                                                                                  emb_dim,
                                                                                  window,
                                                                                  min_count,
                                                                                  workers,
                                                                                  sg,
                                                                                  hs,
                                                                                  sample,
                                                                                  negative,
                                                                                  alpha,
                                                                                  min_alpha
                                                                                                                    
    ))
        sparse_feat_seq_list = sparse_feat_seq_df[sparse_feat].to_list()
        if os.path.exists(os.path.join(conf.DATA_DIR, '%s_%s_w2v.bin'%(sparse_feat,emb_dim))):
            logger.info('%s exists, loading...'% os.path.join(conf.DATA_DIR, '%s_%s_w2v.bin'%(sparse_feat,emb_dim)))
            w2v_model = KeyedVectors.load_word2vec_format(os.path.join(conf.DATA_DIR, '%s_w2v.bin'%sparse_feat), binary=True) 
            logger.info('%s has been loaded'% os.path.join(conf.DATA_DIR, '%s_%s_w2v.bin'%(sparse_feat,emb_dim)))
        else:
            w2v_model = Word2Vec(
                                sparse_feat_seq_list,
                                window=window,
                                size=emb_dim,
                                min_count=min_count,
                                workers=workers,
                                sg=sg
                                )
            w2v_model.wv.save_word2vec_format(os.path.join(conf.DATA_DIR, '%s_%s_w2v.bin'%(sparse_feat,emb_dim)), binary=True)
        
        unk = np.random.uniform(-0.25, 0.25, emb_dim)
        sparse_feat_seq_arr = np.asarray([functools.reduce(lambda a,b : a+b, [w2v_model[j] if j in w2v_model.keys() else unk for j in i])/len(i) for i in sparse_feat_seq_list])
         
    else:
        raise NotImplementedError('input method %s out of method range'%method)
        
    for i in range(1, emb_dim+1):
        sparse_feat_seq_df.loc[:,'%s_%s'%(method,i)] = sparse_feat_seq_arr[:,i-1]
    
    sparse_feat_seq_df.drop(columns=[sparse_feat], inplace=True)
#     logger.info(sparse_feat_seq_df.shape)
    
    return sparse_feat_seq_df

@timer(logger)
def _get_label(train_fe_df):
    label_df = pd.read_feather(os.path.join(conf.DATA_DIR, 'label_round_one_df.feather'))
    train_fe_df = train_fe_df.merge(label_df, how='left', on='user_id')
    
    return train_fe_df

@timer(logger)
def feature_engineering_pandas(
                                train_preprocessed_data_filename='train_preprocessed_df.feather',
                                test_preprocessed_data_filename='test_preprocessed_df.feather',
                                train_fe_save_filename='train_fe_df.feather',
                                test_fe_save_filename='test_fe_df.feather',
                                emb_method='w2v',
                                max_df=1.0,
                                min_df=1,
                                emb_dim=100,
                                window=5,
                                min_count=5,
                                sample=6e-5,
                                negative=0,
                                hs=0,
                                alpha=0.03,
                                min_alpha=0.0007,
                                workers=4,
                                sg=0,
                                num_processes=20,
                                is_train=True,
                       ):
    """

    :return:
    """
    
    train_preprocessed_df, test_preprocessed_df = _load_preprocessed_data(
                                                                           os.path.join(conf.DATA_DIR,                                                                                                    train_preprocessed_data_filename),
                                                                           os.path.join(conf.DATA_DIR,
                                                                           test_preprocessed_data_filename)
    )
    
    index_cols, cate_cols, cont_cols, label_cols = check_columns(train_preprocessed_df.dtypes.to_dict())
    train_preprocessed_df.drop(columns=label_cols, inplace=True)
    test_user_id = test_preprocessed_df['user_id'].unique()  # for further dividing data into train and test
    preprocessed_df = pd.concat([train_preprocessed_df[index_cols + cate_cols + cont_cols], 
                                 test_preprocessed_df[index_cols + cate_cols + cont_cols]],
                                 axis=0)
    del train_preprocessed_df, test_preprocessed_df
    gc.collect()
    
#     # get creative_id emb
    creative_seq_df = _generate_emb_for_sparse_feat(
                                                    preprocessed_df,
                                                    'creative_id',
                                                    emb_method,
                                                    max_df,
                                                    min_df,
                                                    emb_dim,
                                                    window=window,
                                                    min_count=min_count,
                                                    sample=sample,
                                                    negative=negative,
                                                    hs=hs,
                                                    alpha=alpha,
                                                    min_alpha=min_alpha,
                                                    workers=workers,
                                                    sg=sg,
    )

    # get ad_id emb
#     ad_seq_df = _generate_emb_for_sparse_feat(
#                                                     preprocessed_df,
#                                                     'ad_id',
#                                                     emb_method,
#                                                     max_df,
#                                                     min_df,
#                                                     emb_dim,
#                                                     window=window,
#                                                     min_count=min_count,
#                                                     sample=sample,
#                                                     negative=negative,
#                                                     hs=hs,
#                                                     alpha=alpha,
#                                                     min_alpha=min_alpha,
#                                                     workers=workers,
#                                                     sg=sg,
#     )
    
    # TODO: add stats feats and get other embedding sequence
    
    
    
    fe_df = creative_seq_df
#     fe_df = ad_seq_df
    mask = fe_df['user_id'].isin(test_user_id)
    test_fe_df = fe_df[mask]
    train_fe_df = fe_df[~mask]
    
    train_fe_df = _get_label(train_fe_df)
    train_fe_save_path = os.path.join(conf.DATA_DIR, train_fe_save_filename)
    train_fe_df.reset_index(drop=True, inplace=True)
    train_fe_df.to_feather(train_fe_save_path)
    logger.info('train_fe_df with shape %s has been stored in %s'%(train_fe_df.shape, train_fe_save_path))
    
    test_fe_save_path = os.path.join(conf.DATA_DIR, test_fe_save_filename)
    test_fe_df.reset_index(drop=True, inplace=True)
    test_fe_df.to_feather(test_fe_save_path)
    logger.info('test_fe_df with shape %s has been stored in %s'%(test_fe_df.shape, test_fe_save_path))
    
    return train_fe_df, test_fe_df
    
    

if __name__ == "__main__":
    pass