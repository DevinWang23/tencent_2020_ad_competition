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
import pickle as pkl

sys.path.append('../')
import conf
from utils import (
    LogManager,
    timer,
    check_columns, 
    PAD,
    UNK
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
    sg=1,
    iter_=10,
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
        for i in range(1, emb_dim+1):
            sparse_feat_seq_df.loc[:,'%s_%s'%(method,i)] = sparse_feat_seq_arr[:,i-1]
        sparse_feat_seq_df.drop(columns=[sparse_feat], inplace=True)
    #     logger.info(sparse_feat_seq_df.shape)

        return sparse_feat_seq_df

    elif method == 'matrix_factorization':
        raise NotImplementedError
    elif method == 'w2v':
        logger.info('sparse_feat: %s, method: %s, emd_dim: %s, window: %s, min_count: %s, workers: %s, sg: %s, hs: %s, smaple: %s, negative: %s, alpha: %s, min_alpha: %s, iter: %s'
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
                                                                                  min_alpha,
                                                                                  iter_
                                                                                                                    
    ))
        w2v_save_path = os.path.join(conf.DATA_DIR, '%s_window_%s_dim_%s_sg_%s_hs_%s_iter_%s_embedding.bin'%(
                                                                                                 sparse_feat,
                                                                                                 window,
                                                                                                 emb_dim,
                                                                                                 sg,
                                                                                                 hs,
                                                                                                 iter_,
            
        ))
        if os.path.exists(w2v_save_path):
            logger.info('%s exists, loading...'% w2v_save_path)
            w2v_model = KeyedVectors.load_word2vec_format(w2v_save_path, binary=True) 
            logger.info('%s has been loaded'%  w2v_save_path)
        else:
            sparse_feat_seq_list = sparse_feat_seq_df[sparse_feat].to_list()
            w2v_model = Word2Vec(
                                sparse_feat_seq_list,
                                window=window,
                                size=emb_dim,
                                min_count=min_count,
                                workers=workers,
                                sg=sg,
#                                 sample=sample,
#                                 negative=negative,
                                hs=hs,
#                                 alpha=alpha,
#                                 min_alpha=min_alpha,
                                iter= iter_,
                                )
            w2v_model.wv.save_word2vec_format(w2v_save_path, binary=True)
            
        return w2v_model 
#         unk = np.random.uniform(-0.25, 0.25, emb_dim)
#         sparse_feat_seq_arr = np.asarray([functools.reduce(lambda a,b : a+b, [w2v_model[j] if j in w2v_model else unk for j in i])/len(i) for i in sparse_feat_seq_list])    
    else:
        raise NotImplementedError('input method %s out of method range'%method)
        
@timer(logger)
def _build_vocab(
    preprocessed_df, 
    sparse_feat, 
    window,
    emb_dim,
    sg,
    hs, 
    iter_,
    min_freq=1, 
    max_vocab_size=100000000
):  
    vocab_path = os.path.join(conf.DATA_DIR, '%s_window_%s_dim_%s_sg_%s_hs_%s_iter_%s_vocab.pkl'%(
                                                                                          sparse_feat,
                                                                                          window,
                                                                                          emb_dim,
                                                                                          sg,
                                                                                          hs,
                                                                                          iter_
                                                                                         ))
    if os.path.exists(vocab_path):
        vocab_dict = pkl.load(open(vocab_path,'rb'))
        logger.info('%s has been loaded'%vocab_path)
    else:
        tmp_df = preprocessed_df[[sparse_feat]].astype(str)
        sparse_feat_dict = tmp_df[sparse_feat].value_counts().to_dict()
        vocab_list = sorted([i for i in sparse_feat_dict.items() if i[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                         :max_vocab_size] # filter infrequent words, sorted and keep user-defined vocab size
        vocab_dict = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dict.update({UNK: len(vocab_dict), PAD: len(vocab_dict) + 1})
#         vocab_dict.update({PAD: len(vocab_dict)})
        pkl.dump(vocab_dict, open(vocab_path, 'wb'))
        logger.info('%s has been built'%vocab_path)
        
    return vocab_dict

@timer(logger)
def _build_emb_matrix(
    sparse_feat, 
    w2v_model, 
    vocab_dict, 
    window,
    emb_dim,
    sg,
    hs,
    iter_
):    
    vocab_size = len(vocab_dict)
    emb_matrix = np.zeros((vocab_size, emb_dim), dtype='float32')
    
    for i in vocab_dict.items():
        try:
            emb_matrix[i[1]] = w2v_model[i[0]]
        except KeyError:
            if i[0] == PAD:
                emb_matrix[vocab_size-1] = np.zeros(emb_dim, dtype='float32')  # for padding
            elif i[0] == UNK:
                emb_matrix[vocab_size-2] = np.random.uniform(-0.25, 0.25, emb_dim)  # for unk
            else: 
                logger.info('unknown token %s'%i[0])
                break
    
    emb_file_name = '%s_window_%s_dim_%s_sg_%s_hs_%s_iter_%s_embedding'%(
                                                                 sparse_feat,
                                                                 window,
                                                                 emb_dim,
                                                                 sg,
                                                                 hs,
                                                                 iter_,
        
    )
    emb_save_path = os.path.join(conf.DATA_DIR, emb_file_name)
    np.save(emb_save_path, emb_matrix)
    logger.info('%s has been saved into %s'%(emb_file_name, emb_save_path))
    return emb_matrix


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
#                                 sparse_feat_list=[],
                                sparse_feat='creative_id',
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
                                iter_=10,
                                num_processes=20,
                                is_neural_network=False,
                                is_train=True,
                       ):
    """

    :return:
    """
    logger.info('is_train: %s, is_neural_network: %s'%(is_train, is_neural_network))
    train_preprocessed_df, test_preprocessed_df = _load_preprocessed_data(
                                                                           os.path.join(conf.DATA_DIR,                                                                                                        train_preprocessed_data_filename),
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
    
    if is_neural_network:
        w2v_model = _generate_emb_for_sparse_feat(
                                                        preprocessed_df,
                                                        sparse_feat,
                                                        'w2v',
                                                        emb_dim=emb_dim,
                                                        window=window,                                                                                                                     min_count=min_count,
#                                                         sample=sample,
#                                                         negative=negative,
                                                        hs=hs,
#                                                         alpha=alpha,
#                                                         min_alpha=min_alpha,
                                                        workers=workers,
                                                        sg=sg,
                                                        iter_ = iter_,
        )
        
        # build vocab
        vocab_dict = _build_vocab(
                                    preprocessed_df, 
                                    sparse_feat,
                                    window,
                                    emb_dim,
                                    sg,
                                    hs,  
                                    iter_
        )
        
        # build pretrained emb matrix according to the idx of tokens in vocab_dict
        emb_matrix = _build_emb_matrix(
                            sparse_feat, 
                            w2v_model, 
                            vocab_dict, 
                            window,
                            emb_dim,
                            sg,
                            hs,
                            iter_
        )
        
        # generate neural fe_df
        use_cols = ['user_id'] + [sparse_feat]
        sparse_feat_seq_df = preprocessed_df[use_cols]
#         for sparse_feat in sparse_feat_list:
#             sparse_feat_seq_df.loc[:,sparse_feat] = sparse_feat_seq_df[sparse_feat].astype(str)
        sparse_feat_seq_df.loc[:,sparse_feat] = sparse_feat_seq_df[sparse_feat].astype(str)
        sparse_feat_seq_df = sparse_feat_seq_df.groupby(['user_id'])[sparse_feat].apply(list).reset_index()
        sparse_feat_seq_df.loc[:,sparse_feat] = sparse_feat_seq_df[sparse_feat].apply(lambda x:' '.join(x))  # for save df into                                                                                                                  feather format
    else:
        
        # TODO: add stats feats and user-related feats
        pass
    #     # get creative_id emb
    #     sparse_feat_seq_df = _generate_emb_for_sparse_feat(
    #                                                     preprocessed_df,
    #                                                     'creative_id',
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

        # get ad_id emb
#         sparse_feat_seq_df = _generate_emb_for_sparse_feat(
#                                                         preprocessed_df,
#                                                         'ad_id',
#                                                         emb_method,
#                                                         max_df,
#                                                         min_df,
#                                                         emb_dim,
#                                                         window=window,                                                                                                                 min_count=min_count,
#                                                         sample=sample,
#                                                         negative=negative,
#                                                         hs=hs,
#                                                         alpha=alpha,
#                                                         min_alpha=min_alpha,
#                                                         workers=workers,
#                                                         sg=sg,
#         )

        # TODO: add stats feats and get other embedding sequence
    
    
    
#     fe_df = creative_seq_df
    fe_df = sparse_feat_seq_df.copy()
    del sparse_feat_seq_df
    gc.collect()
    
    # divide feature engineered dataset into train and test
    mask = fe_df['user_id'].isin(test_user_id)
    test_fe_df = fe_df[mask]
    train_fe_df = fe_df[~mask]
    
    # save train_fe_df
    train_fe_df = _get_label(train_fe_df)
    train_fe_save_path = os.path.join(conf.DATA_DIR, '%s_window_%s_dim_%s_sg_%s_hs_%s_iter_%s_%s'%(
                                                                                           sparse_feat,
                                                                                           window,
                                                                                           emb_dim,
                                                                                           sg,
                                                                                           hs,
                                                                                           iter_,
                                                                                           train_fe_save_filename
    )
                                     )
    train_fe_df.reset_index(drop=True, inplace=True)
    train_fe_df.to_feather(train_fe_save_path)
    logger.info('train_fe_df with shape %s has been stored in %s'%(train_fe_df.shape, train_fe_save_path))
    
    # save test fe_df
    test_fe_save_path = os.path.join(conf.DATA_DIR,  '%s_window_%s_dim_%s_sg_%s_hs_%s_iter_%s_%s'%(
                                                                                           sparse_feat,
                                                                                           window,
                                                                                           emb_dim,
                                                                                           sg,
                                                                                           hs,
                                                                                           iter_,
                                                                                           test_fe_save_filename
    )
                                    )
    test_fe_df.reset_index(drop=True, inplace=True)
    test_fe_df.to_feather(test_fe_save_path)
    logger.info('test_fe_df with shape %s has been stored in %s'%(test_fe_df.shape, test_fe_save_path))
    
    return train_fe_df, test_fe_df
    
    

if __name__ == "__main__":
    pass
