# -*- coding: utf-8 -*-
"""
Author:  MengQiu Wang
Email: wangmengqiu@ainnovation.com
Date: 29/04/2020

Description: 
   Base class for ensemble
    
"""
from abc import abstractmethod


class BaseEnsemble(object):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict_prob(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def generate_kfolds(self,):
        raise NotImplementedError

    def cross_validation(self,):
        raise NotImplementedError

    def random_search(self,):
        raise NotImplementedError

    def grid_search(self,):
        raise NotImplementedError

    def bayes_optimization(self,):
        raise NotImplementedError

    def train_valid_split(self,):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError
