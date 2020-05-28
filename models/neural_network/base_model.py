# -*- coding: utf-8 -*-
"""
Author:  MengQiu Wang
Email: wangmengqiu@ainnovation.com
Date: 29/04/2020

Description: 
   Base class for neural network
    
"""
import torch

class BaseNeuralNetwork(object):

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
