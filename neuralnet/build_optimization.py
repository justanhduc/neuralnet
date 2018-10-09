"""
Written by Duc
Apr, 2016
Updates on Feb 3, 2017
Updates on Sep 8, 2017
"""
from neuralnet import metrics
from neuralnet import utils
from neuralnet.optimization import *

import numpy as np


class Optimization(utils.ConfigParser):
    def __init__(self, config_file, **kwargs):
        super(Optimization, self).__init__(config_file, **kwargs)
        self.cost_function = self.config['optimization'].get('cost_function', None)
        self.class_weights = self.config['optimization'].get('class_weights', None)
        self.method = self.config['optimization'].get('method', 'adam')
        self.learning_rate = self.config['optimization'].get('learning_rate', 1e-3)
        self.momentum = self.config['optimization'].get('momentum', .95)
        self.epsilon = self.config['optimization'].get('epsilon', 1e-8)
        self.beta1 = self.config['optimization'].get('beta1', .9)
        self.beta2 = self.config['optimization'].get('beta2', .99)
        self.nesterov = self.config['optimization'].get('nesterov', False)
        self.reg_type = self.config['optimization'].get('regularization', None)
        self.reg_coeff = self.config['optimization'].get('regularization_coeff', None)
        self.annealing_factor = np.float32(self.config['optimization'].get('annealing_factor', .1))
        self.final_learning_rate = self.config['optimization'].get('final_learning_rate', 1e-3)
        self.last_iter_to_decrease = self.config['optimization'].get('last_iter_to_decrease', 1e-3)
        self.opt = None

    def build_cost(self, y_pred, y, regularizable=None):
        if self.cost_function.lower() == 'mse':
            cost = metrics.norm_error(y_pred, y)
        elif self.cost_function.lower() == 'sigmoid_ce':
            cost = metrics.binary_cross_entropy(y_pred, y)
        elif self.cost_function.lower() == 'softmax_ce':
            cost = metrics.multinoulli_cross_entropy(y_pred, y)
        else:
            raise NameError('Unknown type of cost function')

        if regularizable:
            cost += self.build_regularization(regularizable)
        return cost

    def build_updates(self, cost, trainable, **kwargs):
        method = kwargs.pop('method', self.method)
        if 'lr' not in kwargs.keys():
            kwargs['lr'] = self.learning_rate
        if 'mom' not in kwargs.keys():
            kwargs['mom'] = self.momentum
        if 'epsilon' not in kwargs.keys():
            kwargs['epsilon'] = self.epsilon
        if 'beta1' not in kwargs.keys():
            kwargs['beta1'] = self.beta1
        if 'beta2' not in kwargs.keys():
            kwargs['beta2'] = self.beta2
        if 'nesterov' not in kwargs.keys():
            kwargs['nesterov'] = self.nesterov
        kwargs['return_op'] = True

        self.opt, updates = optimizer.get(method.lower(), sgd)(cost, trainable, **kwargs)
        return updates

    def build_regularization(self, params, **kwargs):
        try:
            reg_coeff = kwargs.get('reg_coeff', self.reg_coeff)
            reg_type = kwargs.get('reg_type', self.reg_type)
        except AttributeError:
            raise AttributeError('Some attribute does not exist in config or kwargs')

        if reg_type.lower() == 'l2':
            return reg_coeff * metrics.l2_reg(params)
        elif reg_type.lower() == 'l1':
            return reg_coeff * metrics.l1_reg(params)
        else:
            raise NotImplementedError('Regularization should be L1 or L2 only')
