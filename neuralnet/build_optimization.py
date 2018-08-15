"""
Written by Duc
Apr, 2016
Updates on Feb 3, 2017
Updates on Sep 8, 2017
"""
from neuralnet import metrics
from neuralnet import utils
from neuralnet.optimization import *

from theano import tensor as T
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
        self.gamma = self.config['optimization'].get('gamma', .9)
        self.rho = self.config['optimization'].get('rho', .95)
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
        if not trainable:
            raise ValueError('No trainable parameters are given.')

        try:
            method = kwargs.get('method', self.method)
            learning_rate = kwargs.get('learning_rate', self.learning_rate)
            momentum = kwargs.get('momentum', self.momentum)
            epsilon = kwargs.get('epsilon', self.epsilon)
            rho = kwargs.get('rho', self.rho)
            beta1 = kwargs.get('beta1', self.beta1)
            beta2 = kwargs.get('beta2', self.beta2)
            nesterov = kwargs.get('nesterov', self.nesterov)
        except AttributeError:
            raise AttributeError('Some attribute not found')

        if method.lower() == 'adadelta':
            self.opt, updates = adadelta(cost, trainable, rho, epsilon, return_op=True)
        elif method.lower() == 'rmsprop':
            self.opt, updates = rmsprop(cost, trainable, learning_rate, self.gamma, self.epsilon, return_op=True)
        elif method.lower() == 'sgdmomentum':
            self.opt, updates = sgdmomentum(cost, trainable, learning_rate, momentum, nesterov, return_op=True)
        elif method.lower() == 'adagrad':
            self.opt, updates = adagrad(cost, trainable, learning_rate, epsilon, return_op=True)
        elif method.lower() == 'adam':
            self.opt, updates = adam(cost, trainable, learning_rate, beta1, beta2, epsilon, return_op=True)
        elif method.lower() == 'adamax':
            self.opt, updates = adamax(cost, trainable, learning_rate, beta1, beta2, epsilon, return_op=True)
        elif method.lower() == 'sgd':
            self.opt, updates = sgd(cost, trainable, learning_rate, return_op=True)
        elif method.lower() == 'nadam':
            self.opt, updates = nadam(cost, trainable, learning_rate, beta1, beta2, epsilon, return_op=True)
        elif method.lower() == 'amsgrad':
            self.opt, updates = amsgrad(cost, trainable, learning_rate, beta1, beta2, epsilon,
                                        kwargs.get('decay', lambda x, t: x), return_op=True)
        else:
            print('No valid optimization method chosen. Use Vanilla SGD instead')
            self.opt, updates = sgd(cost, trainable, learning_rate, return_op=True)
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

    def decrease_learning_rate(self, **kwargs):
        lr = kwargs.get('lr', None)
        iter = kwargs.get('iter', None)
        if lr is None or iter is None:
            raise ValueError('Learning rate Shared Variable and iteration Variable must be provided')
        utils.decrease_learning_rate(lr, iter, self.learning_rate, self.final_learning_rate, self.last_iter_to_decrease)
