"""
Written by Duc
Apr, 2016
Updates on Feb 3, 2017
Updates on Sep 8, 2017
"""
from neuralnet import metrics
from neuralnet import utils
from neuralnet import optimization

from theano import tensor as T
import numpy as np


class Optimization(utils.ConfigParser):
    def __init__(self, config_file, **kwargs):
        super(Optimization, self).__init__(config_file, **kwargs)

        self.cost_function = self.config['optimization']['cost_function']
        self.class_weights = self.config['optimization']['class_weights']
        self.method = self.config['optimization']['method']
        self.learning_rate = self.config['optimization']['learning_rate']
        self.momentum = self.config['optimization']['momentum']
        self.epsilon = self.config['optimization']['epsilon']
        self.gamma = self.config['optimization']['gamma']
        self.rho = self.config['optimization']['rho']
        self.beta1 = self.config['optimization']['beta1']
        self.beta2 = self.config['optimization']['beta2']
        self.nesterov = self.config['optimization']['nesterov']
        self.regularization = self.config['optimization']['regularization']
        self.regularization_type = self.config['optimization']['regularization_type']
        self.regularization_coeff = self.config['optimization']['regularization_coeff']
        self.decrease_factor = np.float32(self.config['optimization']['decrease_factor'])
        self.final_learning_rate = self.config['optimization']['final_learning_rate']
        self.last_iter_to_decrease = self.config['optimization']['last_iter_to_decrease']

    def build_cost(self, y_pred, y, **kwargs):
        if self.cost_function.lower() == 'mse':
            cost = metrics.MeanSquaredError(y_pred, y)
        elif self.cost_function.lower() == 'sigmoid_ce':
            cost = metrics.BinaryCrossEntropy(y_pred, y)
        elif self.cost_function.lower() == 'softmax_ce':
            cost = metrics.MultinoulliCrossEntropy(y_pred, y)
        else:
            raise NameError('Unknown type of cost function')
        if self.regularization:
            params = kwargs.get('params', None)
            assert params is not None, 'A parameter variable must be provided for regularization.'
            cost += self.build_regularization(params)
        return cost

    def build_updates(self, cost, params, **kwargs):
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

        grads = T.grad(cost, params)
        if method.lower() == 'adadelta':
            opt = optimization.AdaDelta(rho, epsilon)
            updates = opt.get_updates(params, grads)
        elif method.lower() == 'rmsprop':
            opt = optimization.RMSprop(learning_rate, self.gamma, self.epsilon)
            updates = opt.get_updates(params, grads)
        elif method.lower() == 'sgdmomentum':
            opt = optimization.SGDMomentum(learning_rate, momentum, nesterov)
            updates = opt.get_updates(params, grads)
        elif method.lower() == 'adagrad':
            opt = optimization.AdaGrad(learning_rate, epsilon)
            updates = opt.get_updates(params, grads)
        elif method.lower() == 'adam':
            opt = optimization.Adam(learning_rate, beta1, beta2)
            updates = opt.get_updates(params, grads)
        elif method.lower() == 'adamax':
            opt = optimization.AdaMax(learning_rate, beta1, beta2)
            updates = opt.get_updates(params, grads)
        elif method.lower() == 'sgd':
            print('Vanilla SGD used')
            opt = optimization.VanillaSGD(learning_rate)
            updates = opt.get_updates(params, grads)
        else:
            print('No valid optimization method chosen. Use Vanilla SGD instead')
            opt = optimization.VanillaSGD(learning_rate)
            updates = opt.get_updates(params, grads)
        return updates

    def build_regularization(self, params, **kwargs):
        try:
            regularization_coeff = kwargs.get('regularization_coeff', self.regularization_coeff)
            regularization_type = kwargs.get('regularization_type', self.regularization_type)
        except AttributeError:
            raise AttributeError('Some attribute does not exist in config or kwargs')

        if regularization_type.lower() == 'l2':
            print('@ L2 regularization')
            # L2-regularization
            L2 = sum([T.sum(p ** 2) for p in params if '_W' in p.name])
            return regularization_coeff * L2
        elif regularization_type.lower() == 'l1':
            print('@ L1 regularization')
            # L1-regularization
            L1 = sum([T.sum(abs(p)) for p in params if '_W' in p.name])
            return regularization_coeff * L1
        else:
            raise NotImplementedError('Regularization should be L1 or L2 only')

    def decrease_learning_rate(self, **kwargs):
        lr = kwargs.get('learning_rate', None)
        iter = kwargs.get('iteration', None)
        if lr is None or iter is None:
            raise ValueError('Learning rate Shared Variable and iteration Variable must be provided')
        utils.decrease_learning_rate(lr, iter, self.learning_rate, self.final_learning_rate, self.last_iter_to_decrease)