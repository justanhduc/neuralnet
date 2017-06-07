"""
Written by Duc
April, 2016
Updated Feb 3rd, 2017
"""
from theano import tensor as T

import optimization
import utils
import metrics


class Optimization(object):
    def __init__(self, config_file, **kwargs):
        super(Optimization, self).__init__(config_file, **kwargs)
        self.config = utils.load_configuration(config_file)
        try:
            self.cost_function = self.config['optimization']['cost_function']
            self.class_weights = self.config['optimization']['class_weights']
            self.method = self.config['optimization']['method']
            self.learning_rate = self.config['optimization']['learning_rate']
            self.momentum = self.config['optimization']['momentum']
            self.epsilon = self.config['optimization']['epsilon']
            self.rho = self.config['optimization']['rho']
            self.beta1 = self.config['optimization']['beta1']
            self.beta2 = self.config['optimization']['beta2']
            self.nesterov = self.config['optimization']['nesterov']
            self.regularization = self.config['optimization']['regularization']
            self.regularization_type = self.config['optimization']['regularization_type']
            self.regularization_coeff = self.config['optimization']['regularization_coeff']
            self.decrease_factor = self.config['optimization']['decrease_factor']
            self.final_learning_rate = self.config['optimization']['final_learning_rate']
            self.epochs_check_learn_rate = self.config['optimization']['epochs_check_learn_rate']
        except ValueError:
            raise ValueError('Some config value is invalid')

    def build_cost(self, y_pred, y, **kwargs):
        if self.cost_function.lower() == 'ce_gaussian':
            self.cost = metrics.GaussianCrossEntropy(y_pred, y)
        elif self.cost_function.lower() == 'ce_binary':
            self.cost = metrics.BinaryCrossEntropy(y_pred, y)
        elif self.cost_function.lower() == 'ce_multinoulli':
            self.cost = metrics.MultinoulliCrossEntropy(y_pred, y)
        else:
            raise NameError('Unknown type of cost function')

        if self.regularization:
            self.cost += self.build_regularization(**kwargs)

    def build_optimization(self, cost, **kwargs):
        try:
            model = kwargs.get('model')
            method = kwargs.get('method', self.method)
            learning_rate = kwargs.get('learning_rate', self.learning_rate)
            momentum = kwargs.get('momentum', self.momentum)
            epsilon = kwargs.get('epsilon', self.epsilon)
            rho = kwargs.get('rho', self.rho)
            beta1 = kwargs.get('beta1', self.beta1)
            beta2 = kwargs.get('beta2', self.beta2)
            nesterov = kwargs.get('nesterov', self.nesterov)
            step = kwargs.get('step', None)
        except AttributeError:
            raise AttributeError('Some attribute not found')

        params = []
        for i in xrange(len(model)):
            params = params + model[i].params

        grads = T.grad(cost, params)
        if method.lower() == 'adadelta':
            delta = optimization.AdaDelta(rho, epsilon=epsilon, parameters=params)
            outputs = delta.deltaXt(grads)
            updates = [(param_i, param_i - deltaXt_i) for param_i, deltaXt_i in zip(params, outputs)]
        elif method.lower() == 'rmsprop':
            delta = optimization.RMSprop(eta=learning_rate, parameters=params)
            outputs = delta.deltaXt(grads)
            updates = [(param_i, param_i - deltaXt_i) for param_i, deltaXt_i in zip(params, outputs)]
        elif method.lower() == 'sgdmomentum':
            delta = optimization.SGDMomentum(learning_rate, momentum, parameters=params)
            output = delta.deltaXt(grads)
            updates = [(param_i, param_i - deltaXt_i) for param_i, deltaXt_i in zip(params, output)]
        elif method.lower() == 'adagrad':
            delta = optimization.AdaGrad(learning_rate, parameters=params, epsilon=epsilon)
            outputs = delta.deltaXt(grads)
            updates = [(param_i, param_i - deltaXt_i) for param_i, deltaXt_i in zip(params, outputs)]
        elif method.lower() == 'adam':
            if step is None:
                raise ValueError('Step must be provided for ADAM optimizer')
            delta = optimization.Adam(params, learning_rate, beta1, beta2)
            outputs = delta.deltaXt(grads, step)
            updates = [(param_i, param_i - deltaXt_i) for param_i, deltaXt_i in zip(params, outputs)]
        elif method.lower() == 'adamax':
            if step is None:
                raise ValueError('Step must be provided for ADAMAX optimizer')
            delta = optimization.AdaMax(params, learning_rate, beta1, beta2)
            outputs = delta.deltaXt(grads, step)
            updates = [(param_i, param_i - deltaXt_i) for param_i, deltaXt_i in zip(params, outputs)]
        elif method.lower() == 'sgd':
            print('Vanilla SGD used')
            delta = optimization.VanillaSGD(learning_rate)
            updates = [(param_i, param_i - deltaXt_i) for param_i, deltaXt_i in zip(params, delta.deltaXt(grads))]
        else:
            print('No valid optimization method chosen. Use Vanilla SGD instead')
            delta = optimization.VanillaSGD(learning_rate)
            updates = [(param_i, param_i - deltaXt_i) for param_i, deltaXt_i in zip(params, delta.deltaXt(grads))]
        return updates

    def build_regularization(self, **kwargs):
        try:
            model = kwargs.get('model')
            regularization_coeff = kwargs.get('regularization_coeff', self.regularization_coeff)
            regularization_type = kwargs.get('regularization_type', self.regularization_type)
        except AttributeError:
            raise AttributeError('Some attribute does not exist in config or kwargs')

        if regularization_type.lower() == 'l2':
            print('  # L2 regularization')
            # L2-regularization
            L2 = sum([(model[i].W ** 2).sum() for i in range(len(model))])
            return regularization_coeff * L2
        elif regularization_type.lower() == 'l1':
            print('  # L1 regularization')
            # L1-regularization
            L1 = sum([abs(model[i].W).sum() for i in range(len(model))])
            return regularization_coeff * L1
        else:
            raise NotImplementedError('Regularization should be L1 or L2 only')
