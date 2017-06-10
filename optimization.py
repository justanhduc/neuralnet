'''
Written by Duc
April, 2016
Updated Feb 3rd, 2017
Updates Feb 25, 2017: AdaMax, Adam
'''

import theano
from theano import tensor as T
import numpy as np


class VanillaSGD(object):
    def __init__(self, alpha):
        self.alpha = T.cast(T.as_tensor_variable(alpha), theano.config.floatX)
        print('@ VANILLA GRADIENT DESCEND. ETA = %s / ALPHA = %s ' % alpha)

    def deltaXt(self, grad):
        return [self.alpha * grad_i for grad_i in grad]


class AdaDelta(object):
    """
        rho: decay rate (usually >0.9 and <1)
    epsilon: constant (usually 1e-8 ~ 1e-4)
    parameters: all weights of the network
    grad: gradient from T.grad
    Example:
        delta = AdaDelta(0.95, 0.000001, parameters=params)
        grads = T.grad(cost, params)
        updates = [
            (param_i, param_i - grad_i)
            for param_i, grad_i in zip(params, delta.deltaXt(grads))
        ]
    """

    def __init__(self, rho, epsilon, parameters):
        self.Eg2 = [T.zeros(p.get_value().shape, dtype=theano.config.floatX) for p in parameters]
        self.Edelx2 = [T.zeros(p.get_value().shape, dtype=theano.config.floatX) for p in parameters]
        self.prev_del = [T.zeros(p.get_value().shape, dtype=theano.config.floatX) for p in parameters]
        self.rho = T.as_tensor_variable(np.cast[theano.config.floatX](rho))
        self.epsilon = T.as_tensor_variable(np.cast[theano.config.floatX](epsilon))
        print('@ ADADELTA. RHO = %s / EPSILON = %s ' % (self.rho, self.epsilon))

    def deltaXt(self, grad):
        self.Eg2 = [self.rho*Eg2_i + (1. - self.rho)*grad_i**2 for Eg2_i, grad_i in zip(self.Eg2, grad)]
        delta = [T.sqrt(prev_del2_i + self.epsilon)/T.sqrt(grad2_i + self.epsilon)*grad_i
                 for prev_del2_i, grad2_i, grad_i in zip(self.Edelx2, self.Eg2, grad)]
        self.Edelx2 = [self.rho*delx2_i + (1. - self.rho)*delta_i**2 for delx2_i, delta_i in zip(self.Edelx2, delta)]
        self.prev_del = delta
        return delta


class SGDMomentum(object):
    def __init__(self, eta, alpha, parameters):
        self.prev_delta = [T.zeros(p.get_value().shape, dtype=theano.config.floatX) for p in parameters]
        self.eta = T.cast(T.as_tensor_variable(eta), dtype=theano.config.floatX)
        self.alpha = T.cast(T.as_tensor_variable(alpha), dtype=theano.config.floatX)
        print('@ GRADIENT DESCEND MOMENTUM. ETA = %s / ALPHA = %s ' % (eta, alpha))

    def deltaXt(self, grad):
        delta = [self.eta * grad_i + self.alpha * prev_grad_i for grad_i, prev_grad_i in zip(grad, self.prev_delta)]
        self.prev_delta = delta
        return delta


class AdaGrad(object):
    def __init__(self, eta, parameters, epsilon=1e-6):
        self.eta = T.cast(T.as_tensor_variable(eta), theano.config.floatX)
        self.epsilon = T.cast(T.as_tensor_variable(epsilon), theano.config.floatX)
        self.G = [T.zeros(p.get_value().shape, dtype=theano.config.floatX) for p in parameters]
        print('# ADAGRAD. ETA = %s ' % eta)

    def deltaXt(self, grad):
        self.G = [g_i + grad_i**2 for g_i, grad_i in zip(self.G, grad)]
        delta = [self.eta * grad_i / T.sqrt(self.epsilon + g_i) for grad_i, g_i in zip(grad, self.G)]
        return delta


class RMSprop(object):
    def __init__(self, parameters, eta=1e-3, gamma=0.9, epsilon=1e-6):
        self.eta = T.cast(T.as_tensor_variable(eta), theano.config.floatX)
        self.gamma = T.cast(T.as_tensor_variable(gamma), theano.config.floatX)
        self.epsilon = T.cast(T.as_tensor_variable(epsilon), theano.config.floatX)
        self.Eg2 = [T.zeros(p.get_value().shape, dtype=theano.config.floatX) for p in parameters]
        print('# RMSPROP. ETA = %s / GAMMA = %s ' % (eta, gamma))

    def deltaXt(self, grad):
        self.Eg2 = [self.gamma * Eg2_i + (1. - self.gamma) * grad_i**2 for Eg2_i, grad_i in zip(self.Eg2, grad)]
        delta = [self.eta * grad_i / T.sqrt(Eg2_i + self.epsilon) for grad_i, Eg2_i in zip(grad, self.Eg2)]
        return delta


class Adam(object):
    def __init__(self, parameters, alpha=1e-3, beta1=0.9, beta2=0.999):
        self.alpha = T.cast(T.as_tensor_variable(alpha), theano.config.floatX)
        self.beta1 = T.cast(T.as_tensor_variable(beta1), theano.config.floatX)
        self.beta2 = T.cast(T.as_tensor_variable(beta2), theano.config.floatX)
        self.m = [T.zeros(p.get_value().shape, dtype=theano.config.floatX) for p in parameters]
        self.v = [T.zeros(p.get_value().shape, dtype=theano.config.floatX) for p in parameters]
        print('# ADAM. ALPHA = %s / BETA1 = %s / BETA2 = %s' % (alpha, beta1, beta2))

    def deltaXt(self, grad, step):
        t = T.cast(step, theano.config.floatX)
        self.m = [self.beta1 * m_i + (1. - self.beta1) * grad_i for m_i, grad_i in zip(self.m, grad)]
        self.v = [self.beta2 * v_i + (1. - self.beta2) * grad_i**2 for v_i, grad_i in zip(self.v, grad)]
        m_hat = [1. / (1. - self.beta1 ** t) * m_i for m_i in self.m]
        v_hat = [1. / (1. - self.beta2 ** t) * v_i for v_i in self.v]
        delta = [T.cast(self.alpha * m_hat_i / (T.sqrt(v_hat_i) + 1e-8), theano.config.floatX)
                 for m_hat_i, v_hat_i in zip(m_hat, v_hat)]
        return delta


class AdaMax(object):
    def __init__(self, parameters, alpha=2e-3, beta1=0.9, beta2=0.999):
        self.alpha = T.cast(T.as_tensor_variable(alpha), theano.config.floatX)
        self.beta1 = T.cast(T.as_tensor_variable(beta1), theano.config.floatX)
        self.beta2 = T.cast(T.as_tensor_variable(beta2), theano.config.floatX)
        self.m = [T.zeros(p.get_value().shape, dtype=theano.config.floatX) for p in parameters]
        self.u = T.cast(T.as_tensor_variable(0.), theano.config.floatX)
        print('# ADAMAX. ALPHA = %s / BETA1 = %s / BETA2 = %s' % (alpha, beta1, beta2))

    def deltaXt(self, grad, t):
        self.m = [self.beta1 * m_i + (1 - self.beta1) * grad_i for m_i, grad_i in zip(self.m, grad)]
        self.u = T.maximum(self.u, max([T.abs_(grad_i).sum() for grad_i in grad]))
        delta = [T.cast(self.alpha / (1 - self.beta1 ** t) * m_i / self.u, theano.config.floatX)
                 for m_i in self.m]
        return delta
