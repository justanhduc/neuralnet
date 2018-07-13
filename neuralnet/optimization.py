'''
Written by Duc
Apr, 2016
Updates on Feb 3, 2017
Updates on Feb 25, 2017: AdaMax, Adam
Major updates on Sep 8, 2017: All algorithms now return updates in OrderedDict (inspired by and collected from Lasagne)
'''

import theano
from theano import tensor as T
import numpy as np
from collections import OrderedDict
import abc
import sys

sys.setrecursionlimit(10000)
__all__ = ['sgd', 'sgdmomentum', 'adadelta', 'adagrad', 'adam', 'adamax', 'nadam', 'rmsprop', 'amsgrad',
           'anneal_learning_rate']


class Optimizer(metaclass=abc.ABCMeta):
    def __init__(self, eta):
        self.eta = T.cast(eta, theano.config.floatX)
        self.params = []

    @abc.abstractmethod
    def get_updates(self, params, grads):
        pass

    def reset(self):
        pass


class VanillaSGD(Optimizer):
    def __init__(self, eta):
        super(VanillaSGD, self).__init__(eta)
        print(('Using VANILLA GRADIENT DESCEND. ETA = %s ' % eta))

    def get_updates(self, params, grads):
        updates = OrderedDict()
        for param, grad in zip(params, grads):
            updates[param] = param - self.eta * grad
        return updates


class AdaDelta(Optimizer):
    """
        rho: decay rate (usually >0.9 and <1)
    epsilon: constant (usually 1e-8 ~ 1e-4)
    parameters: all weights of the network
    grad: gradient from T.grad
    Example:
        opt = AdaDelta(0.95, 1e-6)
        updates = get_updates(parameter_list, grad_list)
    """

    def __init__(self, rho=.95, epsilon=1e-6):
        super(AdaDelta, self).__init__(0.)
        self.rho = T.as_tensor_variable(np.cast[theano.config.floatX](rho))
        self.epsilon = T.as_tensor_variable(np.cast[theano.config.floatX](epsilon))
        print(('Using ADADELTA. RHO = %s EPSILON = %s ' % (self.rho, self.epsilon)))

    def get_updates(self, params, grads):
        updates = OrderedDict()
        for param, grad in zip(params, grads):
            Eg2_i = theano.shared(np.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX),
                                broadcastable=param.broadcastable)
            delta_i_prev = theano.shared(np.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX),
                                broadcastable=param.broadcastable)
            Edelx2_i = theano.shared(np.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX),
                                broadcastable=param.broadcastable)
            self.params += [Eg2_i, delta_i_prev, Edelx2_i]

            delta_i = T.sqrt(Edelx2_i + self.epsilon) / T.sqrt(Eg2_i + self.epsilon) * grad
            updates[param] = param - delta_i
            updates[delta_i_prev] = delta_i
            updates[Edelx2_i] = self.rho * Edelx2_i + (1. - self.rho) * delta_i**2
            updates[Eg2_i] = self.rho * Eg2_i + (1. - self.rho) * grad**2
        return updates

    def reset(self):
        for param in self.params:
            param.set_value(param.get_value() * np.float32(0))


class SGDMomentum(Optimizer):
    def __init__(self, lr, mom, nesterov=False):
        super(SGDMomentum, self).__init__(lr)
        self.alpha = T.cast(mom, dtype=theano.config.floatX)
        self.nesterov = nesterov
        print(('Using STOCHASTIC GRADIENT DESCENT MOMENTUM. ETA = %s MOMENTUM = %s NESTEROV = %s'
              % (lr, mom, nesterov)))

    def get_updates(self, params, grads):
        updates = OrderedDict()
        for param, grad in zip(params, grads):
            updates[param] = param - self.eta * grad
        if not self.nesterov:
            updates = self.apply_momentum(updates)
        else:
            updates = self.apply_nesterov_momentum(updates)
        return updates

    def apply_momentum(self, updates):
        """Returns a modified update dictionary including momentum

        Generates update expressions of the form:

        * ``velocity := momentum * velocity + updates[param] - param``
        * ``param := param + velocity``

        Parameters
        ----------
        updates : OrderedDict
            A dictionary mapping parameters to update expressions
        params : iterable of shared variables, optional
            The variables to apply momentum to. If omitted, will apply
            momentum to all `updates.keys()`.
        momentum : float or symbolic scalar, optional
            The amount of momentum to apply. Higher momentum results in
            smoothing over more update steps. Defaults to 0.9.

        Returns
        -------
        OrderedDict
            A copy of `updates` with momentum updates for all `params`.

        Notes
        -----
        Higher momentum also results in larger update steps. To counter that,
        you can optionally scale your learning rate by `1 - momentum`.

        See Also
        --------
        momentum : Shortcut applying momentum to SGD updates
        """
        params = list(updates.keys())
        updates = OrderedDict(updates)

        for param in params:
            value = param.get_value(borrow=True)
            velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                     broadcastable=param.broadcastable)
            self.params.append(velocity)

            x = self.alpha * velocity + updates[param]
            updates[velocity] = x - param
            updates[param] = x
        return updates

    def apply_nesterov_momentum(self, updates):
        """Returns a modified update dictionary including Nesterov momentum

        Generates update expressions of the form:

        * ``velocity := momentum * velocity + updates[param] - param``
        * ``param := param + momentum * velocity + updates[param] - param``

        Parameters
        ----------
        delta : OrderedDict
            A dictionary mapping parameters to update expressions
        params : iterable of shared variables, optional
            The variables to apply momentum to. If omitted, will apply
            momentum to all `updates.keys()`.
        momentum : float or symbolic scalar, optional
            The amount of momentum to apply. Higher momentum results in
            smoothing over more update steps. Defaults to 0.9.

        Returns
        -------
        OrderedDict
            A copy of `updates` with momentum updates for all `params`.

        Notes
        -----
        Higher momentum also results in larger update steps. To counter that,
        you can optionally scale your learning rate by `1 - momentum`.

        The classic formulation of Nesterov momentum (or Nesterov accelerated
        gradient) requires the gradient to be evaluated at the predicted next
        position in parameter space. Here, we use the formulation described at
        https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617,
        which allows the gradient to be evaluated at the current parameters.

        See Also
        --------
        nesterov_momentum : Shortcut applying Nesterov momentum to SGD updates
        """
        params = list(updates.keys())
        updates = OrderedDict(updates)

        for param in params:
            value = param.get_value(borrow=True)
            velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                     broadcastable=param.broadcastable)
            self.params.append(velocity)

            x = self.alpha * velocity + updates[param] - param
            updates[velocity] = x
            updates[param] = self.alpha * x + updates[param]
        return updates

    def reset(self):
        for param in self.params:
            param.set_value(param.get_value() * np.float32(0))


class AdaGrad(Optimizer):
    def __init__(self, eta, epsilon=1e-6):
        super(AdaGrad, self).__init__(eta)
        self.epsilon = T.cast(epsilon, theano.config.floatX)
        print(('Using ADAGRAD. ETA = %s ' % eta))

    def get_updates(self, params, grads):
        updates = OrderedDict()
        for param, grad in zip(params, grads):
            grad_prev = theano.shared(np.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX),
                                broadcastable=param.broadcastable)
            self.params.append(grad_prev)

            updates[grad_prev] = grad_prev + grad**2
            updates[param] = self.eta * grad / T.sqrt(self.epsilon + grad_prev)
        return updates

    def reset(self):
        for param in self.params:
            param.set_value(param.get_value() * np.float32(0))


class RMSprop(Optimizer):
    def __init__(self, eta=1e-3, gamma=0.9, epsilon=1e-6):
        super(RMSprop, self).__init__(eta)
        self.gamma = T.cast(gamma, theano.config.floatX)
        self.epsilon = T.cast(epsilon, theano.config.floatX)
        print(('Using RMSPROP. ETA = %s GAMMA = %s ' % (eta, gamma)))

    def get_updates(self, params, grads):
        updates = OrderedDict()
        for param, grad in zip(params, grads):
            grad2_prev = theano.shared(np.zeros(param.get_value(borrow=True).shape, dtype=theano.config.floatX),
                                broadcastable=param.broadcastable)
            self.params.append(grad2_prev)

            updates[grad2_prev] = self.gamma * grad2_prev + (1. - self.gamma) * grad ** 2
            updates[param] = param - self.eta * grad / T.sqrt(grad2_prev + self.epsilon)
        return updates

    def reset(self):
        for param in self.params:
            param.set_value(param.get_value() * np.float32(0))


class Adam(Optimizer):
    def __init__(self, alpha=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super(Adam, self).__init__(alpha)
        self.beta1 = T.cast(beta1, theano.config.floatX)
        self.beta2 = T.cast(beta2, theano.config.floatX)
        self.epsilon = epsilon
        print(('Using ADAM. ETA = %s BETA1 = %s BETA2 = %s' % (alpha, beta1, beta2)))

    def get_updates(self, params, grads):
        updates = OrderedDict()

        t_prev = theano.shared(np.float32(0.), 'time')
        self.params.append(t_prev)

        one = T.constant(1)
        t = t_prev + 1
        a_t = self.eta * T.sqrt(one - self.beta2 ** t) / (one - self.beta1 ** t)
        for param, g_t in zip(params, grads):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), param.name + '_grad_mva', broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), param.name + '_grad_sq_mva', broadcastable=param.broadcastable)
            self.params += [m_prev, v_prev]

            m_t = self.beta1 * m_prev + (one - self.beta1) * g_t
            v_t = self.beta2 * v_prev + (one - self.beta2) * g_t ** 2
            step = a_t * m_t / (T.sqrt(v_t) + self.epsilon)

            updates[m_prev] = m_t
            updates[v_prev] = v_t
            updates[param] = param - step

        updates[t_prev] = t
        return updates

    def reset(self):
        for param in self.params:
            param.set_value(param.get_value() * np.float32(0))


class AdaMax(Optimizer):
    def __init__(self, alpha=2e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super(AdaMax, self).__init__(alpha)
        self.beta1 = T.cast(beta1, theano.config.floatX)
        self.beta2 = T.cast(beta2, theano.config.floatX)
        self.epsilon = T.cast(epsilon, theano.config.floatX)
        print(('Using ADAMAX. ETA = %s BETA1 = %s BETA2 = %s' % (alpha, beta1, beta2)))

    def get_updates(self, params, grads):
        updates = OrderedDict()
        t_prev = theano.shared(np.float32(0.))
        self.params.append(t_prev)

        one = T.constant(1)
        t = t_prev + 1
        a_t = self.eta / (one - self.beta1 ** t)
        for param, g_t in zip(params, grads):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            u_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            self.params += [m_prev, u_prev]

            m_t = self.beta1 * m_prev + (one - self.beta1) * g_t
            u_t = T.maximum(self.beta2 * u_prev, abs(g_t))
            step = a_t * m_t / (u_t + self.epsilon)

            updates[m_prev] = m_t
            updates[u_prev] = u_t
            updates[param] = param - step

        updates[t_prev] = t
        return updates

    def reset(self):
        for param in self.params:
            param.set_value(param.get_value() * np.float32(0))


class NAdam(Optimizer):
    def __init__(self, alpha=1e-3, beta1=.99, beta2=.999, epsilon=1e-8, decay=lambda x, t: x * (1. - .5 * .96 ** (t / 250.))):
        super(NAdam, self).__init__(alpha)
        self.beta1 = T.cast(beta1, 'float32')
        self.beta2 = T.cast(beta2, 'float32')
        self.epsilon = T.cast(epsilon, 'float32')
        self.decay = decay
        print('Using NESTEROV ADAM. ETA = %s BETA1 = %s BETA2 = %s' % (alpha, beta1, beta2))

    def get_updates(self, params, grads):
        updates = OrderedDict()

        beta1_acc = theano.shared(1., 'beta1 accumulation')
        t_prev = theano.shared(0, 'time')
        self.params += [beta1_acc, t_prev]

        t = t_prev + 1
        for param, g_t in zip(params, grads):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, value.dtype), broadcastable=param.broadcastable)
            n_prev = theano.shared(np.zeros(value.shape, value.dtype), broadcastable=param.broadcastable)
            self.params += [m_prev, n_prev]

            beta1_t = self.decay(self.beta1, t)
            beta1_tp1 = self.decay(self.beta1, t+1)
            beta1_acc_t = beta1_acc * beta1_t

            g_hat_t = g_t / (1. - beta1_acc_t)
            m_t = self.beta1 * m_prev + (1 - self.beta1) * g_t
            m_hat_t = m_t / (1 - beta1_acc_t * beta1_tp1)
            n_t = self.beta2 * n_prev + (1 - self.beta2) * g_t ** 2
            n_hat_t = n_t / (1. - self.beta2 ** t)
            m_bar_t = (1 - self.beta1) * g_hat_t + beta1_tp1 * m_hat_t

            updates[param] = param - self.eta * m_bar_t / (T.sqrt(n_hat_t) + self.epsilon)
            updates[beta1_acc] = beta1_acc_t
            updates[m_prev] = m_t
            updates[n_prev] = n_t

        updates[t_prev] = t
        return updates

    def reset(self):
        for param in self.params:
            param.set_value(param.get_value() * np.float32(0))


class AMSGrad(Optimizer):
    def __init__(self, alpha=1e-3, beta1=.9, beta2=.99, epsilon=1e-8, decay=lambda x, t: x):
        super(AMSGrad, self).__init__(alpha)
        self.beta1 = T.cast(beta1, 'float32')
        self.beta2 = T.cast(beta2, 'float32')
        self.epsilon = T.cast(epsilon, 'float32')
        self.decay = decay
        print('Using AMSGRAD. ALPHA = %s BETA1 = %s BETA2 = %s' % (alpha, beta1, beta2))

    def get_updates(self, params, grads):
        updates = OrderedDict()

        t_prev = theano.shared(np.float32(0.), 'time step')
        self.params.append(t_prev)

        t = t_prev + 1.
        eta_t = self.decay(self.eta, t)
        a_t = eta_t * T.sqrt(T.constant(1.) - self.beta2 ** t) / (T.constant(1.) - self.beta1 ** t)
        for param, g_t in zip(params, grads):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, value.dtype), broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, value.dtype), broadcastable=param.broadcastable)
            v_hat_prev = theano.shared(np.zeros(value.shape, value.dtype), broadcastable=param.broadcastable)
            self.params += [m_prev, v_prev, v_hat_prev]

            m_t = self.beta1 * m_prev + (1. - self.beta1) * g_t
            v_t = self.beta2 * v_prev + (1. - self.beta2) * g_t ** 2
            v_hat_t = T.maximum(v_hat_prev, v_t)

            updates[param] = param - a_t * m_t / (T.sqrt(v_hat_t) + self.epsilon)
            updates[m_prev] = m_t
            updates[v_prev] = v_t
            updates[v_hat_prev] = v_hat_t

        updates[t_prev] = t
        return updates

    def reset(self):
        for param in self.params:
            param.set_value(param.get_value() * np.float32(0))


def sgd(cost, params, eta=1e-3):
    grads = T.grad(cost, params)
    sgd_op = VanillaSGD(eta)
    return sgd_op, sgd_op.get_updates(params, grads)


def adadelta(cost, params, rho=.95, epsilon=1e-6):
    grads = T.grad(cost, params)
    adadelta_op = AdaDelta(rho, epsilon)
    return adadelta_op, adadelta_op.get_updates(params, grads)


def adam(cost, params, alpha=1e-3, beta1=.9, beta2=.999, epsilon=1e-8):
    grads = T.grad(cost, params)
    adam_op = Adam(alpha, beta1, beta2, epsilon)
    return adam_op, adam_op.get_updates(params, grads)


def amsgrad(cost, params, alpha=1e-3, beta1=.9, beta2=.999, epsilon=1e-8, decay=lambda x, t: x):
    grads = T.grad(cost, params)
    amsgrad_op = AMSGrad(alpha, beta1, beta2, epsilon, decay)
    return amsgrad_op, amsgrad_op.get_updates(params, grads)


def sgdmomentum(cost, params, lr, mom=.95, nesterov=False):
    grads = T.grad(cost, params)
    sgdmom_op = SGDMomentum(lr, mom, nesterov)
    return sgdmom_op, sgdmom_op.get_updates(params, grads)


def rmsprop(cost, params, eta=1e-3, gamma=.9, epsilon=1e-6):
    grads = T.grad(cost, params)
    rmsprop_op = RMSprop(eta, gamma, epsilon)
    return rmsprop_op, rmsprop_op.get_updates(params, grads)


def adagrad(cost, params, eta, epsilon=1e-6):
    grads = T.grad(cost, params)
    adagrad_op = AdaGrad(eta, epsilon)
    return adagrad_op, adagrad_op.get_updates(params, grads)


def nadam(cost, params, alpha=1e-3, beta1=.9, beta2=.999, epsilon=1e-8, decay=lambda x, t: x):
    grads = T.grad(cost, params)
    nadam_op = NAdam(alpha, beta1, beta2, epsilon, decay)
    return nadam_op, nadam_op.get_updates(params, grads)


def adamax(cost, params, alpha=1e-3, beta1=.9, beta2=.999, epsilon=1e-8):
    grads = T.grad(cost, params)
    adamax_op = AdaMax(alpha, beta1, beta2, epsilon)
    return adamax_op, adamax_op.get_updates(params, grads)


def anneal_learning_rate(lr, t, method='half-life', **kwargs):
    if method not in ('half-life', 'step', 'exponential', 'inverse'):
        raise ValueError('Unknown annealing method.')
    if not isinstance(lr, T.sharedvar.ScalarSharedVariable):
        raise TypeError('lr must be a shared variable, got %s.' % type(lr))

    lr_ = lr.get_value()
    if method == 'half-life':
        num_iters = kwargs.pop('num_iters', None)
        decay = kwargs.pop('decay', .1)
        if num_iters is None:
            raise ValueError('num_iters must be provided.')

        if t > num_iters // 2 or t > 3 * num_iters // 4:
            lr.set_value(np.float32(lr_ * decay))
    elif method == 'step':
        step = kwargs.pop('step', None)
        decay = kwargs.pop('decay', .5)
        if step is None:
            raise ValueError('step must be provided.')

        if t % step == 0:
            lr.set_value(np.float32(lr_ * decay))
    elif method == 'exponential':
        decay = kwargs.pop('decay', 1e-4)
        lr.set_value(np.float32(lr_ * np.exp(-decay * t)))
    else:
        decay = kwargs.pop('decay', .01)
        lr.set_value(np.float32(lr_ * 1. / (1. + decay * t)))
