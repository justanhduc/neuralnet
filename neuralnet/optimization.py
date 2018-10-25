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

from .model_zoo import Net

sys.setrecursionlimit(10000)
__all__ = ['sgd', 'sgdmomentum', 'adadelta', 'adagrad', 'adam', 'adamax', 'nadam', 'rmsprop', 'amsgrad',
           'anneal_learning_rate', 'optimizer']


class Optimizer(Net, metaclass=abc.ABCMeta):
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
            value = param.get_value()
            Eg2_i = theano.shared(np.zeros_like(value), param.name+'_grad_sqr_mva', broadcastable=param.broadcastable)
            delta_i_prev = theano.shared(np.zeros_like(value), param.name+'_prev_grad', broadcastable=param.broadcastable)
            Edelx2_i = theano.shared(np.zeros_like(value), param.name+'_velo_sqr_mva', broadcastable=param.broadcastable)
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
            velocity = theano.shared(np.zeros_like(value), param.name+'_prev_velo', broadcastable=param.broadcastable)
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
            velocity = theano.shared(np.zeros_like(value), param.name+'_prev_velo', broadcastable=param.broadcastable)
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
            value = param.get_value()
            grad_prev = theano.shared(np.zeros_like(value), param.name+'_grad_sqr_sum', broadcastable=param.broadcastable)
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
            value = param.get_value()
            grad2_prev = theano.shared(np.zeros_like(value), param.name+'_grad_sqr_mva', broadcastable=param.broadcastable)
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
            m_prev = theano.shared(np.zeros_like(value), param.name + '_grad_mva', broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros_like(value), param.name + '_grad_sqr_mva', broadcastable=param.broadcastable)
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
            m_prev = theano.shared(np.zeros_like(value), param.name+'_grad_mva', broadcastable=param.broadcastable)
            u_prev = theano.shared(np.zeros_like(value), param.name+'_abs_grad_mva', broadcastable=param.broadcastable)
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
        self.beta1 = T.cast(beta1, theano.config.floatX)
        self.beta2 = T.cast(beta2, theano.config.floatX)
        self.epsilon = T.cast(epsilon, theano.config.floatX)
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
            m_prev = theano.shared(np.zeros_like(value), param.name+'_grad_mva', broadcastable=param.broadcastable)
            n_prev = theano.shared(np.zeros_like(value), param.name+'_grad_sqr_mva', broadcastable=param.broadcastable)
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
        self.beta1 = T.cast(beta1, theano.config.floatX)
        self.beta2 = T.cast(beta2, theano.config.floatX)
        self.epsilon = T.cast(epsilon, theano.config.floatX)
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
            m_prev = theano.shared(np.zeros_like(value), param.name+'_grad_mva', broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros_like(value), param.name+'_grad_sqr_mva', broadcastable=param.broadcastable)
            v_hat_prev = theano.shared(np.zeros_like(value), param.name+'_grad_sqr_velo', broadcastable=param.broadcastable)
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


def sgd(cost, params, lr=1e-3, clip_by_norm=False, return_op=False, **kwargs):
    grads = T.grad(cost, params)
    if clip_by_norm:
        grads = total_norm_constraint(grads, clip_by_norm, clip_by_norm)
    sgd_op = VanillaSGD(lr)
    return (sgd_op, sgd_op.get_updates(params, grads)) if return_op else sgd_op.get_updates(params, grads)


def adadelta(cost, params, mom=.95, epsilon=1e-6, clip_by_norm=False, return_op=False, **kwargs):
    grads = T.grad(cost, params)
    if clip_by_norm:
        grads = total_norm_constraint(grads, clip_by_norm, clip_by_norm)
    adadelta_op = AdaDelta(mom, epsilon)
    return (adadelta_op, adadelta_op.get_updates(params, grads)) if return_op else adadelta_op.get_updates(params, grads)


def adam(cost, params, lr=1e-3, beta1=.9, beta2=.999, epsilon=1e-8, clip_by_norm=False, return_op=False, **kwargs):
    grads = T.grad(cost, params)
    if clip_by_norm:
        grads = total_norm_constraint(grads, clip_by_norm, clip_by_norm)
    adam_op = Adam(lr, beta1, beta2, epsilon)
    return (adam_op, adam_op.get_updates(params, grads)) if return_op else adam_op.get_updates(params, grads)


def amsgrad(cost, params, lr=1e-3, beta1=.9, beta2=.999, epsilon=1e-8, decay=lambda x, t: x, clip_by_norm=False,
            return_op=False, **kwargs):
    grads = T.grad(cost, params)
    if clip_by_norm:
        grads = total_norm_constraint(grads, clip_by_norm, clip_by_norm)
    amsgrad_op = AMSGrad(lr, beta1, beta2, epsilon, decay)
    return (amsgrad_op, amsgrad_op.get_updates(params, grads)) if return_op else amsgrad_op.get_updates(params, grads)


def sgdmomentum(cost, params, lr, mom=.95, nesterov=False, clip_by_norm=False, return_op=False, **kwargs):
    grads = T.grad(cost, params)
    if clip_by_norm:
        grads = total_norm_constraint(grads, clip_by_norm, clip_by_norm)
    sgdmom_op = SGDMomentum(lr, mom, nesterov)
    return (sgdmom_op, sgdmom_op.get_updates(params, grads)) if return_op else sgdmom_op.get_updates(params, grads)


def rmsprop(cost, params, lr=1e-3, mom=.9, epsilon=1e-6, clip_by_norm=False, return_op=False, **kwargs):
    grads = T.grad(cost, params)
    if clip_by_norm:
        grads = total_norm_constraint(grads, clip_by_norm, clip_by_norm)
    rmsprop_op = RMSprop(lr, mom, epsilon)
    return (rmsprop_op, rmsprop_op.get_updates(params, grads)) if return_op else rmsprop_op.get_updates(params, grads)


def adagrad(cost, params, lr, epsilon=1e-6, clip_by_norm=False, return_op=False, **kwargs):
    grads = T.grad(cost, params)
    if clip_by_norm:
        grads = total_norm_constraint(grads, clip_by_norm, clip_by_norm)
    adagrad_op = AdaGrad(lr, epsilon)
    return (adagrad_op, adagrad_op.get_updates(params, grads)) if return_op else adagrad_op.get_updates(params, grads)


def nadam(cost, params, lr=1e-3, beta1=.9, beta2=.999, epsilon=1e-8, decay=lambda x, t: x, clip_by_norm=False,
          return_op=False, **kwargs):
    grads = T.grad(cost, params)
    if clip_by_norm:
        grads = total_norm_constraint(grads, clip_by_norm, clip_by_norm)
    nadam_op = NAdam(lr, beta1, beta2, epsilon, decay)
    return (nadam_op, nadam_op.get_updates(params, grads)) if return_op else nadam_op.get_updates(params, grads)


def adamax(cost, params, lr=1e-3, beta1=.9, beta2=.999, epsilon=1e-8, clip_by_norm=False, return_op=False, **kwargs):
    grads = T.grad(cost, params)
    if clip_by_norm:
        grads = total_norm_constraint(grads, clip_by_norm, clip_by_norm)
    adamax_op = AdaMax(lr, beta1, beta2, epsilon)
    return (adamax_op, adamax_op.get_updates(params, grads)) if return_op else adamax_op.get_updates(params, grads)


def anneal_learning_rate(lr, t, method='half-life', **kwargs):
    if not isinstance(lr, (T.sharedvar.ScalarSharedVariable, T.sharedvar.TensorSharedVariable)):
        raise TypeError('lr must be a shared variable, got %s.' % type(lr))

    lr_ = lr.get_value()
    if method == 'half-life':
        num_iters = kwargs.pop('num_iters', None)
        decay = kwargs.pop('decay', .1)
        if num_iters is None:
            raise ValueError('num_iters must be provided.')

        cond = T.cast(T.or_(T.eq(t, num_iters // 2), T.eq(t, 3 * num_iters // 4)), theano.config.floatX)
        lr.default_update = lr * decay * cond + (1. - cond) * lr
    elif method == 'step':
        step = kwargs.pop('step', None)
        decay = kwargs.pop('decay', .5)
        if step is None:
            raise ValueError('step must be provided.')

        cond = T.cast(T.eq(T.mod(t, step), 0), theano.config.floatX)
        lr.default_update = lr * decay * cond + (1. - cond) * lr
    elif method == 'exponential':
        decay = kwargs.pop('decay', 1e-4)
        t = T.cast(t, theano.config.floatX)
        lr.default_update = lr_ * T.exp(-decay * t)
    elif method == 'linear':
        num_iters = kwargs.pop('num_iters', None)
        if num_iters is None:
            raise ValueError('num_iters must be provided.')

        t = T.cast(t, theano.config.floatX)
        lr.default_update = lr_ * (1. - t / np.cast[theano.config.floatX](num_iters))
    elif method == 'inverse':
        decay = kwargs.pop('decay', .01)
        t = T.cast(t, theano.config.floatX)
        lr.default_update = lr_ / (1. + decay * t)
    else:
        raise ValueError('Unknown annealing method.')


def norm_constraint(tensor_var, max_norm, norm_axes=None, epsilon=1e-7):
    """Max weight norm constraints and gradient clipping
    This takes a TensorVariable and rescales it so that incoming weight
    norms are below a specified constraint value. Vectors violating the
    constraint are rescaled so that they are within the allowed range.
    Parameters
    ----------
    tensor_var : TensorVariable
        Theano expression for update, gradient, or other quantity.
    max_norm : scalar
        This value sets the maximum allowed value of any norm in
        `tensor_var`.
    norm_axes : sequence (list or tuple)
        The axes over which to compute the norm.  This overrides the
        default norm axes defined for the number of dimensions
        in `tensor_var`. When this is not specified and `tensor_var` is a
        matrix (2D), this is set to `(0,)`. If `tensor_var` is a 3D, 4D or
        5D tensor, it is set to a tuple listing all axes but axis 0. The
        former default is useful for working with dense layers, the latter
        is useful for 1D, 2D and 3D convolutional layers.
        (Optional)
    epsilon : scalar, optional
        Value used to prevent numerical instability when dividing by
        very small or zero norms.
    Returns
    -------
    TensorVariable
        Input `tensor_var` with rescaling applied to weight vectors
        that violate the specified constraints.
    Notes
    -----
    When `norm_axes` is not specified, the axes over which the norm is
    computed depend on the dimensionality of the input variable. If it is
    2D, it is assumed to come from a dense layer, and the norm is computed
    over axis 0. If it is 3D, 4D or 5D, it is assumed to come from a
    convolutional layer and the norm is computed over all trailing axes
    beyond axis 0. For other uses, you should explicitly specify the axes
    over which to compute the norm using `norm_axes`.
    Credits
    _______
    Copied from Lasagne
    """
    ndim = tensor_var.ndim

    if norm_axes is not None:
        sum_over = tuple(norm_axes)
    elif ndim == 2:  # DenseLayer
        sum_over = (0,)
    elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
        sum_over = tuple(range(1, ndim))
    else:
        raise ValueError(
            "Unsupported tensor dimensionality {}."
            "Must specify `norm_axes`".format(ndim)
        )

    dtype = np.dtype(theano.config.floatX).type
    norms = T.sqrt(T.sum(T.sqr(tensor_var), axis=sum_over, keepdims=True))
    target_norms = T.clip(norms, 0, dtype(max_norm))
    constrained_output = \
        (tensor_var * (target_norms / (dtype(epsilon) + norms)))

    return constrained_output


def total_norm_constraint(tensor_vars, max_norm, epsilon=1e-7,
                          return_norm=False):
    """Rescales a list of tensors based on their combined norm
    If the combined norm of the input tensors exceeds the threshold then all
    tensors are rescaled such that the combined norm is equal to the threshold.
    Scaling the norms of the gradients is often used when training recurrent
    neural networks [1]_.
    Parameters
    ----------
    tensor_vars : List of TensorVariables.
        Tensors to be rescaled.
    max_norm : float
        Threshold value for total norm.
    epsilon : scalar, optional
        Value used to prevent numerical instability when dividing by
        very small or zero norms.
    return_norm : bool
        If true the total norm is also returned.
    Returns
    -------
    tensor_vars_scaled : list of TensorVariables
        The scaled tensor variables.
    norm : Theano scalar
        The combined norms of the input variables prior to rescaling,
        only returned if ``return_norms=True``.
    Notes
    -----
    The total norm can be used to monitor training.
    References
    ----------
    .. [1] Sutskever, I., Vinyals, O., & Le, Q. V. (2014): Sequence to sequence
       learning with neural networks. In Advances in Neural Information
       Processing Systems (pp. 3104-3112).
    Credits
    _______
    Copied from Lasagne
    """
    norm = T.sqrt(sum(T.sum(tensor**2) for tensor in tensor_vars))
    dtype = np.dtype(theano.config.floatX).type
    target_norm = T.clip(norm, 0, dtype(max_norm))
    multiplier = target_norm / (dtype(epsilon) + norm)
    tensor_vars_scaled = [step*multiplier for step in tensor_vars]

    if return_norm:
        return tensor_vars_scaled, norm
    else:
        return tensor_vars_scaled


optimizer = {'sgd': sgd, 'sgdmomentum': sgdmomentum, 'adadelta': adadelta, 'adagrad': adagrad, 'adam': adam,
             'adamax': adamax, 'nadam': nadam, 'rmsprop': rmsprop, 'amsgrad': amsgrad}
