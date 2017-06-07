from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import tensor as T
import theano


def ManhattanDistance(y_pred, y):
    return T.mean(T.abs_(y_pred - y))


def GaussianCrossEntropy(y_pred, y):
    if y.ndim != y_pred.ndim:
        raise TypeError('y should have the same shape as y_pred', ('y', y.type, 'y_pred', y_pred.type))
    return T.mean(T.square(y_pred - y))


def RootMeanSquaredError(y_pred, y):
    if y.ndim != y_pred.ndim:
        raise TypeError(
            'y should have the same shape as y_pred',
            ('y', y.type, 'y_pred', y_pred.type)
        )
    return T.sqrt(T.mean(T.sqr(y_pred - y)))


def SpearmanRho(ypred, y, eps=1e-8):
    rng = RandomStreams()
    error = eps * rng.normal(size=[y.shape[0]], dtype=theano.config.floatX)
    y += error  # to break tied rankings
    rg_ypred = T.cast(T.argsort(ypred), T.config.floatX)
    rg_y = T.cast(T.argsort(y), T.config.floatX)
    n = y.shape[0]
    numerator = 6 * T.sum(T.square(rg_ypred - rg_y))
    denominator = n * (n**2 - 1)
    return 1. - numerator / denominator


def PearsonCorrelation(ypred, y):
    muy_ypred = T.mean(ypred)
    muy_y = T.mean(y)
    numerator = T.sum(T.mul(ypred - muy_ypred, y - muy_y))
    denominator = T.mul(T.sqrt(T.sum(T.square(ypred - muy_ypred))), T.sqrt(T.sum(T.sqr(y - muy_y)))) + 1e-10
    return numerator / denominator


def MultinoulliCrossEntropy(p_y_given_x, y):
    xdev = p_y_given_x - p_y_given_x.max(1, keepdims=True)
    lsm = xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))
    cm2 = -lsm[T.arange(y.shape[0]), y]
    return cm2.mean()


def BinaryCrossEntropy(p_y_given_x, y):
    return T.nnet.binary_crossentropy(p_y_given_x, y).mean()


def MeanClassificationErrors(y_pred, y):
    """Return a float representing the number of errors in the minibatch
    over the total number of examples of the minibatch ; zero one
    loss over the size of the minibatch
    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
              correct label
    """
    # check if y has same dimension of y_pred
    if y.ndim != y_pred.ndim:
        raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', y_pred.type))
    # check if y is of the correct datatype
    if y.dtype.startswith('int'):
        # the T.neq operator returns a vector of 0s and 1s, where 1
        # represents a mistake in prediction
        return T.mean(T.cast(T.neq(y_pred, y), theano.config.floatX))
    else:
        raise NotImplementedError
