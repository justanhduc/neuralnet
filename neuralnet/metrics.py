import numpy as np
import warnings
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def ManhattanDistance(y_pred, y):
    print('Using Manhattan distance loss')
    warnings.warn('This function will be removed soon. Please use NormError instead.', DeprecationWarning)
    if y.ndim != y_pred.ndim:
        raise TypeError('y should have the same shape as y_pred', ('y', y.type, 'y_pred', y_pred.type))
    return T.mean(T.abs_(y_pred - y))


def MeanSquaredError(y_pred, y):
    print('Using Euclidean distance loss')
    warnings.warn('This function will be removed soon. Please use NormError instead.', DeprecationWarning)
    if y.ndim != y_pred.ndim:
        raise TypeError('y should have the same shape as y_pred', ('y', y.type, 'y_pred', y_pred.type))
    return T.mean(T.square(y_pred - y))


def NormError(x, y, p=2):
    print('Using L%d norm loss' % p)
    if y.ndim != x.ndim:
        raise TypeError('y should have the same shape as y_pred', ('y', y.type, 'y_pred', x.type))
    return T.mean(T.abs_(x - y) ** p)


def RootMeanSquaredError(x, y):
    print('Using Root Mean Square loss')
    if y.ndim != x.ndim:
        raise TypeError('y should have the same shape as y_pred', ('y', y.type, 'y_pred', x.type))
    return T.sqrt(T.mean(T.sqr(x - y)))


def HuberLoss(x, y, thres=1.):
    print('Using Huber loss')
    if y.ndim != x.ndim:
        raise TypeError('y should have the same shape as y_pred', ('y', y.type, 'y_pred', x.type))
    e = T.abs_(x - y)
    larger_than_equal_to = .5 * thres ** 2 + thres * (e - thres)
    less_than = .5 * e**2
    mask = T.cast(e >= thres, 'float32')
    return T.sum(mask * larger_than_equal_to + (1. - mask) * less_than)


def FirstDerivativeError(x, y, p=2):
    print('Using First derivative loss '),
    if x.ndim != 4 and y.ndim != 4:
        raise TypeError('y and y_pred should have four dimensions')
    kern_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype='float32')
    kern_x = T.tile(kern_x, (y.shape[1], y.shape[1], 1, 1))

    kern_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
    kern_y = T.tile(kern_y, (y.shape[1], y.shape[1], 1, 1))

    x_grad_x = T.nnet.conv2d(x, kern_x, border_mode='half')
    x_grad_y = T.nnet.conv2d(x, kern_y, border_mode='half')
    x_grad = T.sqrt(T.sqr(x_grad_x) + T.sqr(x_grad_y) + 1e-10)

    y_grad_x = T.nnet.conv2d(y, kern_x, border_mode='half')
    y_grad_y = T.nnet.conv2d(y, kern_y, border_mode='half')
    y_grad = T.sqrt(T.sqr(y_grad_x) + T.sqr(y_grad_y) + 1e-10)
    return NormError(x_grad, y_grad, p)


def GradientDifference(x, y, p):
    print('Using gradient difference loss')
    if y.ndim != x.ndim:
        raise TypeError('y should have the same shape as y_pred', ('y', y.type, 'y_pred', x.type))
    diff_v = T.mean(T.abs_(T.abs_(x[:, :, 1::2, :] - x[:, :, ::2, :]) - T.abs_(y[:, :, 1::2, :] - y[:, :, ::2, :]))**p)
    diff_h = T.mean(T.abs_(T.abs_(x[:, :, :, 1::2] - x[:, :, :, ::2]) - T.abs_(y[:, :, :, 1::2] - y[:, :, :, ::2]))**p)
    return T.mean(diff_h + diff_v)


def TotalVariation(x, type='aniso'):
    print('Using Total variation regularizer')
    assert x.ndim == 4, 'Input must be a tensor image'
    if type == 'aniso':
        del_v = T.sum(T.abs_(x[:, :, 1:, :] - x[:, :, :-1, :]))
        del_h = T.sum(T.abs_(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return del_h + del_v
    elif type == 'iso':
        del_v = T.sqr(x[:, :, 1:, :] - x[:, :, :-1, :])
        del_h = T.sqr(x[:, :, :, 1:] - x[:, :, :, :-1])
        return T.sum(T.sqrt(del_v + del_h))


def KLD(y, y_pred):
    y = T.clip(y, 1e-8, )



def SpearmanRho(ypred, y, eps=1e-8):
    print('Using SROCC metric')
    rng = RandomStreams()
    error = eps * rng.normal(size=[y.shape[0]], dtype=theano.config.floatX)
    y += error  # to break tied rankings
    rg_ypred = T.cast(T.argsort(ypred), T.config.floatX)
    rg_y = T.cast(T.argsort(y), T.config.floatX)
    n = y.shape[0]
    numerator = 6 * T.sum(T.square(rg_ypred - rg_y))
    denominator = n * (n**2 - 1)
    return 1. - numerator / denominator


def PearsonCorrelation(x, y):
    print('Using PLCC metric')
    muy_ypred = T.mean(x)
    muy_y = T.mean(y)
    numerator = T.sum(T.mul(x - muy_ypred, y - muy_y))
    denominator = T.mul(T.sqrt(T.sum(T.square(x - muy_ypred))), T.sqrt(T.sum(T.sqr(y - muy_y)))) + 1e-10
    return numerator / denominator


def MultinoulliCrossEntropy(p_y_given_x, y):
    print('Using multinoulli cross entropy')
    # xdev = p_y_given_x - p_y_given_x.max(1, keepdims=True)
    # lsm = xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))
    # cm2 = -lsm[T.arange(y.shape[0]), y]
    # cost = cm2.mean()
    cost = T.nnet.categorical_crossentropy(p_y_given_x, y).mean()
    return cost


def BinaryCrossEntropy(p_y_given_x, y):
    print('Using binary cross entropy')
    if y.ndim != p_y_given_x.ndim:
        raise TypeError('y should have the same shape as p_y_given_x', ('y', y.type, 'p_y_given_x', p_y_given_x.type))
    return T.nnet.binary_crossentropy(p_y_given_x + 1e-7, y).mean()


def MeanClassificationErrors(p_y_given_x, y, binary_threshold=0.5):
    """Return a float representing the number of errors in the minibatch
    over the total number of examples of the minibatch ; zero one
    loss over the size of the minibatch
    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
              correct label
    """
    # check if y has same dimension of y_pred
    # if y.ndim != p_y_given_x.ndim:
    #     raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', p_y_given_x.type))
    # check if y is of the correct datatype
    print('Using classification error rate metric')
    if y.dtype.startswith('int'):
        # the T.neq operator returns a vector of 0s and 1s, where 1
        # represents a mistake in prediction
        y_pred = T.cast(p_y_given_x >= binary_threshold if p_y_given_x.ndim == 1 else T.argmax(p_y_given_x, 1), y.dtype)
        return T.mean(T.cast(T.neq(y_pred, y), theano.config.floatX))
    else:
        raise NotImplementedError


def gaussian2(size, sigma):
    """Returns a normalized circularly symmetric 2D gauss kernel array

    f(x,y) = A.e^{-(x^2/2*sigma^2 + y^2/2*sigma^2)} where

    A = 1/(2*pi*sigma^2)

    as define by Wolfram Mathworld
    http://mathworld.wolfram.com/GaussianFunction.html
    """
    A = 1. / (2.0 * np.pi * sigma ** 2)
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = A * np.exp(-((x ** 2 / (2.0 * sigma ** 2)) + (y ** 2 / (2.0 * sigma ** 2))))
    return np.float32(g)


def _fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = T.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = T.exp(-((T.cast(x, 'float32') ** 2 + T.cast(y, 'float32') ** 2) / (2.0 * sigma ** 2)))
    return g / T.sum(g)


def SSIM(img1, img2, max_val=1., filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, cs_map=False):
    """Return the Structural Similarity Map between `img1` and `img2`.
    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
    Returns:
    Pair containing the mean SSIM and contrast sensitivity between `img1` and
    `img2`.
    Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
    """
    print('Using SSIM metric')
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d', img1.ndim)

    img1 = T.cast(img1, theano.config.floatX)
    img2 = T.cast(img2, theano.config.floatX)
    _, _, height, width = T.shape(img1)

    # Filter size can't be larger than height or width of images.
    size = T.min((filter_size, height, width))

    # Scale down sigma if a smaller filter size is used.
    sigma = (T.cast(size, 'float32') * filter_sigma / filter_size) if filter_size else T.as_tensor_variable(np.float32(1))

    if filter_size:
        window = T.cast(T.reshape(_fspecial_gauss(size, sigma), (1, 1, size, size)), theano.config.floatX)
        mu1 = T.nnet.conv2d(img1, window, border_mode='valid')
        mu2 = T.nnet.conv2d(img2, window, border_mode='valid')
        sigma11 = T.nnet.conv2d(img1 * img1, window, border_mode='valid')
        sigma22 = T.nnet.conv2d(img2 * img2, window, border_mode='valid')
        sigma12 = T.nnet.conv2d(img1 * img2, window, border_mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = T.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2 + 1e-10)))
    output = ssim if not cs_map else (ssim, T.mean(v1 / v2))
    return output


def MS_SSIM(img1, img2, max_val=1., filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, weights=None):
    """Return the MS-SSIM score between `img1` and `img2`.
    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
    weights: List of weights for each level; if none, use five levels and the
      weights from the original paper.
    Returns:
    MS-SSIM score between `img1` and `img2`.
    Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, depth, height, width].
    """
    print('Using MSSSIM metric')
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d', img1.ndim)

    # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
    weights = np.array(weights if weights else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype='float32')
    levels = weights.size
    downsample_filter = np.ones((1, 1, 2, 2), dtype=theano.config.floatX) / 4.0
    mssim = []
    mcs = []
    for idx in range(levels):
        ssim, cs = SSIM(img1, img2, max_val=max_val, filter_size=filter_size, filter_sigma=filter_sigma, k1=k1, k2=k2, cs_map=True)
        mssim.append(ssim)
        mcs.append(cs ** weights[idx])
        filtered = [T.nnet.conv2d(im, downsample_filter, border_mode='half') for im in (img1, img2)]
        img1, img2 = [x[:, :, ::2, ::2] for x in filtered]
    mssim = T.as_tensor_variable(mssim)
    mcs = T.as_tensor_variable(mcs)
    return T.prod(mcs) * (mssim[levels-1] ** weights[levels-1])


def PSNR(x, y):
    """PSNR for [0,1] images"""
    print('Using PSNR metric for [0, 1] images')
    return -10 * T.log(T.mean(T.square(y - x))) / T.log(10.)


def PSNR255(x, y):
    print('Using PSNR metric for [0, 255] images')
    x = T.round(x)
    y = T.round(y)
    return 20. * T.log(255.) / T.log(10.) - 10. * T.log(T.mean(T.square(y - x))) / T.log(10.)
