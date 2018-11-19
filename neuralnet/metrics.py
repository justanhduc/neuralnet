import numpy as np
import warnings
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from neuralnet.model_zoo import VGG16, VGG19
from neuralnet import utils

__all__ = ['manhattan_distance', 'mean_classification_error', 'mean_squared_error', 'msssim',
           'multinoulli_cross_entropy', 'root_mean_squared_error', 'psnr', 'psnr255', 'pearson_correlation',
           'ssim', 'spearman', 'first_derivative_error', 'huber_loss', 'binary_cross_entropy', 'norm_error',
           'gradient_difference', 'total_variation', 'pulling_away', 'vgg16_loss', 'dog_loss',
           'log_loss', 'gram_vgg19_loss', 'l1_reg', 'l2_reg']


def manhattan_distance(y_pred, y):
    print('Using Manhattan distance loss')
    warnings.warn('This function will be removed soon. Please use NormError instead.', DeprecationWarning)
    if y.ndim != y_pred.ndim:
        raise TypeError('y should have the same shape as y_pred', ('y', y.type, 'y_pred', y_pred.type))
    return T.mean(T.abs_(y_pred - y))


def mean_squared_error(y_pred, y):
    print('Using Euclidean distance loss')
    warnings.warn('This function will be removed soon. Please use NormError instead.', DeprecationWarning)
    if y.ndim != y_pred.ndim:
        raise TypeError('y should have the same shape as y_pred', ('y', y.type, 'y_pred', y_pred.type))
    return T.mean(T.square(y_pred - y))


def norm_error(x, y, p=2):
    print('Using L%d norm loss' % p)
    if y.ndim != x.ndim:
        raise TypeError('y should have the same shape as y_pred', ('y', y.type, 'y_pred', x.type))
    return T.mean(T.abs_(x - y) ** p)


def root_mean_squared_error(x, y):
    print('Using Root Mean Square loss')
    if y.ndim != x.ndim:
        raise TypeError('y should have the same shape as y_pred', ('y', y.type, 'y_pred', x.type))
    return T.sqrt(T.mean(T.sqr(x - y)))


def huber_loss(x, y, thres=1.):
    print('Using Huber loss')
    if y.ndim != x.ndim:
        raise TypeError('y should have the same shape as y_pred', ('y', y.type, 'y_pred', x.type))
    e = T.abs_(x - y)
    larger_than_equal_to = .5 * thres ** 2 + thres * (e - thres)
    less_than = .5 * e**2
    mask = T.cast(e >= thres, theano.config.floatX)
    return T.mean(mask * larger_than_equal_to + (1. - mask) * less_than)


def first_derivative_error(x, y, p=2, depth=3):
    print('Using First derivative loss '),
    if x.ndim != 4 and y.ndim != 4:
        raise TypeError('y and y_pred should have four dimensions')
    kern_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=theano.config.floatX)
    kern_x = utils.make_tensor_kernel_from_numpy((depth, depth), kern_x)

    kern_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=theano.config.floatX)
    kern_y = utils.make_tensor_kernel_from_numpy((depth, depth), kern_y)

    grad_x = T.nnet.conv2d(T.concatenate((x, y), 1), kern_x, border_mode='half')
    grad_y = T.nnet.conv2d(T.concatenate((x, y), 1), kern_y, border_mode='half')
    x_grad = T.sqrt(T.sqr(grad_x[:, :depth]) + T.sqr(grad_y[:, :depth]) + 1e-10)
    y_grad = T.sqrt(T.sqr(grad_x[:, depth:]) + T.sqr(grad_y[:, depth:]) + 1e-10)
    return norm_error(x_grad, y_grad, p)


def gradient_difference(x, y, p):
    print('Using gradient difference loss')
    if y.ndim != x.ndim:
        raise TypeError('y should have the same shape as y_pred', ('y', y.type, 'y_pred', x.type))
    diff_v = T.mean(T.abs_(T.abs_(x[:, :, 1::2, :] - x[:, :, ::2, :]) - T.abs_(y[:, :, 1::2, :] - y[:, :, ::2, :]))**p)
    diff_h = T.mean(T.abs_(T.abs_(x[:, :, :, 1::2] - x[:, :, :, ::2]) - T.abs_(y[:, :, :, 1::2] - y[:, :, :, ::2]))**p)
    return T.mean(diff_h + diff_v)


def total_variation(x, type='aniso', p=2, depth=3):
    print('Using Total variation regularizer')
    assert x.ndim == 4, 'Input must be a tensor image'

    kern_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=theano.config.floatX)
    kern_x = utils.make_tensor_kernel_from_numpy((depth, depth), kern_x)

    kern_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=theano.config.floatX)
    kern_y = utils.make_tensor_kernel_from_numpy((depth, depth), kern_y)
    x_grad_x = T.nnet.conv2d(x, kern_x, border_mode='half')
    x_grad_y = T.nnet.conv2d(x, kern_y, border_mode='half')
    x_grad = T.sqrt(T.sqr(x_grad_x) + T.sqr(x_grad_y) + 1e-10)
    return norm_error(x_grad, T.zeros_like(x_grad, theano.config.floatX), p)


def gram_vgg19_loss(x, y, weight_file, p=2, low_mem=False):
    if x.ndim != 4 and y.ndim != 4:
        raise TypeError('x and y must be image tensors.')
    print('Using Gram matrix VGG19 loss')
    net = VGG19((None, 3, 224, 224), False)
    net.load_params(weight_file)
    mean = T.constant(np.array([123.68, 116.779, 103.939], dtype=theano.config.floatX)[None, :, None, None], 'mean')
    x -= mean
    y -= mean
    if low_mem:
        out_x = net(x)
        out_y = net(y)
    else:
        inputs = T.concatenate((x, y))
        out = net(inputs)
        out_x, out_y = out[:x.shape[0]], out[x.shape[0]:]
    out_x = out_x.flatten(3)
    out_y = out_y.flatten(3)
    gram_x = T.batched_dot(out_x, out_x.dimshuffle(0, 2, 1))
    gram_y = T.batched_dot(out_y, out_y.dimshuffle(0, 2, 1))
    return norm_error(gram_x, gram_y, p)


def vgg16_loss(x, y, weight_file, p=2, low_mem=False):
    if x.ndim != 4 and y.ndim != 4:
        raise TypeError('x and y must be image tensors.')
    print('Using VGG16 loss')
    mean = T.constant(np.array([123.68, 116.779, 103.939], dtype=theano.config.floatX)[None, :, None, None], 'mean')
    net = VGG16((None, 3, 224, 224), False)
    net.load_params(weight_file)
    x -= mean
    y -= mean
    if low_mem:
        x_out = net(x)
        y_out = net(y)
    else:
        out = net(T.concatenate((x, y)))
        x_out = out[:x.shape[0]]
        y_out = out[x.shape[0]:]
    return norm_error(x_out, y_out, p)


def pulling_away(x, y=None):
    if not y:
        if x.ndim != 2:
            raise TypeError('x must be a 2D matrix, got a %dD tensor.' % x.ndim)
        eye = T.eye(x.shape[0], dtype=theano.config.floatX)
        x_hat = x / T.sqrt(T.sum(T.sqr(x), 1, keepdims=True))
        corr = T.dot(x_hat, x_hat.T) ** 2.
        f = 1. / T.cast(4 * x.shape[0] * (x.shape[0]-1), theano.config.floatX) * T.sum(corr * (1. - eye))
        return f
    else:
        if x.ndim != 1:
            raise TypeError('Inputs must be 2 1D vectors.')
        x_hat = x / T.sqrt(T.sum(T.sqr(x)))
        y_hat = y / T.sqrt(T.sum(T.sqr(y)))
        corr = T.dot(x_hat, y_hat) ** 2.
        return corr / 2.


def l2_reg(params):
    print('Using L2 regularization')
    l2 = sum([T.sum(p ** 2) for p in params])
    return l2


def l1_reg(params):
    print('Using L1 regularization')
    l1 = sum([T.sum(T.abs_(p)) for p in params])
    return l1


def spearman(ypred, y, axis=None, eps=1e-8):
    print('Using SROCC metric')

    rng = RandomStreams()
    error = eps * rng.normal(size=[y.shape[0]], dtype=theano.config.floatX)
    y += error  # to break tied rankings

    if axis is None:
        ypred = ypred.flatten()
        y = y.flatten()

    rg_ypred = T.cast(T.argsort(ypred, axis=axis), T.config.floatX)
    rg_y = T.cast(T.argsort(y, axis=axis), T.config.floatX)
    n = y.shape[0]
    numerator = 6 * T.sum(T.square(rg_ypred - rg_y))
    denominator = n * (n**2 - 1)
    return 1. - numerator / denominator


def pearson_correlation(x, y):
    print('Using PLCC metric')
    muy_ypred = T.mean(x)
    muy_y = T.mean(y)
    numerator = T.sum(T.mul(x - muy_ypred, y - muy_y))
    denominator = T.mul(T.sqrt(T.sum(T.square(x - muy_ypred))), T.sqrt(T.sum(T.sqr(y - muy_y)))) + 1e-10
    return numerator / denominator


def multinoulli_cross_entropy(p_y_given_x, y):
    print('Using multinoulli cross entropy')
    cost = T.nnet.categorical_crossentropy(p_y_given_x, y).mean()
    return cost


def binary_cross_entropy(p_y_given_x, y):
    print('Using binary cross entropy')
    if y.ndim != p_y_given_x.ndim:
        raise TypeError('y should have the same shape as p_y_given_x', ('y', y.type, 'p_y_given_x', p_y_given_x.type))
    return T.nnet.binary_crossentropy(p_y_given_x + 1e-7, y).mean()


def mean_classification_error(p_y_given_x, y, binary_threshold=0.5):
    print('Using classification error rate metric')
    if y.dtype.startswith('int'):
        y_pred = T.cast(p_y_given_x >= binary_threshold if p_y_given_x.ndim == 1 else T.argmax(p_y_given_x, 1), y.dtype)
        return T.mean(T.cast(T.neq(y_pred, y), theano.config.floatX))
    else:
        raise NotImplementedError


def dog_loss(x, y, size=21, sigma1=1, sigma2=1.6, p=2, **kwargs):
    print('Using Difference of Gaussians loss')
    depth = kwargs.get('depth', 3)
    diff = utils.difference_of_gaussian(T.concatenate((x, y), 1), 2*depth, size, sigma1, sigma2)
    return norm_error(diff[:, :depth], diff[:, depth:], p)


def log_loss(x, y, size=9, sigma=1., p=2, **kwargs):
    print('Using Laplacian of Gaussian loss')
    depth = kwargs.get('depth', 3)
    kern = utils.laplacian_of_gaussian_kernel(size, sigma)
    kern = utils.make_tensor_kernel_from_numpy((depth, depth), kern)
    d2x = T.nnet.conv2d(x, kern, border_mode='half')
    d2y = T.nnet.conv2d(y, kern, border_mode='half')
    return norm_error(d2x, d2y, p)


def ssim(img1, img2, max_val=1., filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, cs_map=False):
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
    sigma = (T.cast(size, theano.config.floatX) * filter_sigma / filter_size) if filter_size else T.as_tensor_variable(np.float32(1))

    if filter_size:
        window = T.cast(T.reshape(utils.fspecial_gauss(size, sigma), (1, 1, size, size)), theano.config.floatX)
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


def msssim(img1, img2, max_val=1., filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, weights=None):
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
    weights = np.array(weights if weights else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=theano.config.floatX)
    levels = weights.size
    downsample_filter = np.ones((1, 1, 2, 2), dtype=theano.config.floatX) / 4.0
    mssim = []
    mcs = []
    for idx in range(levels):
        _ssim, cs = ssim(img1, img2, max_val=max_val, filter_size=filter_size, filter_sigma=filter_sigma, k1=k1, k2=k2, cs_map=True)
        mssim.append(_ssim)
        mcs.append(cs ** weights[idx])
        filtered = [T.nnet.conv2d(im, downsample_filter, border_mode='half') for im in (img1, img2)]
        img1, img2 = [x[:, :, ::2, ::2] for x in filtered]
    mssim = T.as_tensor_variable(mssim)
    mcs = T.as_tensor_variable(mcs)
    return T.prod(mcs) * (mssim[levels-1] ** weights[levels-1])


def psnr(x, y, mask=None):
    """PSNR for [0,1] images"""
    print('Using PSNR metric for [0, 1] images')
    mask = mask.astype(theano.config.floatX) if mask else None
    return -10 * T.log(T.sum(T.square((y - x)*mask)) / T.sum(mask)) / T.log(10.) if mask \
        else -10 * T.log(T.mean(T.square(y - x))) / T.log(10.)


def psnr255(x, y):
    print('Using PSNR metric for [0, 255] images')
    x = T.round(x)
    y = T.round(y)
    return 20. * T.log(255.) / T.log(10.) - 10. * T.log(T.mean(T.square(y - x))) / T.log(10.)
