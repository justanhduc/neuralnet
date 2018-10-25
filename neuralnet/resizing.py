import theano
import numpy as np
from theano import tensor as T
from theano.tensor.nnet import conv2d as conv
from theano.gpuarray.dnn import dnn_pool as pool
from functools import partial
import numbers

from neuralnet import utils
from neuralnet.layers import *
from neuralnet.init import *

int_types = (numbers.Integral, np.integer)
__all__ = ['PixelShuffleLayer', 'ConvMeanPoolLayer', 'MeanPoolConvLayer', 'ReshapingLayer', 'UpsamplingLayer',
           'PoolingLayer', 'DownsamplingLayer', 'DetailPreservingPoolingLayer', 'GlobalAveragePoolingLayer',
           'MaxPoolingLayer', 'AveragePoolingLayer', 'UpProjectionUnit', 'DownProjectionUnit',
           'ReflectPaddingConv', 'ReflectLayer']


class DownsamplingLayer(Layer):
    """
    Original Pytorch code: https://github.com/DmitryUlyanov/deep-image-prior/blob/master/models/downsampler.py
    http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    """

    def __init__(self, input_shape, factor, kernel_type='gauss1sq2', phase=0, kernel_width=None, support=None,
                 sigma=None, preserve_size=True, layer_name='Downsampling'):
        assert phase in [0, 0.5], 'phase should be 0 or 0.5'

        super(DownsamplingLayer, self).__init__(input_shape, layer_name)
        if kernel_type == 'lanczos2':
            support = 2
            kernel_width = 4 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'lanczos3':
            support = 3
            kernel_width = 6 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'gauss12':
            kernel_width = 7
            sigma = 1 / 2
            kernel_type_ = 'gauss'

        elif kernel_type == 'gauss1sq2':
            kernel_width = 9
            sigma = 1. / np.sqrt(2)
            kernel_type_ = 'gauss'

        elif kernel_type in ['lanczos', 'gauss', 'box']:
            kernel_type_ = kernel_type

        else:
            assert False, 'Wrong kernel name.'

        self.factor = factor
        self.kernel_type = kernel_type
        self.phase = phase
        self.kernel_width = kernel_width
        self.support = support
        self.sigma = sigma
        self.descriptions = '{} Downsampling: factor {} phase {} width {}'.format(layer_name, factor, phase,
                                                                                  kernel_width)
        # note that `kernel width` will be different to actual size for phase = 1/2
        kernel = utils.get_kernel(factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma)
        self.kernel = utils.make_tensor_kernel_from_numpy((input_shape[1], input_shape[1], kernel_width, kernel_width),
                                                          kernel)
        self.preserve_size = preserve_size

        if preserve_size:
            if kernel_width % 2 == 1:
                pad = int((kernel_width - 1) / 2.)
            else:
                pad = int((kernel_width - factor) / 2.)
            self.padding = partial(utils.replication_pad, padding=pad)

    def get_output(self, input):
        if self.preserve_size:
            x = self.padding(input)
        else:
            x = input
        out = conv(x, self.kernel, subsample=(self.factor, self.factor),
                   filter_shape=(self.input_shape[1], self.input_shape[1], self.kernel_width, self.kernel_width))
        return out

    @property
    @utils.validate
    def output_shape(self):
        return tuple(self.input_shape[:2]) + tuple([s//self.factor for s in self.input_shape[2:]])


class PoolingLayer(Layer):
    def __init__(self, input_shape, window_size=(2, 2), ignore_border=True, stride=None, pad='valid', mode='max',
                 layer_name='Pooling'):
        """

        :param input_shape:
        :param window_size:
        :param ignore_border:
        :param stride:
        :param pad:
        :param mode: {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        :param layer_name:
        """
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)
        assert mode in ('max', 'sum', 'average_inc_pad', 'average_exc_pad'), 'Invalid pooling mode. ' \
                                                                             'Mode should be \'max\', \'sum\', ' \
                                                                             '\'average_inc_pad\' or \'average_exc_pad\', ' \
                                                                             'got %s' % mode

        super(PoolingLayer, self).__init__(input_shape, layer_name)
        self.ws = window_size
        self.ignore_border = ignore_border
        self.stride = stride if stride else tuple(window_size)
        self.mode = mode
        if isinstance(pad, (list, tuple)):
            self.pad = tuple(pad)
        elif isinstance(pad, int):
            self.pad = (pad, pad)
        elif isinstance(pad, str):
            if pad == 'half':
                self.pad = (window_size[0] // 2, window_size[1] // 2)
            elif pad == 'valid':
                self.pad = (0, 0)
            elif pad == 'full':
                self.pad = (window_size[0] - 1, window_size[1] - 1)
            else:
                raise NotImplementedError
        else:
            raise TypeError

        if self.pad != (0, 0):
            self.ignore_border = True

        self.descriptions = ''.join(('{} {} PoolingLayer: size: {}'.format(layer_name, mode, window_size),
                                     ' stride: {}'.format(stride), ' {} -> {}'.format(input_shape, self.output_shape)))

    def get_output(self, input):
        return pool(input, self.ws, self.stride, self.mode, self.pad) if self.ignore_border else T.signal.pool.pool_2d(
            input, self.ws, self.ignore_border, self.stride, self.pad, self.mode)

    @property
    @utils.validate
    def output_shape(self):
        size = list(self.input_shape)
        size[2] = (size[2] + 2 * self.pad[0] - self.ws[0]) // self.stride[0] + 1
        size[3] = (size[3] + 2 * self.pad[1] - self.ws[1]) // self.stride[1] + 1

        if np.mod(self.input_shape[2], self.stride[0]):
            if not self.ignore_border:
                size[2] += np.mod(self.input_shape[2], self.stride[0])
        if np.mod(self.input_shape[3], self.stride[1]):
            if not self.ignore_border:
                size[3] += np.mod(self.input_shape[3], self.stride[1])
        return tuple(size)


class DetailPreservingPoolingLayer(Layer):
    """Implementation of https://arxiv.org/abs/1804.04076"""

    def __init__(self, input_shape, window_size=(2, 2), learn_filter='False', symmetric=True, epsilon=np.sqrt(.001),
                 layer_name='Detail Preserving Pooling Layer'):
        assert input_shape[2] / 2 == input_shape[2] // 2 and input_shape[3] / 2 == input_shape[
            3] // 2, 'Input must have even rows and columns.'
        assert isinstance(window_size, (list, tuple, int)), 'window_size must be a list, tuple, or int, got %s.' % type(
            window_size)

        super(DetailPreservingPoolingLayer, self).__init__(input_shape, layer_name)
        self.ws = tuple(window_size) if isinstance(window_size, (list, tuple)) else (window_size, window_size)
        self.learn_filter = learn_filter
        self.symmetric = symmetric
        self.epsilon_sqr = np.float32(epsilon ** 2)
        self.alpha_ = theano.shared(np.zeros((input_shape[1],), theano.config.floatX), 'alpha_', borrow=True)
        self.lambda_ = theano.shared(np.zeros((input_shape[1],), theano.config.floatX), 'lambda_', borrow=True)

        self.params += [self.alpha_, self.lambda_]
        self.trainable += [self.alpha_, self.lambda_]

        if learn_filter:
            self.kern_vals = GlorotNormal()((input_shape[1], input_shape[1], 3, 3))
            self.kern = theano.shared(self.kern_vals.copy(), 'down_filter', borrow=True)
            self.params.append(self.kern)
            self.trainable.append(self.kern)
            self.regularizable.append(self.kern)
        else:
            gauss_filter = T.as_tensor_variable(np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], theano.config.floatX) / 16.)
            self.kern = T.zeros((self.input_shape[1], self.input_shape[1], 3, 3), theano.config.floatX)
            for i in range(self.input_shape[1]):
                self.kern = T.set_subtensor(self.kern[i, i], gauss_filter)
        self.descriptions = '{} Detail Preserving Pooling Layer: {} -> {}'.format(layer_name, input_shape,
                                                                                  self.output_shape)

    def __downsampling(self, input):
        output = pool(input, self.ws, self.stride, 'average_exc_pad')
        output = conv(output, self.kern, border_mode='half')
        return output

    def __penalty(self, x, lam):
        if self.symmetric:
            return T.exp(lam / 2. * T.log(x ** 2. + self.epsilon_sqr))
        else:
            return T.exp(lam / 2. * T.log(T.maximum(0., x) ** 2. + self.epsilon_sqr))

    def get_output(self, input):
        alpha = T.exp(self.alpha_).dimshuffle('x', 0, 'x', 'x')
        lam = T.exp(self.lambda_).dimshuffle('x', 0, 'x', 'x')

        down = self.__downsampling(input)
        down_up = utils.unpool(down, self.ws)
        W = alpha + self.__penalty(input - down_up, lam)
        output = pool(input * W, self.ws, mode='average_exc_pad')
        weight = 1. / pool(W, self.ws, mode='average_exc_pad')
        return output * weight

    @property
    @utils.validate
    def output_shape(self):
        return self.input_shape[:2] + (self.input_shape[2] // self.ws[0], self.input_shape[3] // self.ws[1])

    def reset(self):
        if self.learn_filter:
            self.kern.set_value(self.kern_vals.copy())


class UpProjectionUnit(Sequential):
    """
    implementation of the paper "Deep Back Projection Network"
    """

    def __init__(self, input_shape, filter_size, activation='relu', up_ratio=2, learnable=True,
                 layer_name='Up Projection'):
        super(UpProjectionUnit, self).__init__(input_shape=input_shape, layer_name=layer_name)

        self.filter_size = filter_size
        self.activation = activation
        self.up_ratio = up_ratio

        if learnable:
            self.append(TransposedConvolutionalLayer(self.input_shape, input_shape[1], filter_size,
                                                     (input_shape[2] * 2, input_shape[3] * 2), stride=(up_ratio, up_ratio),
                                                     activation=activation, layer_name=layer_name + '/up1'))
        else:
            self.append(UpsamplingLayer(self.input_shape, up_ratio, layer_name=layer_name + '/up1'))

        self.append(ConvolutionalLayer(self.output_shape, input_shape[1], filter_size, stride=(up_ratio, up_ratio),
                                       activation=activation, layer_name=layer_name+'/conv'))

        if learnable:
            self.append(TransposedConvolutionalLayer(self.input_shape, input_shape[1], filter_size,
                                                     (input_shape[2] * 2, input_shape[3] * 2), stride=(up_ratio, up_ratio),
                                                     activation=activation, layer_name=layer_name + '/up2'))
        else:
            self.append(UpsamplingLayer(self.input_shape, up_ratio, layer_name=layer_name + '/up2'))

        self.descriptions = '{} Up Projection Unit: {} -> {} upsampling by {}'.format(layer_name, input_shape,
                                                                                      self.output_shape, up_ratio)

    def get_output(self, input):
        out1 = self[self.layer_name+'/up1'](input)
        out2 = self[self.layer_name+'/conv'](out1)
        res = out2 - input
        out2 = self[self.layer_name+'/up2'](res)
        return out2 + out1


class DownProjectionUnit(Sequential):
    """
    implementation of the paper "Deep Back Projection Network"
    """

    def __init__(self, input_shape, filter_size, activation='relu', down_ratio=2, learnable=True,
                 layer_name='Down Projection'):
        super(DownProjectionUnit, self).__init__(input_shape=input_shape, layer_name=layer_name)

        self.filter_size = filter_size
        self.activation = activation
        self.down_ratio = down_ratio

        self.append(ConvolutionalLayer(input_shape,input_shape[1], filter_size, stride=(down_ratio, down_ratio),
                                       activation=activation, layer_name=layer_name+'/conv1'))

        if learnable:
            self.append(TransposedConvolutionalLayer(self.output_shape, input_shape[1], filter_size,
                                                     (input_shape[2] * 2, input_shape[3] * 2),
                                                     stride=(down_ratio, down_ratio), activation=activation,
                                                     layer_name=layer_name + '/up'))
        else:
            self.append(UpsamplingLayer(self.output_shape, down_ratio, layer_name=layer_name + '/up'))

        self.append(ConvolutionalLayer(self.output_shape, input_shape[1], filter_size, stride=(down_ratio, down_ratio),
                                       activation=activation, layer_name=layer_name+'/conv2'))

        self.descriptions = '{} Down Projection Unit: {} -> {} downsampling by {}'.format(layer_name, input_shape,
                                                                                          self.output_shape, down_ratio)

    def get_output(self, input):
        out1 = self[self.layer_name+'/conv1'](input)
        out2 = self[self.layer_name+'/up'](out1)
        res = out2 - input
        out2 = self[self.layer_name+'/conv2'](res)
        return out2 + out1


class PixelShuffleLayer(Layer):
    def __init__(self, input_shape, num_filters, filter_size, rate=2, activation='linear', init=HeNormal(gain=1.),
                 biases=True, layer_name='Upsample Conv', **kwargs):
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(PixelShuffleLayer, self).__init__(input_shape, layer_name)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.rate = rate
        self.activation = activation
        self.biases = biases

        self.shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]*rate, self.input_shape[3]*rate)
        self.conv = ConvolutionalLayer(self.shape, num_filters, filter_size, init=init, activation=self.activation,
                                       layer_name=self.layer_name, no_bias=not self.biases, **kwargs)
        self.params += self.conv.params
        self.trainable += self.conv.trainable
        self.regularizable += self.conv.regularizable
        self.descriptions = '{} Upsample Conv: {} -> {}'.format(layer_name, self.input_shape, self.output_shape)

    def get_output(self, input):
        output = input
        output = T.concatenate([output for _ in range(self.rate ** 2)], 1)
        output = utils.depth_to_space(output, self.rate)
        return self.conv(output)

    @property
    def output_shape(self):
        return (self.shape[0], self.num_filters, self.shape[2], self.shape[3])

    def reset(self):
        self.conv.reset()


class UpsamplingLayer(Layer):
    def __init__(self, input_shape, ratio=None, frac_ratio=None, layer_name='Upsampling', method='bilinear'):
        if ratio != int(ratio):
            raise NotImplementedError
        if ratio and frac_ratio:
            raise NotImplementedError
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)
        assert method.lower() in ('bilinear', 'nearest'), 'Unknown %s upsampling method.' % method

        super(UpsamplingLayer, self).__init__(tuple(input_shape), layer_name)
        self.ratio = ratio
        self.frac_ratio = frac_ratio
        self.method = method.lower()
        self.descriptions = '{} x{} Resizing Layer {} -> {}'.format(layer_name, self.ratio, self.input_shape,
                                                                    self.output_shape)

    def get_output(self, input):
        if self.method == 'bilinear':
            return T.nnet.abstract_conv.bilinear_upsampling(input, ratio=self.ratio) if self.ratio \
                else T.nnet.abstract_conv.bilinear_upsampling(input, frac_ratio=self.frac_ratio)
        else:
            return T.repeat(T.repeat(input, self.ratio, 2), self.ratio, 3)

    @property
    @utils.validate
    def output_shape(self):
        return self.input_shape[0], self.input_shape[1], self.input_shape[2] * self.ratio, self.input_shape[
            3] * self.ratio


class ReshapingLayer(Layer):
    def __init__(self, input_shape, new_shape, layer_name='Reshape Layer'):
        super(ReshapingLayer, self).__init__(tuple(input_shape), layer_name)
        self.new_shape = tuple(new_shape)
        self.descriptions = 'Reshaping Layer: {} -> {}'.format(self.input_shape, self.output_shape)

    def get_output(self, input):
        return T.reshape(input, self.new_shape)

    @property
    @utils.validate
    def output_shape(self):
        if -1 in self.new_shape:
            if self.new_shape[0] == -1:
                output = list(self.new_shape)
                output[0] = None
                return tuple(output)
            else:
                prod_shape = np.prod(self.input_shape[1:])
                prod_new_shape = np.prod(self.new_shape) * -1
                shape = [x if x != -1 else prod_shape // prod_new_shape for x in self.input_shape]
                return tuple(shape)
        else:
            return tuple(self.new_shape)


class ReflectLayer(Layer):
    def __init__(self, input_shape, width, batch_ndim=2, layer_name='Reflect Layer'):
        super(ReflectLayer, self).__init__(input_shape, layer_name)
        self.width = width
        self.batch_ndim = batch_ndim
        self.descriptions = '{} Reflect layer: width {} no padding before {}'.format(layer_name, width, batch_ndim)

    @property
    def output_shape(self):
        output_shape = list(self.input_shape)

        if isinstance(self.width, int):
            widths = [self.width] * (len(self.input_shape) - self.batch_ndim)
        else:
            widths = self.width

        for k, w in enumerate(widths):
            if output_shape[k + self.batch_ndim] is None:
                continue
            else:
                try:
                    l, r = w
                except TypeError:
                    l = r = w
                output_shape[k + self.batch_ndim] += l + r
        return tuple(output_shape)

    def get_output(self, input):
        return utils.reflect_pad(input, self.width, self.batch_ndim)


def MeanPoolConvLayer(input_shape, num_filters, filter_size, activation='linear', ws=(2, 2), init=HeNormal(gain=1.),
                      no_bias=False, layer_name='Mean Pool Conv', **kwargs):
    assert input_shape[2] / 2 == input_shape[2] // 2 and input_shape[3] / 2 == input_shape[
        3] // 2, 'Input must have even shape.'

    block = Sequential(input_shape=input_shape, layer_name=layer_name)
    block.append(PoolingLayer(input_shape, ws, stride=ws, ignore_border=True, mode='average_exc_pad',
                              layer_name=layer_name + '/meanpool'))
    block.append(ConvolutionalLayer(block.output_shape, num_filters, filter_size, init, no_bias,
                                    layer_name=layer_name + '/conv', activation=activation, **kwargs))
    return block


def ConvMeanPoolLayer(input_shape, num_filters, filter_size, activation='linear', ws=(2, 2), init=HeNormal(gain=1.),
                      no_bias=False, layer_name='Conv Mean Pool', **kwargs):
    assert input_shape[2] / 2 == input_shape[2] // 2 and input_shape[3] / 2 == input_shape[
        3] // 2, 'Input must have even shape.'

    block = Sequential(input_shape=input_shape, layer_name=layer_name)
    block.append(ConvolutionalLayer(block.output_shape, num_filters, filter_size, init, no_bias,
                                    layer_name=layer_name + '/conv', activation=activation, **kwargs))
    block.append(PoolingLayer(block.output_shape, ws, stride=ws, ignore_border=True, mode='average_exc_pad',
                              layer_name=layer_name + '/meanpool'))
    return block


def ReflectPaddingConv(input_shape, num_filters, filter_size=3, stride=1, activation='relu', use_batchnorm=True,
                       layer_name='Reflect Padding Conv', **kwargs):
    assert filter_size % 2 == 1
    pad_size = filter_size >> 1
    block = Sequential(input_shape=input_shape, layer_name=layer_name)
    block.append(ReflectLayer(block.output_shape, pad_size, layer_name=layer_name+'/Reflect'))
    if use_batchnorm:
        block.append(
            ConvNormAct(block.output_shape, num_filters, filter_size, Normal(.02), border_mode=0, stride=stride,
                        activation=activation, layer_name=layer_name + '/conv_bn_act', normalization='gn',
                        groups=num_filters))
    else:
        block.append(ConvolutionalLayer(block.output_shape, num_filters, filter_size, Normal(.02), border_mode=0,
                                        stride=stride, activation=activation, layer_name=layer_name+'/conv'))
    return block


class PaddingLayer(Layer):
    """
    Pad a tensor with a constant value. Adapted from Lasagne
    Parameters
    ----------
    width : int, iterable of int, or iterable of tuple
        Padding width. If an int, pads each axis symmetrically with the same
        amount in the beginning and end. If an iterable of int, defines the
        symmetric padding width separately for each axis. If an iterable of
        tuples of two ints, defines a seperate padding width for each beginning
        and end of each axis.
    val : float
        The constant value used for padding
    batch_ndim : integer
        Dimensions before the value will not be padded.
    """
    def __init__(self, input_shape, width, val=0, batch_ndim=2, layer_name='Padding Layer'):
        super(PaddingLayer, self).__init__(input_shape, layer_name)
        self.width = width
        self.val = val
        self.batch_ndim = batch_ndim
        self.descriptions = '{} Padding Layer: from dim {} width {} value {}'.format(layer_name, batch_ndim, width, val)

    def get_output(self, input):
        input_shape = input.shape
        input_ndim = input.ndim

        output_shape = list(input_shape)
        indices = [slice(None) for _ in output_shape]

        if isinstance(self.width, int_types):
            widths = [self.width] * (input_ndim - self.batch_ndim)
        else:
            widths = self.width

        for k, w in enumerate(widths):
            try:
                l, r = w
            except TypeError:
                l = r = w
            output_shape[k + self.batch_ndim] += l + r
            indices[k + self.batch_ndim] = slice(l, l + input_shape[k + self.batch_ndim])

        if self.val:
            out = T.ones(output_shape) * self.val
        else:
            out = T.zeros(output_shape)
        return T.set_subtensor(out[tuple(indices)], input)

    @property
    @utils.validate
    def output_shape(self):
        shape = list(self.input_shape)
        for i in range(self.batch_ndim, len(self.input_shape)):
            shape[i] += self.width * 2
        return tuple(shape)


def GlobalAveragePoolingLayer(input_shape, layer_name='GlbAvg Pooling'):
    return PoolingLayer(input_shape, input_shape[2:], True, (1, 1), (0, 0), 'average_exc_pad', layer_name)


def MaxPoolingLayer(input_shape, ws, ignore_border=True, stride=None, pad='valid', layer_name='Max Pooling'):
    return PoolingLayer(input_shape, ws, ignore_border, stride, pad, 'max', layer_name)


def AveragePoolingLayer(input_shape, ws, ignore_border=True, stride=None, pad='valid', layer_name='Avg Pooling'):
    return PoolingLayer(input_shape, ws, ignore_border, stride, pad, 'average_exc_pad', layer_name)
