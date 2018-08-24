"""
Written and collected by Duc Nguyen
"""
__author__ = 'Duc Nguyen'

import time
import abc
import theano
import numpy as np
from theano import tensor as T
from theano.tensor.nnet import conv2d as conv
from theano.tensor.signal.pool import pool_2d as pool
from collections import OrderedDict
from functools import partial

from neuralnet import utils
from neuralnet.init import *

__all__ = ['Layer', 'Sequential', 'ConvolutionalLayer', 'FullyConnectedLayer', 'TransformerLayer',
           'TransposedConvolutionalLayer', 'BatchNormLayer', 'BatchRenormLayer', 'DecorrBatchNormLayer',
           'DenseBlock', 'SumLayer', 'StackingConv', 'ScalingLayer', 'SlicingLayer',
           'PixelShuffleLayer', 'LSTMCell', 'ActivationLayer', 'AttConvLSTMCell', 'ConvMeanPoolLayer',
           'ConcatLayer', 'ConvLSTMCell', 'ConvNormAct', 'MeanPoolConvLayer', 'ReshapingLayer',
           'RecursiveResNetBlock', 'ResizingLayer', 'ResNetBlock', 'ResNetBottleneckBlock', 'GRUCell',
           'IdentityLayer', 'DropoutLayer', 'PoolingLayer', 'InceptionModule1', 'InceptionModule2',
           'InceptionModule3', 'DownsamplingLayer', 'DetailPreservingPoolingLayer', 'NetworkInNetworkBlock',
           'GlobalAveragePoolingLayer', 'MaxPoolingLayer', 'SoftmaxLayer', 'TransposingLayer',
           'set_training_status', 'AveragePoolingLayer', 'WarpingLayer', 'GroupNormLayer', 'UpProjectionUnit',
           'DownProjectionUnit']


def validate(func):
    """make sure output shape is a list of ints"""
    def func_wrapper(self):
        out = [int(x) if x is not None else x for x in func(self)]
        return tuple(out)
    return func_wrapper


class NetMethod:
    def get_output(self, input):
        raise NotImplementedError

    def __str__(self):
        return self.descriptions

    @property
    def output_shape(self):
        raise NotImplementedError

    def reset(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.get_output(*args, **kwargs)


class Layer(NetMethod, metaclass=abc.ABCMeta):
    training_flag = False

    def __init__(self, input_shape, layer_name=''):
        assert isinstance(input_shape, list) or isinstance(input_shape, tuple), \
            'input_shape must be list or tuple. Received %s' % type(input_shape)
        self.input_shape = tuple(input_shape)
        self.rng = np.random.RandomState(int(time.time()))
        self.params = []
        self.trainable = []
        self.regularizable = []
        self.layer_name = layer_name
        self.descriptions = ''

    @staticmethod
    def set_training_status(training):
        Layer.training_flag = training


class Sequential(OrderedDict, NetMethod):
    """
    Mimicking Pytorch Sequential class but inheriting OrderedDict
    """
    def __init__(self, layer_list=(), input_shape=None, layer_name='Sequential'):
        assert layer_list or input_shape, 'Either layer_list or input_shape must be specified.'
        assert isinstance(layer_list, (list, tuple, Sequential)), 'layer_list must be a list or tuple, got %s.' % type(
            layer_list)
        assert all([isinstance(l, (Sequential, Layer)) for l in
                    layer_list]), 'All elements of layer_list should be instances of Layer or Sequential.'

        self.params, self.trainable, self.regularizable = [], [], []
        self.descriptions = ''
        self.layer_name = layer_name
        self.input_shape = tuple(input_shape) if input_shape else layer_list[0].input_shape
        if isinstance(layer_list, (list, tuple)):
            name_list = [l.layer_name for l in layer_list]
            for i in range(len(name_list)):
                name = name_list.pop()
                if name in name_list:
                    raise ValueError('%s already existed in the network.' % name)
            super(Sequential, self).__init__(zip([l.layer_name for l in layer_list], layer_list))
        else:
            super(Sequential, self).__init__(layer_list)

        self.__idx = 0
        self.__max = len(self)

    def __iter__(self):
        self.__idx = 0
        self.__max = len(self)
        return self

    def __next__(self):
        if self.__idx >= self.__max:
            raise StopIteration
        self.__idx += 1
        return self[self.__idx - 1]

    def __getitem__(self, item):
        assert isinstance(item, (int, slice, str)), 'index should be either str, int or slice, got %s.' % type(item)
        if isinstance(item, int):
            keys = list(self.keys())
            return self[keys[item]]
        elif isinstance(item, slice):
            keys = list(self.keys())
            return Sequential(self[keys[item]], layer_name=self.layer_name)
        else:
            return super(Sequential, self).__getitem__(item)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError('key must be a str, got %s.' % type(key))
        if not isinstance(value, (Sequential, Layer)):
            raise TypeError('value must be an instance of Layer or Sequential, got %s.' % type(value))
        if key in self.keys():
            raise NameError('key existed.')
        super(Sequential, self).__setitem__(key, value)
        self.params += value.params
        self.trainable += value.trainable
        self.regularizable += value.regularizable
        self.descriptions += value.descriptions + '\n'

    def get_output(self, input):
        out = input
        for layer in self.values():
            out = layer(out)
        return out

    @property
    def output_shape(self):
        if self:
            return self[-1].output_shape
        else:
            return self.input_shape

    def append(self, layer):
        if not isinstance(layer, (Sequential, Layer)):
            raise TypeError('layer must be an instance of Layer or Sequential, got %s.' % type(layer))
        if layer.layer_name in self:
            raise NameError('Name %s already existed.' % layer.layer_name)
        self[layer.layer_name] = layer

    def update(self, other):
        if other is None:
            return
        if isinstance(other, Sequential):
            for layer in other:
                if layer.layer_name not in self.keys():
                    self[layer.layer_name] = layer
                else:
                    raise NameError('Name %s already existed.' % layer.layer_name)
        elif isinstance(other, Layer):
            if other.layer_name not in self.keys():
                self[other.layer_name] = other
            else:
                raise NameError('Name %s already existed.' % other.layer_name)
        else:
            raise TypeError('Cannot update a Sequential instance with a %s instance.' % type(other))

    def __add__(self, other):
        assert isinstance(other, Sequential), 'Cannot concatenate a Sequential object with a %s object.' % type(other)
        res = Sequential(input_shape=tuple(self.input_shape), layer_name=self.layer_name)

        for key in self.keys():
            res[key] = self[key]

        for key in other.keys():
            res[key] = other[key]

        res.__idx = 0
        res.__max = len(res)
        return res

    def reset(self):
        for layer in self:
            layer.reset()


class ActivationLayer(Layer):
    def __init__(self, input_shape, activation='relu', layer_name='Activation', **kwargs):
        super(ActivationLayer, self).__init__(input_shape, layer_name)
        self.activation = utils.function[activation]
        self.kwargs = kwargs
        self.descriptions = '{} Activation layer: {}'.format(self.layer_name, activation)

        if activation == 'prelu':
            self.alpha = theano.shared(np.float32(.1), layer_name + '/alpha')
            self.params += [self.alpha]
            self.trainable += [self.alpha]
            self.kwargs['alpha'] = self.alpha

    def get_output(self, input):
        return self.activation(input, **self.kwargs)

    @property
    def output_shape(self):
        return tuple(self.input_shape)

    def reset(self):
        if self.activation is utils.function['prelu']:
            self.alpha.set_value(np.float32(.1))


class DownsamplingLayer(Layer):
    """
    Original Pytorch code: https://github.com/DmitryUlyanov/deep-image-prior/blob/master/models/downsampler.py
    http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    """

    def __init__(self, input_shape, factor, kernel_type='gauss1sq2', phase=0, kernel_width=None, support=None, sigma=None,
                 preserve_size=True, layer_name='DownsamplingLayer'):
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
        self.descriptions = '{} Downsampling: factor {} phase {} width {}'.format(layer_name, factor, phase, kernel_width)
        # note that `kernel width` will be different to actual size for phase = 1/2
        kernel = utils.get_kernel(factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma)
        self.kernel = utils.make_tensor_kernel_from_numpy((input_shape[1], input_shape[1], kernel_width, kernel_width), kernel)
        self.preserve_size = preserve_size

        if preserve_size:
            if kernel_width % 2 == 1:
                pad = int((kernel_width - 1) / 2.)
            else:
                pad = int((kernel_width - factor) / 2.)
            self.padding = partial(utils.replication_pad2d, padding=pad)

    def get_output(self, input):
        if self.preserve_size:
            x = self.padding(input)
        else:
            x = input
        out = conv(x, self.kernel, subsample=(self.factor, self.factor),
                   filter_shape=(self.input_shape[1], self.input_shape[1], self.kernel_width, self.kernel_width))
        return out

    @property
    @validate
    def output_shape(self):
        return tuple(self.input_shape[:2]) + tuple([s//self.factor for s in self.input_shape[2:]])


class PoolingLayer(Layer):
    def __init__(self, input_shape, window_size=(2, 2), ignore_border=True, stride=(2, 2), pad='valid', mode='max',
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
        return pool(input, self.ws, self.ignore_border, self.stride, self.pad, self.mode)

    @property
    @validate
    def output_shape(self):
        size = list(self.input_shape)
        if np.mod(size[2], self.ws[0]):
            size[2] -= np.mod(size[2], self.ws[0])
        if np.mod(size[3], self.ws[1]):
            size[3] -= np.mod(size[3], self.ws[1])

        size[2] = (size[2] + 2 * self.pad[0] - self.ws[0]) // self.stride[0] + 1
        size[3] = (size[3] + 2 * self.pad[1] - self.ws[1]) // self.stride[1] + 1

        if np.mod(self.input_shape[2], self.ws[0]):
            if not self.ignore_border:
                size[2] += np.mod(self.input_shape[2], self.ws[0])
        if np.mod(self.input_shape[3], self.ws[1]):
            if not self.ignore_border:
                size[3] += np.mod(self.input_shape[3], self.ws[1])
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
        self.alpha_ = theano.shared(np.zeros((input_shape[1],), 'float32'), 'alpha_', borrow=True)
        self.lambda_ = theano.shared(np.zeros((input_shape[1],), 'float32'), 'lambda_', borrow=True)

        self.params += [self.alpha_, self.lambda_]
        self.trainable += [self.alpha_, self.lambda_]

        if learn_filter:
            self.kern_vals = GlorotNormal()((input_shape[1], input_shape[1], 3, 3))
            self.kern = theano.shared(self.kern_vals.copy(), 'down_filter', borrow=True)
            self.params.append(self.kern)
            self.trainable.append(self.kern)
            self.regularizable.append(self.kern)
        else:
            gauss_filter = T.as_tensor_variable(np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], 'float32') / 16.)
            self.kern = T.zeros((self.input_shape[1], self.input_shape[1], 3, 3), 'float32')
            for i in range(self.input_shape[1]):
                self.kern = T.set_subtensor(self.kern[i, i], gauss_filter)
        self.descriptions = '{} Detail Preserving Pooling Layer: {} -> {}'.format(layer_name, input_shape,
                                                                                  self.output_shape)

    def __downsampling(self, input):
        output = pool(input, self.ws, True, mode='average_exc_pad')
        output = T.nnet.conv2d(output, self.kern, border_mode='half')
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
        output = pool(input * W, self.ws, True, mode='average_exc_pad')
        weight = 1. / pool(W, self.ws, True, mode='average_exc_pad')
        return output * weight

    @property
    @validate
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
                 layer_name='UpProjectionUnit'):
        super(UpProjectionUnit, self).__init__(input_shape=input_shape, layer_name=layer_name)

        self.filter_size = filter_size
        self.activation = activation
        self.up_ratio = up_ratio

        if learnable:
            self.append(TransposedConvolutionalLayer(self.input_shape, input_shape[1], filter_size,
                                                     (input_shape[2] * 2, input_shape[3] * 2), stride=(up_ratio, up_ratio),
                                                     activation=activation, layer_name=layer_name + '/up1'))
        else:
            self.append(ResizingLayer(self.input_shape, up_ratio, layer_name=layer_name+'/up1'))

        self.append(ConvolutionalLayer(self.output_shape, input_shape[1], filter_size, stride=(up_ratio, up_ratio),
                                       activation=activation, layer_name=layer_name+'/conv'))

        if learnable:
            self.append(TransposedConvolutionalLayer(self.input_shape, input_shape[1], filter_size,
                                                     (input_shape[2] * 2, input_shape[3] * 2), stride=(up_ratio, up_ratio),
                                                     activation=activation, layer_name=layer_name + '/up2'))
        else:
            self.append(ResizingLayer(self.input_shape, up_ratio, layer_name=layer_name + '/up2'))

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
                 layer_name='DownProjectionUnit'):
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
            self.append(ResizingLayer(self.output_shape, down_ratio, layer_name=layer_name+'/up'))

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


class DropoutLayer(Layer):
    def __init__(self, input_shape, drop_prob=0.5, gaussian=False, layer_name='Dropout'):
        assert len(input_shape) == 2 or len(input_shape) == 4, \
            'input_shape must have 2 or 4 elements. Received %d' % len(input_shape)

        super(DropoutLayer, self).__init__(input_shape, layer_name)
        self.gaussian = gaussian
        self.srng = theano.sandbox.rng_mrg.MRG_RandomStreams(self.rng.randint(1, int(time.time())))
        self.keep_prob = T.as_tensor_variable(np.float32(1. - drop_prob))
        self.descriptions = '{} Dropout Layer: p={:.2f}'.format(layer_name, 1. - drop_prob)

    def get_output(self, input):
        mask = self.srng.normal(input.shape) + 1. if self.gaussian else self.srng.binomial(n=1, p=self.keep_prob,
                                                                                           size=input.shape,
                                                                                           dtype='float32')
        output_on = input * mask
        output_off = input if self.gaussian else input * self.keep_prob
        return output_on if self.training_flag else output_off

    @property
    @validate
    def output_shape(self):
        return tuple(self.input_shape)


class FullyConnectedLayer(Layer):
    def __init__(self, input_shape, num_nodes, init=HeNormal(gain=1.), no_bias=False, layer_name='fc',
                 activation='relu', keep_dims=False, **kwargs):
        """

        :param input_shape:
        :param num_nodes:
        :param He_init:
        :param He_init_gain:
        :param no_bias:
        :param layer_name:
        :param activation:
        :param target:
        :param kwargs:
        """
        assert len(input_shape) == 2 or len(input_shape) == 4, \
            'input_shape must have 2 or 4 elements. Received %d' % len(input_shape)

        in_shape = tuple(input_shape) if len(input_shape) == 2 else (input_shape[0], np.prod(input_shape[1:]))
        super(FullyConnectedLayer, self).__init__(in_shape, layer_name)
        self.num_nodes = num_nodes
        self.activation = utils.function[activation]
        self.no_bias = no_bias
        self.keep_dims = keep_dims
        self.kwargs = kwargs

        self.W_values = init((self.input_shape[1], num_nodes))
        self.W = theano.shared(value=np.copy(self.W_values), name=self.layer_name + '/W', borrow=True)
        self.trainable.append(self.W)
        self.params.append(self.W)
        self.regularizable.append(self.W)

        if not self.no_bias:
            self.b_values = np.zeros((num_nodes,), dtype=theano.config.floatX)
            self.b = theano.shared(value=np.copy(self.b_values), name=self.layer_name + '/b', borrow=True)
            self.trainable.append(self.b)
            self.params.append(self.b)

        if activation == 'prelu':
            self.alpha = theano.shared(np.float32(.1), layer_name + '/alpha')
            self.params += [self.alpha]
            self.trainable += [self.alpha]
            self.kwargs['alpha'] = self.alpha

        self.descriptions = '{} FC: in_shape = {} weight shape = {} -> {} activation: {}'\
            .format(self.layer_name, self.input_shape, (self.input_shape[1], num_nodes), self.output_shape, activation)

    def get_output(self, input):
        output = T.dot(input.flatten(2), self.W) + self.b if not self.no_bias else T.dot(input.flatten(2), self.W)
        return self.activation(output, **self.kwargs) if self.keep_dims else T.squeeze(self.activation(output, **self.kwargs))

    @property
    @validate
    def output_shape(self):
        return (self.input_shape[0], self.num_nodes // self.kwargs.get('maxout_size', 4)) if self.activation is 'maxout' \
            else (self.input_shape[0], self.num_nodes)

    def reset(self):
        self.W.set_value(np.copy(self.W_values))
        if not self.no_bias:
            self.b.set_value(np.copy(self.b_values))
        if self.activation is utils.function['prelu']:
            self.alpha.set_value(np.float32(.1))


class ConvolutionalLayer(Layer):
    def __init__(self, input_shape, num_filters, filter_size, init=HeNormal(gain=1.), no_bias=True, border_mode='half',
                 stride=(1, 1), dilation=(1, 1), layer_name='conv', activation='relu', **kwargs):
        """

        :param input_shape:
        :param num_filters:
        :param filter_size:
        :param He_init:
        :param He_init_gain:
        :param no_bias:
        :param border_mode:
        :param stride:
        :param dilation:
        :param layer_name:
        :param activation:
        :param target:
        :param kwargs:
        """
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)
        assert isinstance(num_filters, int) and isinstance(filter_size, (int, list, tuple))
        assert isinstance(border_mode, (int, list, tuple, str)), 'border_mode should be either \'int\', ' \
                                                                 '\'list\', \'tuple\' or \'str\', got {}'.format(type(border_mode))
        assert isinstance(stride, (int, list, tuple))

        super(ConvolutionalLayer, self).__init__(input_shape, layer_name)
        self.filter_shape = (num_filters, input_shape[1], filter_size[0], filter_size[1]) if isinstance(filter_size, (list, tuple)) \
            else (num_filters, input_shape[1], filter_size, filter_size)
        self.no_bias = no_bias
        self.activation = utils.function[activation]
        self.border_mode = border_mode
        self.subsample = tuple(stride) if isinstance(stride, (tuple, list)) else (stride, stride)
        self.dilation = dilation
        self.filter_flip = kwargs.pop('filter_flip', True)
        self.kwargs = kwargs

        self.W_values = init(self.filter_shape)
        self.W = theano.shared(np.copy(self.W_values), name=self.layer_name + '/W', borrow=True)
        self.trainable.append(self.W)
        self.params.append(self.W)

        if not self.no_bias:
            self.b_values = np.zeros(self.filter_shape[0], dtype=theano.config.floatX)
            self.b = theano.shared(np.copy(self.b_values), self.layer_name + '/b', borrow=True)
            self.trainable.append(self.b)
            self.params.append(self.b)

        if activation == 'prelu':
            self.alpha = theano.shared(np.float32(.1), layer_name + '/alpha')
            self.params += [self.alpha]
            self.trainable += [self.alpha]
            self.kwargs['alpha'] = self.alpha

        self.regularizable += [self.W]
        self.descriptions = ''.join(('{} Conv Layer: '.format(self.layer_name), 'border mode: {} '.format(border_mode),
                                     'subsampling: {} dilation {} '.format(stride, dilation), 'input shape: {} x '.format(input_shape),
                                     'filter shape: {} '.format(self.filter_shape), '-> output shape {} '.format(self.output_shape),
                                     'activation: {} '.format(activation)))

    def get_output(self, input):
        output = conv(input=input, filters=self.W, border_mode=self.border_mode, subsample=self.subsample,
                      filter_flip=self.filter_flip, filter_shape=self.filter_shape)
        if not self.no_bias:
            output += self.b.dimshuffle(('x', 0, 'x', 'x'))
        return self.activation(output, **self.kwargs)

    @property
    @validate
    def output_shape(self):
        size = list(self.input_shape)
        assert len(size) == 4, "Shape must consist of 4 elements only"

        k1, k2 = self.filter_shape[2] + (self.filter_shape[2] - 1)*(self.dilation[0] - 1), \
                 self.filter_shape[3] + (self.filter_shape[3] - 1)*(self.dilation[1] - 1)

        if isinstance(self.border_mode, str):
            if self.border_mode == 'half':
                p = (k1 // 2, k2 // 2)
            elif self.border_mode == 'valid':
                p = (0, 0)
            elif self.border_mode == 'full':
                p = (k1 - 1, k2 - 1)
            else:
                raise NotImplementedError
        elif isinstance(self.border_mode, (list, tuple)):
            p = tuple(self.border_mode)
        elif isinstance(self.border_mode, int):
            p = (self.border_mode, self.border_mode)
        else:
            raise NotImplementedError

        size[2] = (size[2] - k1 + 2*p[0]) // self.subsample[0] + 1
        size[3] = (size[3] - k2 + 2*p[1]) // self.subsample[1] + 1

        size[1] = self.filter_shape[0] // self.kwargs.get('maxout_size', 4) if self.activation == utils.maxout else self.filter_shape[0]
        return tuple(size)

    def reset(self):
        self.W.set_value(np.copy(self.W_values))
        if not self.no_bias:
            self.b.set_value(np.copy(self.b_values))
        if self.activation is utils.function['prelu']:
            self.alpha.set_value(np.float32(.1))


class PerturbativeLayer(Layer):
    def __init__(self, input_shape, num_filters, init=HeNormal(), noise_level=.1, activation='relu', no_bias=True,
                 layer_name='PerturbativeLayer', **kwargs):
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(PerturbativeLayer, self).__init__(input_shape, layer_name)
        self.num_filters = num_filters
        self.noise_level = noise_level
        self.activation = utils.function[activation]
        self.no_bias = no_bias
        self.kwargs = kwargs

        self.noise = (2. * T.as_tensor_variable(np.random.rand(*input_shape[1:]).astype('float32')) - 1.) * noise_level
        self.W_values = init((num_filters, input_shape[1]))
        self.W = theano.shared(self.W_values, layer_name+'/W', borrow=True)
        self.params += [self.W]
        self.trainable += [self.W]
        self.regularizable += [self.W]

        if not no_bias:
            self.b = theano.shared(np.zeros((num_filters,), 'float32'), layer_name+'/b', borrow=True)
            self.params += [self.b]
            self.trainable += [self.b]

        if activation == 'prelu':
            self.alpha = theano.shared(.1, layer_name+'/alpha')
            self.kwargs['alpha'] = self.alpha
            self.params += [self.alpha]
            self.trainable += [self.alpha]

        self.descriptions = '{} Perturbative layer: {} -> {}, noise level {} activation {}'.format(layer_name,
                                                                                                   input_shape, self.output_shape,
                                                                                                   noise_level, activation)

    def get_output(self, input):
        input += self.noise.dimshuffle('x', 0, 1, 2)
        input = self.activation(input, **self.kwargs)
        kern = self.W.dimshuffle(0, 1, 'x', 'x')
        output = T.nnet.conv2d(input, kern, border_mode='half')
        return output if self.no_bias else output + self.b.dimshuffle('x', 0, 'x', 'x')

    @property
    def output_shape(self):
        return (self.input_shape[0],) + (self.num_filters,) + self.input_shape[2:]

    def reset(self):
        self.W.set_value(self.W_values)
        if self.activation is utils.function['prelu']:
            self.alpha.set_value(.1)
        if not self.no_bias:
            self.b.set_value(np.zeros((self.num_filters,), 'float32'))


class InceptionModule1(Layer):
    def __init__(self, input_shape, num_filters=48, border_mode='half', stride=(1, 1), activation='relu',
                 layer_name='inception_mixed1'):
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(InceptionModule1, self).__init__(input_shape, layer_name)
        self.border_mode = border_mode
        self.stride = stride
        self.activation = activation

        self.module = [[], [], [], []]
        self.module[0].append(ConvNormAct(input_shape, num_filters, (1, 1), border_mode=border_mode,
                                          stride=(1, 1), activation=activation, layer_name=layer_name + '/branch1_conv1x1'))
        self.module[0].append(ConvNormAct(self.module[0][-1].output_shape, num_filters * 4 // 3, 3, border_mode=border_mode,
                                          stride=(1, 1), activation=activation, layer_name=layer_name + '/branch1_conv3x3'))
        self.module[0].append(ConvNormAct(self.module[0][-1].output_shape, num_filters * 4 // 3, 3, border_mode=border_mode,
                                          stride=stride, activation=activation, layer_name=layer_name + '/branch1_conv3x3'))

        self.module[1].append(ConvNormAct(input_shape, num_filters * 4 // 3, 1, border_mode=border_mode, stride=(1, 1),
                                          activation=activation, layer_name=layer_name + '/branch2_conv1x1'))
        self.module[1].append(ConvNormAct(self.module[1][-1].output_shape, num_filters * 2, 3, border_mode=border_mode,
                                          stride=stride, activation=activation, layer_name=layer_name + '/branch2_conv3x3'))

        self.module[2].append(PoolingLayer(input_shape, (3, 3), stride=stride, mode='average_exc_pad', pad='half',
                                           ignore_border=True, layer_name=layer_name + '/branch3_pooling'))
        self.module[2].append(ConvNormAct(self.module[2][-1].output_shape, num_filters * 2 // 3, 1, border_mode=border_mode,
                                          stride=(1, 1), activation=activation, layer_name=layer_name + '/branch3_conv1x1'))

        self.module[3].append(ConvNormAct(input_shape, num_filters * 4 // 3, 1, border_mode=border_mode, stride=stride,
                                          activation=activation, layer_name=layer_name + '/branch4_conv1x1'))

        self.descriptions = '{} Inception module 1: {} -> {}'.format(layer_name, input_shape, self.output_shape)

        self.params += [p for block in self.module for layer in block for p in layer.params]
        self.trainable += [p for block in self.module for layer in block for p in layer.trainable]
        self.regularizable += [p for block in self.module for layer in block for p in layer.regularizable]

    def get_output(self, input):
        output = [utils.inference(input, block) for block in self.module]
        return T.concatenate(output, 1)

    @property
    def output_shape(self):
        depth = 0
        for block in self.module:
            depth += block[-1].output_shape[1]
        return (self.module[0][-1].output_shape[0], depth, self.module[0][-1].output_shape[2],
                self.module[0][-1].output_shape[3])

    def reset(self):
        for block in self.module:
            for layer in block:
                layer.reset()


class InceptionModule2(Layer):
    def __init__(self, input_shape, num_filters=128, filter_size=7, border_mode='half', stride=(1, 1), activation='relu',
                 layer_name='inception_mixed2'):
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(InceptionModule2, self).__init__(input_shape, layer_name)
        self.filter_size = filter_size
        self.border_mode = border_mode
        self.stride = stride
        self.activation = activation

        self.module = [[], [], [], []]
        self.module[0].append(ConvNormAct(input_shape, num_filters, (1, 1), border_mode=border_mode, stride=(1, 1),
                                          activation=activation, layer_name=layer_name + '/branch1_conv1x1'))
        self.module[0].append(ConvNormAct(self.module[0][-1].output_shape, num_filters, (filter_size, 1),
                                          border_mode=border_mode, stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '/branch1_conv7x1_1'))
        self.module[0].append(ConvNormAct(self.module[0][-1].output_shape, num_filters, (1, filter_size),
                                          border_mode=border_mode, stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '/branch1_conv1x7_1'))
        self.module[0].append(ConvNormAct(self.module[0][-1].output_shape, num_filters, (filter_size, 1),
                                          border_mode=border_mode, stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '/branch1_conv7x1_2'))
        self.module[0].append(ConvNormAct(self.module[0][-1].output_shape, num_filters * 3 // 2, (1, filter_size),
                                          border_mode=border_mode, stride=stride, activation=activation,
                                          layer_name=layer_name + '/branch1_conv1x7_2'))

        self.module[1].append(ConvNormAct(input_shape, 64, (1, 1), border_mode=border_mode, stride=(1, 1),
                                          activation=activation, layer_name=layer_name + '/branch2_conv1x1'))
        self.module[1].append(ConvNormAct(self.module[1][-1].output_shape, num_filters, (filter_size, 1),
                                          border_mode=border_mode, stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '/branch2_conv7x1'))
        self.module[1].append(ConvNormAct(self.module[1][-1].output_shape, num_filters * 3 // 2, (1, filter_size),
                                          border_mode=border_mode, stride=stride, activation=activation,
                                          layer_name=layer_name + '/branch2_conv1x7'))

        self.module[2].append(PoolingLayer(input_shape, (3, 3), stride=stride, mode='average_exc_pad', pad='half',
                                           ignore_border=True, layer_name=layer_name + '/branch3_pooling'))
        self.module[2].append(ConvNormAct(self.module[2][-1].output_shape, num_filters * 3 // 2, (1, 1),
                                          border_mode=border_mode, stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '/branch3_conv1x1'))

        self.module[3].append(ConvNormAct(input_shape, num_filters * 3 // 2, (1, 1), border_mode=border_mode,
                                          stride=stride, activation=activation, layer_name=layer_name + '/branch4_conv1x1'))

        self.descriptions = '{} Inception module 2: {} -> {}'.format(layer_name, input_shape, self.output_shape)

        self.params += [p for block in self.module for layer in block for p in layer.params]
        self.trainable += [p for block in self.module for layer in block for p in layer.trainable]
        self.regularizable += [p for block in self.module for layer in block for p in layer.regularizable]

    def get_output(self, input):
        output = [utils.inference(input, block) for block in self.module]
        return T.concatenate(output, 1)

    @property
    def output_shape(self):
        depth = 0
        for block in self.module:
            depth += block[-1].output_shape[1]
        return (self.module[0][-1].output_shape[0], depth, self.module[0][-1].output_shape[2],
                self.module[0][-1].output_shape[3])

    def reset(self):
        for block in self.module:
            for layer in block:
                layer.reset()


class InceptionModule3(Layer):
    def __init__(self, input_shape, num_filters=320, border_mode='half', stride=(1, 1), activation='relu',
                 layer_name='inception_mixed3'):
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(InceptionModule3, self).__init__(input_shape, layer_name)
        self.border_mode = border_mode
        self.stride = stride
        self.activation = activation

        self.module = [[], [], [], []]
        self.module[0].append(ConvNormAct(input_shape, num_filters * 7 // 5, (1, 1), border_mode=border_mode,
                                          stride=(1, 1), activation=activation, layer_name=layer_name + '/branch1_conv1x1'))
        self.module[0].append(ConvNormAct(self.module[0][-1].output_shape, num_filters * 6 // 5, (3, 3),
                                          border_mode=border_mode, stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '/branch1_conv3x3'))
        self.module[0].append([[], []])
        self.module[0][-1][0].append(ConvNormAct(self.module[0][1].output_shape, num_filters * 6 // 5, (3, 1),
                                                 border_mode=border_mode, stride=stride, activation=activation,
                                                 layer_name=layer_name + '/branch1_conv3x1'))
        self.module[0][-1][1].append(ConvNormAct(self.module[0][1].output_shape, num_filters * 6 // 5, (3, 1),
                                                 border_mode=border_mode, stride=stride, activation=activation,
                                                 layer_name=layer_name + '/branch1_conv1x3'))

        self.module[1].append(ConvNormAct(input_shape, num_filters * 7 // 5, (1, 1), border_mode=border_mode,
                                          stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '/branch2_conv1x1'))
        self.module[1].append([[], []])
        self.module[1][-1][0].append(ConvNormAct(self.module[1][0].output_shape, num_filters * 6 // 5, (3, 1),
                                                 border_mode=border_mode, stride=stride, activation=activation,
                                                 layer_name=layer_name + '/branch2_conv3x1'))
        self.module[1][-1][1].append(ConvNormAct(self.module[1][0].output_shape, num_filters * 6 // 5, (3, 1),
                                                 border_mode=border_mode, stride=stride, activation=activation,
                                                 layer_name=layer_name + '/branch2_conv1x3'))

        self.module[2].append(PoolingLayer(input_shape, (3, 3), stride=stride, mode='average_exc_pad', pad='half',
                                           ignore_border=True, layer_name=layer_name + '/branch3_pooling'))
        self.module[2].append(ConvNormAct(self.module[2][-1].output_shape, num_filters * 2 // 3, (1, 1),
                                          border_mode=border_mode, stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '/branch3_conv1x1'))

        self.module[3].append(ConvNormAct(input_shape, num_filters * 4 // 3, (1, 1), border_mode=border_mode, stride=stride,
                                          activation=activation, layer_name=layer_name + '/branch4_conv1x1'))

        for block in self.module:
            for layer in block:
                if not isinstance(layer, (list, tuple)):
                    self.params += layer.params
                    self.trainable += layer.trainable
                    self.regularizable += layer.regularizable
                else:
                    for l in layer:
                        self.params += l[0].params
                        self.trainable += l[0].trainable
                        self.regularizable += l[0].regularizable

        self.descriptions = '{} Inception module 3: {} -> {}'.format(layer_name, input_shape, self.output_shape)

    def get_output(self, input):
        output = []
        for block in self.module:
            out = input
            for layer in block:
                if not isinstance(layer, (list, tuple)):
                    out = layer(out)
                else:
                    o = []
                    for l in layer:
                        o.append(l[0](out))
                    out = T.concatenate(o, 1)
            output.append(out)
        return T.concatenate(output, 1)

    @property
    def output_shape(self):
        depth = 0
        for block in self.module:
            if not isinstance(block[-1], (list, tuple)):
                depth += block[-1].output_shape[1]
            else:
                for l in block[-1]:
                    depth += l[0].output_shape[1]
        return (self.module[3][-1].output_shape[0], depth, self.module[3][-1].output_shape[2],
                self.module[3][-1].output_shape[3])

    def reset(self):
        for block in self.module:
            for layer in block:
                if isinstance(layer, list):
                    for l in layer:
                        l.reset()
                else:
                    layer.reset()


class TransposedConvolutionalLayer(Layer):
    def __init__(self, input_shape, num_filters, filter_size, output_shape=None, init=HeNormal(gain=1.),
                 layer_name='Transconv', padding='half', stride=(2, 2), activation='relu', **kwargs):
        """

        :param input_shape:
        :param num_filters:
        :param filter_size:
        :param output_shape:
        :param He_init:
        :param layer_name:
        :param W:
        :param b:
        :param padding:
        :param stride:
        :param activation:
        :param target:
        """
        assert isinstance(num_filters, int) and isinstance(filter_size, (int, list, tuple))
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(TransposedConvolutionalLayer, self).__init__(input_shape, layer_name)
        self.filter_shape = (input_shape[1], num_filters, filter_size[0], filter_size[1]) if isinstance(filter_size, (list, tuple)) \
            else (input_shape[1], num_filters, filter_size, filter_size)
        self.output_shape_tmp = (input_shape[0], num_filters, output_shape[0], output_shape[1]) \
            if output_shape is not None else (output_shape,) * 4
        self.padding = padding
        self.stride = stride
        self.activation = utils.function[activation]
        self.kwargs = kwargs

        self.W_values = init(self.filter_shape)
        self.b_values = np.zeros((self.filter_shape[1],), dtype=theano.config.floatX)
        self.W = theano.shared(np.copy(self.W_values), self.layer_name + '/W', borrow=True)
        self.b = theano.shared(np.copy(self.b_values), self.layer_name + '/b', borrow=True)
        self.params += [self.W, self.b]
        self.trainable += [self.W, self.b]
        self.regularizable.append(self.W)

        if activation == 'prelu':
            self.alpha = theano.shared(np.float32(.1), layer_name + '/alpha')
            self.params += [self.alpha]
            self.trainable += [self.alpha]
            self.kwargs['alpha'] = self.alpha

        self.descriptions = '{} Transposed Conv Layer: {} x {} -> {} padding {} stride {} activation {}'.format(
            layer_name, input_shape, self.filter_shape, self.output_shape, padding, stride, activation)

    def _get_deconv_filter(self):
        """
        This function is collected
        :param f_shape: self.filter_shape
        :return: an initializer for get_variable
        """
        width = self.filter_shape[2]
        height = self.filter_shape[3]
        f = int(np.ceil(width/2.))
        c = (2 * f - 1 - f % 2) / (2. * f)
        bilinear = np.zeros([self.filter_shape[2], self.filter_shape[3]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(self.filter_shape)
        for j in range(self.filter_shape[1]):
            for i in range(self.filter_shape[0]):
                weights[i, j, :, :] = bilinear
        return weights.astype(theano.config.floatX)

    def get_output(self, output):
        trans_conv_op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(imshp=self.output_shape,
                                                                       kshp=self.filter_shape, subsample=self.stride,
                                                                       border_mode=self.padding)
        input = trans_conv_op(self.W, output, self.output_shape[-2:])
        input = input + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.activation(input, **self.kwargs)

    @property
    @validate
    def output_shape(self):
        if not any(self.output_shape_tmp):
            if self.padding == 'half':
                p = (self.filter_shape[2] // 2, self.filter_shape[3] // 2)
            elif self.padding == 'valid':
                p = (0, 0)
            elif self.padding == 'full':
                p = (self.filter_shape[2] - 1, self.filter_shape[3] - 1)
            else:
                raise NotImplementedError

            in_shape = self.input_shape
            h = ((in_shape[2] - 1) * self.stride[0]) + self.filter_shape[2] + \
                np.mod(in_shape[2]+2*p[0]-self.filter_shape[2], self.stride[0]) - 2*p[0]
            w = ((in_shape[3] - 1) * self.stride[1]) + self.filter_shape[3] + \
                np.mod(in_shape[3]+2*p[1]-self.filter_shape[3], self.stride[1]) - 2*p[1]
            self.output_shape_tmp = [self.output_shape_tmp[0], self.filter_shape[1], h, w]
        return tuple(self.output_shape_tmp)

    def reset(self):
        self.W.set_value(np.copy(self.W_values))
        self.b.set_value(np.copy(self.b_values))
        if self.activation is utils.function['prelu']:
            self.alpha.set_value(np.float32(.1))


class ResNetBlock(Sequential):
    upscale_factor = 1

    def __init__(self, input_shape, num_filters, stride=(1, 1), dilation=(1, 1), activation='relu', downsample=None,
                 layer_name='ResBlock', normalization='bn', groups=32, block=None, **kwargs):
        """

        :param input_shape:
        :param num_filters:
        :param stride:
        :param dilation:
        :param activation:
        :param downsample:
        :param layer_name:
        :param normalization:
        :param groups:
        :param kwargs:
        """
        assert downsample or (input_shape[1] == num_filters), 'Cannot have identity branch when input dim changes.'
        assert normalization in (None, 'bn', 'gn')
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)
        assert isinstance(stride, (int, list, tuple))

        super(ResNetBlock, self).__init__(input_shape=input_shape, layer_name=layer_name)
        self.num_filters = num_filters
        self.stride = stride if isinstance(stride, (list, tuple)) else (stride, stride)
        self.dilation = dilation
        self.activation = activation
        self.normalization = normalization
        self.downsample = downsample
        self.groups = groups
        self.simple_block = lambda name: block(input_shape=self.input_shape, num_filters=self.num_filters,
                                               stride=self.stride[0], dilation=self.dilation, activation=self.activation,
                                               normalization=self.normalization, block_name=name,
                                               **self.kwargs) if block else self._build_simple_block(name)
        self.kwargs = kwargs

        if activation == 'prelu':
            self.alpha = theano.shared(np.float32(.1), layer_name + '/alpha')
            self.params += [self.alpha]
            self.trainable += [self.alpha]
            self.kwargs['alpha'] = self.alpha

        self.append(Sequential(self.simple_block(layer_name + '_1'), layer_name=layer_name+'/main'))
        if downsample:
            downsample = Sequential(input_shape=input_shape, layer_name=layer_name+'/down')
            downsample.append(ConvolutionalLayer(self.input_shape, num_filters, 1, stride=stride,
                                                 layer_name=layer_name+'/down', activation='linear'))
            if self.normalization:
                downsample.append(BatchNormLayer(downsample[-1].output_shape, layer_name=layer_name + '/down_bn',
                                                 activation='linear') if normalization == 'bn'
                                  else GroupNormLayer(downsample[-1].output_shape, layer_name=layer_name + '/down_gn',
                                                      groups=groups, activation='linear'))
            self.append(downsample)

        self.descriptions = '{} ResNet Basic Block {} -> {} {} filters stride {} dilation {} {} {}'.\
            format(layer_name, self.input_shape, self.output_shape, num_filters, stride, dilation, activation,
                   ' '.join([' '.join((k, str(v))) for k, v in kwargs.items()]))

    def _build_simple_block(self, block_name):
        block = [ConvolutionalLayer(self.input_shape, self.num_filters, 3, border_mode='half', stride=self.stride,
                                    dilation=self.dilation, layer_name=block_name + '/conv1', no_bias=True,
                                    activation='linear')]

        if self.normalization:
            block.append(BatchNormLayer(block[-1].output_shape, activation=self.activation,
                                        layer_name=block_name + '/conv1_bn', **self.kwargs) if self.normalization == 'bn'
                         else GroupNormLayer(block[-1].output_shape, activation=self.activation,
                                             layer_name=block_name+'/conv1_gn', groups=self.groups, **self.kwargs))
        else:
            block.append(ActivationLayer(block[-1].output_shape, self.activation, block_name+'/act1', **self.kwargs))

        block.append(ConvolutionalLayer(block[-1].output_shape, self.num_filters, 3, border_mode='half', dilation=self.dilation,
                                        layer_name=block_name + '/conv2', no_bias=True, activation='linear'))

        if self.normalization:
            block.append(BatchNormLayer(block[-1].output_shape, layer_name=block_name + '/conv2_bn', activation='linear')
                         if self.normalization == 'bn' else GroupNormLayer(block[-1].output_shape, activation='linear',
                                                                           layer_name=block_name + '/conv2_gn', groups=self.groups))
        return block

    def get_output(self, input):
        res = input
        output = self[self.layer_name+'/main'](input)

        if self.downsample:
            res = self[self.layer_name+'/down'](res)
        return utils.function[self.activation](output + res, **self.kwargs)

    def reset(self):
        super(ResNetBlock, self).reset()
        if self.activation == 'prelu':
            self.alpha.set_value(np.float32(.1))


class ResNetBottleneckBlock(Sequential):
    upscale_factor = 4

    def __init__(self, input_shape, num_filters, stride=1, dilation=(1, 1), activation='relu', downsample=False,
                 layer_name='ResBottleneckBlock', normalization='bn', block=None, **kwargs):
        """

        :param input_shape:
        :param num_filters:
        :param stride:
        :param dilation:
        :param activation:
        :param downsample:
        :param upscale_factor:
        :param layer_name:
        :param kwargs:
        """
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(ResNetBottleneckBlock, self).__init__(input_shape=input_shape, layer_name=layer_name)
        self.num_filters = num_filters
        self.stride = stride
        self.dilation = dilation
        self.activation = activation
        self.downsample = downsample
        self.normalization = normalization
        self.simple_block = lambda name: block(input_shape=self.input_shape, num_filters=self.num_filters,
                                               stride=self.stride, dilation=self.dilation, activation=self.activation,
                                               normalization=self.normalization, block_name=name,
                                               **self.kwargs) if block else self._build_simple_block(name)
        self.kwargs = kwargs

        self.append(Sequential(self.simple_block(layer_name + '_1'), layer_name=layer_name + '/main'))
        if downsample:
            self.append(ConvNormAct(self.input_shape, num_filters * self.upscale_factor, 1, stride=stride,
                                    layer_name=layer_name+'/down', activation='linear', **self.kwargs))

        if activation == 'prelu':
            self.alpha = theano.shared(np.float32(.1), layer_name + '/alpha')
            self.params += [self.alpha]
            self.trainable += [self.alpha]
            self.kwargs['alpha'] = self.alpha

        self.descriptions = '{} ResNet Bottleneck {} -> {} num filters {} stride {} dilation {} {}'. \
            format(layer_name, input_shape, self.output_shape, num_filters, stride, dilation, activation)

    def _build_simple_block(self, block_name):
        layers = []
        layers.append(ConvNormAct(self.input_shape, self.num_filters, 1, no_bias=True, activation=self.activation,
                                  layer_name=block_name+'/conv_bn_act_1', **self.kwargs))

        layers.append(ConvNormAct(layers[-1].output_shape, self.num_filters, 3, activation=self.activation,
                                  stride=self.stride, layer_name=block_name+'/conv_bn_act_2', no_bias=True, **self.kwargs))

        layers.append(ConvNormAct(layers[-1].output_shape, self.num_filters * self.upscale_factor, 1, activation='linear',
                                  layer_name=block_name+'/conv_bn_act_3', no_bias=True, **self.kwargs))
        return layers

    def get_output(self, input):
        res = input
        output = self[self.layer_name+'/main'](input)

        if self.downsample:
            res = self[self.layer_name+'/down'](res)
        return utils.function[self.activation](output + res, **self.kwargs)

    def reset(self):
        super(ResNetBottleneckBlock, self).reset()
        if self.activation == 'prelu':
            self.alpha.set_value(np.float32(.1))


class NoiseResNetBlock(Layer):
    def __init__(self, input_shape, num_filters, noise_level=.1, activation='relu', left_branch=False,
                 layer_name='NoiseResBlock', normalization='bn', groups=32, **kwargs):
        """

        :param input_shape:
        :param num_filters:
        :param noise_level:
        :param activation:
        :param left_branch:
        :param layer_name:
        :param normalization:
        :param groups:
        :param kwargs:
        """
        assert left_branch or (input_shape[1] == num_filters), 'Cannot have identity branch when input dim changes.'
        assert normalization in (None, 'bn', 'gn')
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(NoiseResNetBlock, self).__init__(input_shape, layer_name)
        self.num_filters = num_filters
        self.noise_level = noise_level
        self.activation = activation
        self.left_branch = left_branch
        self.normalization = normalization
        self.groups = groups
        self.kwargs = kwargs

        if activation == 'prelu':
            self.alpha = theano.shared(np.float32(.1), layer_name + '/alpha')
            self.params += [self.alpha]
            self.trainable += [self.alpha]
            self.kwargs['alpha'] = self.alpha

        self.block = list(self._build_simple_block(block_name=layer_name + '/1'))
        self.params += [p for layer in self.block for p in layer.params]
        self.trainable += [p for layer in self.block for p in layer.trainable]
        self.regularizable += [p for layer in self.block for p in layer.regularizable]

        if self.left_branch:
            self.shortcut = []
            self.shortcut.append(PerturbativeLayer(self.input_shape, self.num_filters, noise_level=self.noise_level,
                                                   activation=self.activation, no_bias=True, layer_name=layer_name+'/2'))
            if self.normalization:
                self.shortcut.append(BatchNormLayer(self.shortcut[-1].output_shape, layer_name=layer_name + '/2_bn',
                                                    activation='linear') if normalization == 'bn'
                                     else GroupNormLayer(self.shortcut[-1].output_shape, layer_name=layer_name + '/2_gn',
                                                         groups=groups, activation='linear'))
            self.params += [p for layer in self.shortcut for p in layer.params]
            self.trainable += [p for layer in self.shortcut for p in layer.trainable]
            self.regularizable += [p for layer in self.shortcut for p in layer.regularizable]

        self.descriptions = '{} ResNet Block 1 {} -> {} {} filters left branch {} {} {}'.\
            format(layer_name, self.input_shape, self.output_shape, num_filters, left_branch, activation, ' '.
                   join([' '.join((k, str(v))) for k, v in kwargs.items()]))

    def _build_simple_block(self, block_name):
        block = [PerturbativeLayer(self.input_shape, self.num_filters, noise_level=self.noise_level,
                                   activation=self.activation, no_bias=True, layer_name=self.layer_name+'_noise1')]

        if self.normalization:
            block.append(BatchNormLayer(block[-1].output_shape, activation='linear',
                                        layer_name=block_name + '/noise1_bn', **self.kwargs) if self.normalization == 'bn'
                         else GroupNormLayer(block[-1].output_shape, activation='linear',
                                             layer_name=block_name+'/noise1_gn', groups=self.groups, **self.kwargs))

        block.append(PerturbativeLayer(block[-1].output_shape, self.num_filters, noise_level=self.noise_level,
                                       activation=self.activation, no_bias=True, layer_name=self.layer_name+'/noise2'))

        if self.normalization:
            block.append(BatchNormLayer(block[-1].output_shape, layer_name=block_name + '/noise2_bn', activation='linear')
                         if self.normalization == 'bn' else GroupNormLayer(block[-1].output_shape, activation='linear',
                                                                           layer_name=block_name + '/noise2_gn', groups=self.groups))
        return block

    def get_output(self, input):
        output = input
        for layer in self.block:
            output = layer(output)

        res = input
        if self.left_branch:
            for layer in self.shortcut:
                res = layer(res)
        return utils.function[self.activation](output + res, **self.kwargs)

    @property
    @validate
    def output_shape(self):
        return self.block[-1].output_shape

    def reset(self):
        for layer in self.block:
            layer.reset()

        if self.left_branch:
            for layer in self.shortcut:
                layer.reset()

        if self.activation == 'prelu':
            self.alpha.set_value(np.float32(.1))


class RecursiveResNetBlock(Layer):
    def __init__(self, input_shape, num_filters, filter_size, recursive=1, stride=(1, 1), dilation=(1, 1), activation='relu',
                 layer_name='RecursiveResBlock', normalization='bn', groups=32, **kwargs):
        assert normalization in (
        None, 'bn', 'gn'), 'normalization must be either None, \'bn\' or \'gn\', got %s.' % normalization
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(RecursiveResNetBlock, self).__init__(input_shape, layer_name)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.recursive = recursive
        self.stride = stride
        self.dilation = dilation
        self.activation = activation
        self.kwargs = kwargs

        if activation == 'prelu':
            self.alpha = theano.shared(np.float32(.1), layer_name + '/alpha')
            self.params += [self.alpha]
            self.trainable += [self.alpha]
            self.kwargs['alpha'] = self.alpha

        self.first_conv = ConvolutionalLayer(input_shape, num_filters, filter_size, stride=stride,
                                             layer_name=layer_name+'/first_conv', activation='linear')
        if normalization:
            self.normalization = partial(BatchNormLayer, activation='linear') \
                if normalization == 'bn' else partial(GroupNormLayer, groups=groups, activation='linear')
        else:
            self.normalization = None

        self.block = Sequential(self._recursive_block())
        self.params += self.first_conv.params + self.block.params
        self.trainable += self.first_conv.trainable + self.block.trainable
        self.regularizable += self.first_conv.regularizable + self.block.regularizable
        self.descriptions = '{} Recursive Residual Block: {} -> {} stride {} activation {}'.\
            format(layer_name, input_shape, self.output_shape, stride, activation)

    def _recursive_block(self):
        shape = self.first_conv.output_shape
        block = []
        if self.normalization:
            block.append(self.normalization(shape, self.layer_name+'/norm1'))
        block += [
            ActivationLayer(shape, self.activation, self.layer_name + '/act1', **self.kwargs),
            ConvolutionalLayer(shape, self.num_filters, self.filter_size, layer_name=self.layer_name+'/conv1',
                               activation='linear')
            ]
        if self.normalization:
            block.append(self.normalization(shape, self.layer_name+'/norm2'))
        block += [
            ActivationLayer(shape, self.activation, self.layer_name + '/act2', **self.kwargs),
            ConvolutionalLayer(shape, self.num_filters, self.filter_size, layer_name=self.layer_name + '/conv2',
                               activation='linear')
        ]
        return block

    def get_output(self, input):
        input = self.first_conv(input)
        first = input
        input = self.block(input) + first

        def step(x, *args):
            x = self.block(x)
            return x + first

        if self.recursive > 1:
            unroll = self.kwargs.pop('unroll', False)
            non_seqs = list(self.params) + [first]
            if unroll or isinstance(self.normalization, BatchNormLayer):
                output = utils.unroll_scan(step, None, input, non_seqs, self.recursive - 1)
            else:
                output = theano.scan(step, outputs_info=input, non_sequences=non_seqs, n_steps=self.recursive - 1, strict=True)[0]
            output = output[-1]
        else:
            output = input
        return utils.function[self.activation](output, **self.kwargs)

    @property
    def output_shape(self):
        return self.block.output_shape

    def reset(self):
        self.first_conv.reset()
        self.block.reset()


class DenseBlock(Layer):
    def __init__(self, input_shape, transit=False, num_conv_layer=6, growth_rate=32, dropout=False, activation='relu',
                 layer_name='DenseBlock', pool_transition=True, normlization='bn', target='dev0', **kwargs):
        """

        :param input_shape:
        :param transit:
        :param num_conv_layer:
        :param growth_rate:
        :param dropout:
        :param activation:
        :param layer_name:
        :param pool_transition:
        :param target:
        :param kwargs:
        """
        assert normlization in ('bn', 'gn'), \
            'normalization should be either \'bn\' or \'gn\', got %s' % normlization
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(DenseBlock, self).__init__(tuple(input_shape), layer_name)
        self.transit = transit
        self.num_conv_layer = num_conv_layer
        self.growth_rate = growth_rate
        self.activation = activation
        self.dropout = dropout
        self.pool_transition = pool_transition
        self.target = target
        self.normalization = BatchNormLayer if normlization == 'bn' else GroupNormLayer
        self.kwargs = kwargs

        if not self.transit:
            self.block = self._dense_block(self.input_shape, self.num_conv_layer, self.growth_rate, self.dropout,
                                           self.activation, self.layer_name)
        else:
            self.block = self._transition(self.input_shape, self.dropout, self.activation,
                                          self.layer_name + '/transition')

        self.descriptions = '{} Dense Block: {} -> {} {} conv layers growth rate {} transit {} dropout {} {}'.\
            format(layer_name, input_shape, self.output_shape, num_conv_layer, growth_rate, transit, dropout, activation)

    def _bn_act_conv(self, input_shape, num_filters, filter_size, dropout, activation, stride=1, layer_name='bn_re_conv'):
        block = [
            self.normalization(input_shape, activation=activation, layer_name=layer_name + '/bn', **self.kwargs),
            ConvolutionalLayer(input_shape, num_filters, filter_size, stride=stride, activation='linear',
                               layer_name=layer_name + '/conv')
        ]
        for layer in block:
            self.params += layer.params
            self.trainable += layer.trainable
            self.regularizable += layer.regularizable
        if dropout:
            block.append(DropoutLayer(block[-1].output_shape, dropout, layer_name=layer_name + 'dropout'))
        return block

    def _transition(self, input_shape, dropout, activation, layer_name='transition'):
        if self.pool_transition:
            block = self._bn_act_conv(input_shape, input_shape[1], 1, dropout, activation, layer_name=layer_name)
            block.append(PoolingLayer(block[-1].output_shape, (2, 2), mode='average_inc_pad', ignore_border=False,
                                      layer_name=layer_name + 'pooling'))
        else:
            block = self._bn_act_conv(input_shape, input_shape[1], 1, dropout, activation, stride=2, layer_name=layer_name)
        return block

    def _dense_block(self, input_shape, num_layers, growth_rate, dropout, activation, layer_name='dense_block'):
        block, input_channels = [], input_shape[1]
        i_shape = list(input_shape)
        for n in range(num_layers):
            block.append(self._bn_act_conv(i_shape, growth_rate, 3, dropout, activation, layer_name=layer_name + '_%d' % n))
            input_channels += growth_rate
            i_shape[1] = input_channels
        return block

    def get_output(self, input):
        feed = input
        if not self.transit:
            for layer in self.block:
                output = utils.inference(feed, layer)
                feed = T.concatenate((feed, output), 1)
        else:
            feed = utils.inference(feed, self.block)
        return feed

    @property
    @validate
    def output_shape(self):
        if not self.transit:
            shape = (self.input_shape[0], self.input_shape[1] + self.growth_rate * self.num_conv_layer,
                     self.input_shape[2], self.input_shape[3])
        else:
            shape = self.block[-1].output_shape
        return tuple(shape)

    def reset(self):
        for layers in self.block:
            if isinstance(layers, Layer):
                layers.reset()
            else:
                for layer in layers:
                    layer.reset()


class BatchNormLayer(Layer):
    def __init__(self, input_shape, layer_name='BN', epsilon=1e-4, running_average_factor=1e-1, axes='spatial',
                 activation='relu', no_scale=False, **kwargs):
        '''

        :param input_shape: (int, int, int, int) or (int, int)
        :param layer_name: str
        :param epsilon: float
        :param running_average_factor: float
        :param axes: 'spatial' or 'per-activation'
        '''
        super(BatchNormLayer, self).__init__(input_shape, layer_name)
        self.epsilon = np.float32(epsilon)
        self.running_average_factor = running_average_factor
        self.activation = utils.function[activation]
        self.no_scale = no_scale
        self.axes = (0,) + tuple(range(2, len(input_shape))) if axes == 'spatial' else (0,)
        self.shape = (self.input_shape[1],) if axes == 'spatial' else self.input_shape[1:]
        self.kwargs = kwargs

        self.gamma_values = np.ones(self.shape, dtype=theano.config.floatX)
        self.gamma = theano.shared(np.copy(self.gamma_values), name=layer_name + '/gamma', borrow=True)

        self.beta_values = np.zeros(self.shape, dtype=theano.config.floatX)
        self.beta = theano.shared(np.copy(self.beta_values), name=layer_name + '/beta', borrow=True)

        self.running_mean = theano.shared(np.zeros(self.shape, dtype=theano.config.floatX),
                                          name=layer_name + '/running_mean', borrow=True)
        self.running_var = theano.shared(np.zeros(self.shape, dtype=theano.config.floatX),
                                         name=layer_name + '/running_var', borrow=True)

        self.params += [self.gamma, self.beta, self.running_mean, self.running_var]
        self.trainable += [self.beta] if self. no_scale else [self.beta, self.gamma]
        self.regularizable += [self.gamma] if not self.no_scale else []

        if activation == 'prelu':
            self.alpha = theano.shared(np.float32(.1), layer_name + '/alpha')
            self.params += [self.alpha]
            self.trainable += [self.alpha]
            self.kwargs['alpha'] = self.alpha

        self.descriptions = '{} BatchNorm Layer: shape: {} -> {} running_average_factor = {:.4f} activation: {}'\
            .format(layer_name, self.input_shape, self.output_shape, self.running_average_factor, activation)

    def batch_normalization_train(self, input):
        out, _, _, mean_, var_ = T.nnet.bn.batch_normalization_train(input, self.gamma, self.beta, self.axes,
                                                                     self.epsilon, self.running_average_factor,
                                                                     self.running_mean, self.running_var)

        # Update running mean and variance
        # Tricks adopted from Lasagne implementation
        # http://lasagne.readthedocs.io/en/latest/modules/layers/normalization.html
        running_mean = theano.clone(self.running_mean, share_inputs=False)
        running_var = theano.clone(self.running_var, share_inputs=False)
        running_mean.default_update = mean_
        running_var.default_update = var_
        out += 0 * (running_mean + running_var)
        return out

    def batch_normalization_test(self, input):
        out = T.nnet.bn.batch_normalization_test(input, self.gamma, self.beta, self.running_mean, self.running_var,
                                                 axes=self.axes, epsilon=self.epsilon)
        return out

    def get_output(self, input):
        return self.activation(self.batch_normalization_train(input) if self.training_flag
                               else self.batch_normalization_test(input), **self.kwargs)

    @property
    @validate
    def output_shape(self):
        return tuple(self.input_shape)

    def reset(self):
        self.gamma.set_value(np.copy(self.gamma_values))
        self.beta.set_value(np.copy(self.beta_values))
        if self.activation is utils.function['prelu']:
            self.alpha.set_value(np.float32(.1))


class DecorrBatchNormLayer(Layer):
    """
    From the paper "Decorrelated Batch Normalization" - Lei Huang, Dawei Yang, Bo Lang, Jia Deng
    """
    def __init__(self, input_shape, layer_name='DBN', epsilon=1e-4, running_average_factor=1e-1, activation='relu',
                 no_scale=False, **kwargs):
        """

        :param input_shape:
        :param layer_name:
        :param epsilon:
        :param running_average_factor:
        :param activation:
        :param no_scale:
        :param kwargs:
        """
        super(DecorrBatchNormLayer, self).__init__(input_shape, layer_name)
        self.epsilon = np.float32(epsilon)
        self.running_average_factor = running_average_factor
        self.activation = utils.function[activation]
        self.no_scale = no_scale
        self.axes = (0,) #+ tuple(range(2, len(input_shape)))
        self.shape = (self.input_shape[1],)
        self.kwargs = kwargs

        self.gamma_values = np.ones(self.shape, dtype=theano.config.floatX)
        self.gamma = theano.shared(np.copy(self.gamma_values), name=layer_name + '/gamma', borrow=True)

        self.beta_values = np.zeros(self.shape, dtype=theano.config.floatX)
        self.beta = theano.shared(np.copy(self.beta_values), name=layer_name + '/beta', borrow=True)

        self.running_mean = theano.shared(np.zeros(self.shape, dtype=theano.config.floatX),
                                          name=layer_name + '/running_mean', borrow=True)
        self.running_var = theano.shared(np.zeros(self.shape, dtype=theano.config.floatX),
                                         name=layer_name + '/running_var', borrow=True)

        self.params += [self.gamma, self.beta, self.running_mean, self.running_var]
        self.trainable += [self.beta] if self. no_scale else [self.beta, self.gamma]
        self.regularizable += [self.gamma] if not self.no_scale else []

        self.descriptions = '{} DecorrelatedBatchNorm Layer: shape: {} -> {} running_average_factor = {:.4f} activation: {}'\
            .format(layer_name, self.input_shape, self.output_shape, self.running_average_factor, activation)

    def batch_normalization_train(self, input):
        out, _, _, mean_, var_ = T.nnet.bn.batch_normalization_train(input, self.gamma, self.beta, self.axes,
                                                                     self.epsilon, self.running_average_factor,
                                                                     self.running_mean, self.running_var)

        # Update running mean and variance
        # Tricks adopted from Lasagne implementation
        # http://lasagne.readthedocs.io/en/latest/modules/layers/normalization.html
        running_mean = theano.clone(self.running_mean, share_inputs=False)
        running_var = theano.clone(self.running_var, share_inputs=False)
        running_mean.default_update = mean_
        running_var.default_update = var_
        out += 0 * (running_mean + running_var)
        return out

    def batch_normalization_test(self, input):
        out = T.nnet.bn.batch_normalization_test(input, self.gamma, self.beta, self.running_mean, self.running_var,
                                                 axes=self.axes, epsilon=self.epsilon)
        return out

    def get_output(self, input):
        m, c, h, w = T.shape(input)
        X = input.dimshuffle((1, 0, 2, 3))
        X = X.flatten(2)
        Muy = T.mean(X, axis=1)
        X_centered = X - Muy
        Sigma = 1. / m * T.dot(X_centered, X_centered.T)
        D, Lambda, _ = T.nlinalg.svd(Sigma)
        Z = T.dot(T.dot(D, T.nlinalg.diag(T.sqrt(T.nlinalg.diag(Lambda)))), D.T)
        X = T.dot(Z, X)
        out = self.activation(self.batch_normalization_train(X.T) if self.training_flag
                               else self.batch_normalization_test(X.T), **self.kwargs)
        out = T.reshape(out.T, (c, m, h, w))
        out = out.dimshuffle((1, 0, 2, 3))
        return out

    @property
    @validate
    def output_shape(self):
        return tuple(self.input_shape)

    def reset(self):
        self.gamma.set_value(np.copy(self.gamma_values))
        self.beta.set_value(np.copy(self.beta_values))


class GroupNormLayer(Layer):
    """
    Implementation of the paper "Group Normalization" - Wu et al.
    group = 1 -> Layer Normalization
    group = input_shape[1] -> Instance Normalization
    """
    def __init__(self, input_shape, layer_name='GN', groups=32, epsilon=1e-4, activation='relu', **kwargs):
        assert input_shape[1] / groups == input_shape[1] // groups, 'groups must divide the number of input channels.'

        super(GroupNormLayer, self).__init__(tuple(input_shape), layer_name)
        self.groups = groups
        self.epsilon = np.float32(epsilon)
        self.activation = utils.function[activation]
        self.kwargs = kwargs
        self.gamma_values = np.ones(self.input_shape[1], dtype=theano.config.floatX)
        self.gamma = theano.shared(np.copy(self.gamma_values), name=layer_name + '/gamma', borrow=True)

        self.beta_values = np.zeros(self.input_shape[1], dtype=theano.config.floatX)
        self.beta = theano.shared(np.copy(self.beta_values), name=layer_name + '/beta', borrow=True)

        self.params += [self.gamma, self.beta]
        self.trainable += [self.gamma, self.beta]
        self.regularizable += [self.gamma]

        if activation == 'prelu':
            self.alpha = theano.shared(np.float32(.1), layer_name + '/alpha')
            self.params += [self.alpha]
            self.trainable += [self.alpha]
            self.kwargs['alpha'] = self.alpha

        self.descriptions = '{} GroupNorm Layer: shape: {} -> {} activation: {}'\
            .format(layer_name, self.input_shape, self.output_shape, activation)

    def get_output(self, input):
        gamma = self.gamma.dimshuffle('x', 0, 'x', 'x')
        beta = self.beta.dimshuffle('x', 0, 'x', 'x')
        if self.groups == 1:
            input_ = input.dimshuffle(1, 0, 2, 3)
            ones = T.ones_like(T.mean(input_, (0, 2, 3), keepdims=True), 'float32')
            zeros = T.zeros_like(T.mean(input_, (0, 2, 3), keepdims=True), 'float32')
            output, _, _ = T.nnet.bn.batch_normalization_train(input_, ones, zeros, 'spatial', self.epsilon)
            output = gamma * output.dimshuffle(1, 0, 2, 3) + beta
        elif self.groups == self.input_shape[1]:
            output, _, _ = T.nnet.bn.batch_normalization_train(input, gamma, beta, (2, 3))
        else:
            n, c, h, w = T.shape(input)
            input_ = T.reshape(input, (n, self.groups, -1, h, w))
            mean = T.mean(input_, (2, 3, 4), keepdims=True)
            var = T.var(input_, (2, 3, 4), keepdims=True)
            input_ = (input_ - mean) / T.sqrt(var + self.epsilon)
            input_ = T.reshape(input_, (n, c, h, w))
            output = gamma * input_ + beta
        return self.activation(output, **self.kwargs)

    @property
    def output_shape(self):
        return tuple(self.input_shape)

    def reset(self):
        self.gamma.set_value(np.copy(self.gamma_values))
        self.beta.set_value(np.copy(self.beta_values))
        if self.activation is utils.function['prelu']:
            self.alpha.set_value(np.float32(.1))


class BatchRenormLayer(Layer):
    def __init__(self, input_shape, layer_name='BRN', epsilon=1e-4, r_max=1, d_max=0, running_average_factor=0.1,
                 axes='spatial', activation='relu'):
        '''

        :param input_shape: (int, int, int, int) or (int, int)
        :param layer_name: str
        :param epsilon: float
        :param running_average_factor: float
        :param axes: 'spatial' or 'per-activation'
        '''
        super(BatchRenormLayer, self).__init__(tuple(input_shape), layer_name)
        self.epsilon = epsilon
        self.running_average_factor = running_average_factor
        self.activation = utils.function[activation]
        self.r_max = theano.shared(np.float32(r_max), name=layer_name + 'rmax')
        self.d_max = theano.shared(np.float32(d_max), name=layer_name + 'dmax')
        self.axes = (0,) + tuple(range(2, len(input_shape))) if axes == 'spatial' else (0,)
        self.shape = (self.input_shape[1],) if axes == 'spatial' else self.input_shape[1:]

        self.gamma_values = np.ones(self.shape, dtype=theano.config.floatX)
        self.beta_values = np.zeros(self.shape, dtype=theano.config.floatX)
        self.gamma = theano.shared(self.gamma_values, name=layer_name + '/gamma', borrow=True)
        self.beta = theano.shared(self.beta_values, name=layer_name + '/beta', borrow=True)
        self.running_mean = theano.shared(np.zeros(self.shape, dtype=theano.config.floatX),
                                          name=layer_name + '/running_mean', borrow=True)
        self.running_var = theano.shared(np.zeros(self.shape, dtype=theano.config.floatX),
                                         name=layer_name + '/running_var', borrow=True)
        self.params += [self.gamma, self.beta, self.running_mean, self.running_var]
        self.trainable += [self.gamma, self.beta]
        self.regularizable.append(self.gamma)
        self.descriptions = '{} Batch Renorm Layer: running_average_factor = {:.4f}'.format(layer_name, self.running_average_factor)

    def get_output(self, input):
        batch_mean = T.mean(input, axis=self.axes)
        batch_std = T.sqrt(T.var(input, axis=self.axes) + 1e-10)
        r = T.clip(batch_std / T.sqrt(self.running_var + 1e-10), -self.r_max, self.r_max)
        d = T.clip((batch_mean - self.running_mean) / T.sqrt(self.running_var + 1e-10), -self.d_max, self.d_max)
        out = T.nnet.bn.batch_normalization_test(input, self.gamma, self.beta, batch_mean - d * batch_std / (r + 1e-10),
                                                 T.sqr(batch_std / (r + 1e-10)), axes=self.axes, epsilon=self.epsilon)
        if self.training_flag:
            # Update running mean and variance
            # Tricks adopted from Lasagne implementation
            # http://lasagne.readthedocs.io/en/latest/modules/layers/normalization.html
            m = T.cast(T.prod(input.shape) / T.prod(self.gamma.shape), 'float32')
            running_mean = theano.clone(self.running_mean, share_inputs=False)
            running_var = theano.clone(self.running_var, share_inputs=False)
            running_mean.default_update = running_mean + self.running_average_factor * (batch_mean - running_mean)
            running_var.default_update = running_var * (1. - self.running_average_factor) + \
                                         self.running_average_factor * (m / (m - 1)) * T.sqr(batch_std)
            out += 0 * (running_mean + running_var)
        return self.activation(out)

    @property
    @validate
    def output_shape(self):
        return tuple(self.input_shape)

    def reset(self):
        self.gamma.set_value(self.gamma_values)
        self.beta.set_value(self.beta_values)


class TransformerLayer(Layer):
    """Implementation of the bilinear interpolation transformer layer in https://arxiv.org/abs/1506.020250. Based on
    the implementation in Lasagne.
    coordinates is a tensor of shape (n, 2, h, w). coordinates[:, 0, :, :] is the horizontal coordinates and the other
    is the vertical coordinates.
    """

    def __init__(self, input_shape, transform_shape, downsample_factor=1, border_mode='nearest', layer_name='Transformer', **kwargs):
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(TransformerLayer, self).__init__(tuple(input_shape), layer_name)
        self.transform_shape = tuple(transform_shape)
        self.downsample_factor = (downsample_factor, downsample_factor) if isinstance(downsample_factor, int) else tuple(downsample_factor)
        self.border_mode = border_mode
        self.kwargs = kwargs
        self.descriptions = '%s Transformer layer.' % layer_name

    @property
    def output_shape(self):
        shape = self.input_shape
        factors = self.downsample_factor
        return tuple(list(shape[:2]) + [None if s is None else int(s // f) for s, f in zip(shape[2:], factors)])

    def get_output(self, inputs):
        input, theta = inputs
        return utils.transform_affine(theta, input, self.downsample_factor, self.border_mode)


class WarpingLayer(Layer):
    def __init__(self, input_shape, border_mode='nearest', layer_name='WarpingLayer'):
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(WarpingLayer, self).__init__(input_shape, layer_name)
        self.border_mode = border_mode
        self.layer_name = layer_name
        self.descriptions = '{} Warping layer: bilinear interpolation border mode: {}'.format(layer_name, border_mode)

    def get_output(self, input):
        image, flow = input

        def meshgrid(height, width):
            x_t = T.dot(T.ones(shape=[height, 1]),
                        T.transpose(
                            T.as_tensor_variable(np.linspace(-1.0, 1.0, width, dtype='float32')).dimshuffle(0, 'x'),
                            [1, 0]))
            y_t = T.dot(T.as_tensor_variable(np.linspace(-1.0, 1.0, height, dtype='float32')).dimshuffle(0, 'x'),
                        T.ones(shape=[1, width]))
            x_t_flat = T.reshape(x_t, (1, -1))
            y_t_flat = T.reshape(y_t, (1, -1))
            grid_x = T.reshape(x_t_flat, [1, height, width])
            grid_y = T.reshape(y_t_flat, [1, height, width])
            return grid_x, grid_y

        gx, gy = meshgrid(self.input_shape[2], self.input_shape[3])
        gx = T.as_tensor_variable(gx, ndim=2).astype('float32').dimshuffle('x', 0, 1)
        gy = T.as_tensor_variable(gy, ndim=2).astype('float32').dimshuffle('x', 0, 1)
        x_coor = gx + flow[:, 0]
        y_coor = gy + flow[:, 1]
        output = utils.interpolate_bilinear(image, x_coor, y_coor, border_mode=self.border_mode)
        return output

    @property
    @validate
    def output_shape(self):
        return self.input_shape


class IdentityLayer(Layer):
    def __init__(self, input_shape, layer_name='Identity'):
        super(IdentityLayer, self).__init__(tuple(input_shape), layer_name)
        self.descriptions = '%s Identity layer.' % layer_name

    @property
    def output_shape(self):
        return tuple(self.input_shape)

    def get_output(self, input):
        return input


class ResizingLayer(Layer):
    def __init__(self, input_shape, ratio=None, frac_ratio=None, layer_name='Upsampling'):
        if ratio != int(ratio):
            raise NotImplementedError
        if ratio and frac_ratio:
            raise NotImplementedError
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(ResizingLayer, self).__init__(tuple(input_shape), layer_name)
        self.ratio = ratio
        self.frac_ratio = frac_ratio
        self.descriptions = '{} x{} Resizing Layer {} -> {}'.format(layer_name, self.ratio, self.input_shape, self.output_shape)

    def get_output(self, input):
        return T.nnet.abstract_conv.bilinear_upsampling(input, ratio=self.ratio) if self.ratio \
            else T.nnet.abstract_conv.bilinear_upsampling(input, frac_ratio=self.frac_ratio)

    @property
    @validate
    def output_shape(self):
        return (self.input_shape[0], self.input_shape[1], self.input_shape[2] * self.ratio, self.input_shape[3] * self.ratio)


class ReshapingLayer(Layer):
    def __init__(self, input_shape, new_shape, layer_name='reshape'):
        super(ReshapingLayer, self).__init__(tuple(input_shape), layer_name)
        self.new_shape = tuple(new_shape)
        self.descriptions = 'Reshaping Layer: {} -> {}'.format(self.input_shape, self.output_shape)

    def get_output(self, input):
        return T.reshape(input, self.new_shape)

    @property
    @validate
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


class SlicingLayer(Layer):
    def __init__(self, input_shape, to_idx, from_idx=(0, 0), axes=(2, 3), layer_name='Slicing Layer'):
        '''

        :param input_shape: (int, int, int, int)
        :param to_idx:
        :param from_idx:
        :param axes:
        :param layer_name:
        '''
        assert isinstance(to_idx, (int, list, tuple)) and isinstance(to_idx, (int, list, tuple)) and isinstance(to_idx, (int, list, tuple))
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(SlicingLayer, self).__init__(tuple(input_shape), layer_name)
        self.from_idx = from_idx
        self.to_idx = to_idx
        self.axes = axes
        self.descriptions = '{} Slicing Layer: {} slice from {} to {} at {} -> {}'.format(layer_name, input_shape,
                                                                                          from_idx, to_idx, axes, self.output_shape)

    def get_output(self, input):
        assign = dict({0: lambda x, fr, to: x[fr:to], 1: lambda x, fr, to: x[:, fr:to],
                       2: lambda x, fr, to: x[:, :, fr:to], 3: lambda x, fr, to: x[:, :, :, fr:to]})
        if isinstance(self.from_idx, (tuple, list)) and isinstance(self.to_idx, (tuple, list)) \
                and isinstance(self.axes, (tuple, list)):
            assert len(self.from_idx) == len(self.to_idx) == len(self.axes), \
                "Numbers of elements in from_idx, to_idx, and axes must match"
            output = input
            for idx, axis in enumerate(self.axes):
                output = assign[axis](output, self.from_idx[idx], self.to_idx[idx])
        elif isinstance(self.from_idx, int) and isinstance(self.to_idx, int) and isinstance(self.axes, int):
            output = assign[self.axes](input, self.from_idx, self.to_idx)
        else:
            raise NotImplementedError
        return output

    @property
    @validate
    def output_shape(self):
        shape = list(self.input_shape)
        for idx, axis in enumerate(self.axes):
            shape[axis] = self.to_idx[idx] - self.from_idx[idx]
        return shape


class ConcatLayer(Layer):
    def __init__(self, input_shapes, axis=1, layer_name='ConcatLayer'):
        super(ConcatLayer, self).__init__(input_shapes, layer_name=layer_name)
        self.axis = axis
        self.descriptions = ''.join(('%s Concat Layer: axis %d' % (layer_name, axis), ' '.join([str(x) for x in input_shapes]),
                                     ' -> {}'.format(self.output_shape)))

    def get_output(self, input):
        return T.concatenate(input, self.axis)

    @property
    def output_shape(self):
        depth = sum([self.input_shape[i][self.axis] for i in range(len(self.input_shape))])
        shape = list(self.input_shape[0])
        shape[self.axis] = depth
        return tuple(shape)


class SumLayer(Layer):
    def __init__(self, input_shape, weight=1., layer_name='SumLayer'):
        super(SumLayer, self).__init__(input_shape, layer_name)
        self.weight = weight
        self.descriptions = '{} Sum Layer: weight {}'.format(layer_name, weight)

    def get_output(self, input):
        assert isinstance(input, (list, tuple)), 'Input must be a list or tuple of same-sized tensors.'
        return sum(input) * np.float32(self.weight)

    @property
    def output_shape(self):
        return self.input_shape


class TransposingLayer(Layer):
    def __init__(self, input_shape, transpose, layer_name='TransposeLayer'):
        super(TransposingLayer, self).__init__(input_shape, layer_name)
        self.transpose = transpose
        self.descriptions = '{} Transposing layer: {} -> {}'.format(layer_name, [i for i in range(len(input_shape))], transpose)

    def get_output(self, input):
        return T.transpose(input, self.transpose)

    @property
    @validate
    def output_shape(self):
        return tuple([self.input_shape[i] for i in self.transpose])


class ScalingLayer(Layer):
    """
    Adapted from Lasagne
    lasagne.layers.ScaleLayer(incoming, scales=lasagne.init.Constant(1),
    shared_axes='auto', **kwargs)

    A layer that scales its inputs by learned coefficients.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    scales : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for the scale.  The scale
        shape must match the incoming shape, skipping those axes the scales are
        shared over (see the example below).  See
        :func:`lasagne.utils.create_param` for more information.

    shared_axes : 'auto', int or tuple of int
        The axis or axes to share scales over. If ``'auto'`` (the default),
        share over all axes except for the second: this will share scales over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.

    Notes
    -----
    The scales parameter dimensionality is the input dimensionality minus the
    number of axes the scales are shared over, which matches the bias parameter
    conventions of :class:`DenseLayer` or :class:`Conv2DLayer`. For example:

    >>> layer = ScalingLayer((20, 30, 40, 50), shared_axes=(0, 2))
    >>> layer.scales.get_value().shape
    (30, 50)
    """
    def __init__(self, input_shape, scales=1, shared_axes='auto', layer_name='ScaleLayer'):
        super(ScalingLayer, self).__init__(input_shape, layer_name)
        if shared_axes == 'auto':
            # default: share scales over all but the second axis
            shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(shared_axes, int):
            shared_axes = (shared_axes,)
        self.shared_axes = shared_axes
        # create scales parameter, ignoring all dimensions in shared_axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.shared_axes]
        if any(size is None for size in shape):
            raise ValueError("ScaleLayer needs specified input sizes for "
                             "all axes that scales are not shared over.")
        self.scales = theano.shared(scales * np.ones(shape, theano.config.floatX), name=layer_name + 'scales', borrow=True)
        self.params.append(self.scales)
        self.trainable.append(self.scales)
        self.descriptions = '{} ScaleLayer: scales = {}'.format(layer_name, scales)

    def get_output(self, input):
        axes = iter(list(range(self.scales.ndim)))
        pattern = ['x' if input_axis in self.shared_axes
                   else next(axes) for input_axis in range(input.ndim)]
        return input * self.scales.dimshuffle(*pattern)

    @property
    @validate
    def output_shape(self):
        return self.input_shape


class Gate:
    """
    lasagne.layers.recurrent.Gate(W_in=lasagne.init.Normal(0.1),
    W_hid=lasagne.init.Normal(0.1), W_cell=lasagne.init.Normal(0.1),
    b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.sigmoid)
    Simple class to hold the parameters for a gate connection.  We define
    a gate loosely as something which computes the linear mix of two inputs,
    optionally computes an element-wise product with a third, adds a bias, and
    applies a nonlinearity.
    Parameters
    ----------
    W_in : Theano shared variable, numpy array or callable
        Initializer for input-to-gate weight matrix.
    W_hid : Theano shared variable, numpy array or callable
        Initializer for hidden-to-gate weight matrix.
    W_cell : Theano shared variable, numpy array, callable, or None
        Initializer for cell-to-gate weight vector.  If None, no cell-to-gate
        weight vector will be stored.
    b : Theano shared variable, numpy array or callable
        Initializer for input gate bias vector.
    nonlinearity : callable or None
        The nonlinearity that is applied to the input gate activation. If None
        is provided, no nonlinearity will be applied.
    Examples
    --------
    For :class:`LSTMLayer` the bias of the forget gate is often initialized to
    a large positive value to encourage the layer initially remember the cell
    value, see e.g. [1]_ page 15.
    References
    ----------
    .. [1] Gers, Felix A., Jürgen Schmidhuber, and Fred Cummins. "Learning to
           forget: Continual prediction with LSTM." Neural computation 12.10
           (2000): 2451-2471.
    """
    def __init__(self, shape, W_in=Normal(std=.1), W_hid=Normal(.1), W_cell=False, b=Constant(0.), activation='sigmoid',
                 layer_name='Gate'):
        assert isinstance(shape, (list, tuple)), 'shape must be a list or tuple, got %s.' % type(shape)
        assert len(shape) == 2 or len(shape) == 4, 'shape must have 2 or 4 elements, got %d.' % len(shape)

        self.shape = shape
        hid_shape = tuple([shape[0], shape[0]] + list(shape[2:])) if len(shape) == 4 else (shape[1], shape[1])
        self.W_in = theano.shared(W_in(shape), layer_name+'/W_in')
        self.W_hid = theano.shared(W_hid(hid_shape), layer_name+'/W_hid')
        self.params = [self.W_in, self.W_hid]
        self.trainable = [self.W_in, self.W_hid]
        self.regularizable = [self.W_in, self.W_hid]

        if W_cell:
            self.W_cell = theano.shared(W_cell(hid_shape), layer_name+'/W_cell')
            self.params.append(self.W_cell)
            self.trainable.append(self.W_cell)
            self.regularizable.append(self.W_cell)
        self.b = theano.shared(b(hid_shape[0]), layer_name + '/bias')
        self.params.append(self.b)
        self.trainable.append(self.b)
        self.activation = utils.function[activation]


class LSTMCell(Layer):
    def __init__(self, input_shape, num_units, use_peephole=False, backward=False, learn_init=False, grad_step=-1,
                 grad_clip=0, activation='tanh', layer_name='LSTMCell', **kwargs):
        """

        :param input_shape:
        :param num_units:
        :param use_peephole:
        :param backward:
        :param learn_init:
        :param grad_step:
        :param grad_clip:
        :param activation:
        :param layer_name:
        """
        assert isinstance(input_shape, (list, tuple)), 'input_shape must be a list or tuple, got %s.' % input_shape
        assert len(input_shape) == 3, 'input_shape must contain exactly 3 elements, got %d.' % len(input_shape)

        super(LSTMCell, self).__init__(tuple(input_shape), layer_name)
        self.num_units = num_units
        self.use_peephole = use_peephole
        self.backward = backward
        self.learn_init = learn_init
        self.grad_step = grad_step
        self.grad_clip = grad_clip
        self.activation = utils.function[activation]
        self.kwargs = kwargs

        n_in = self.input_shape[-1]
        self.in_gate = Gate((n_in, num_units), W_cell=False, layer_name='in_gate')
        self.forget_gate = Gate((n_in, num_units), W_cell=False, layer_name='forget_gate')
        self.cell_gate = Gate((n_in, num_units), W_cell=False, layer_name='cell_gate')
        self.out_gate = Gate((n_in, num_units), W_cell=False, layer_name='out_gate')

        self.cell_init = theano.shared(np.zeros((1, num_units), 'float32'), 'cell_init')
        self.hid_init = theano.shared(np.zeros((1, num_units), 'float32'), 'hid_init')
        self.params += self.in_gate.trainable + self.forget_gate.trainable + \
                       self.cell_gate.trainable + self.out_gate.trainable + [self.cell_init, self.hid_init]
        self.trainable += self.in_gate.trainable + self.forget_gate.trainable + \
                          self.cell_gate.trainable + self.out_gate.trainable
        self.regularizable += self.in_gate.regularizable + self.forget_gate.regularizable + \
                              self.cell_gate.regularizable + self.out_gate.regularizable
        if self.learn_init:
            self.trainable += [self.cell_init, self.hid_init]
        self.descriptions = '%s LSTMCell: shape = (%d, %d)' % (self.layer_name, n_in, num_units)

    def get_output(self, input):
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape
        W_in_stacked = T.concatenate([self.in_gate.W_in, self.forget_gate.W_in, self.cell_gate.W_in, self.out_gate.W_in], 1)
        W_hid_stacked = T.concatenate([self.in_gate.W_hid, self.forget_gate.W_hid, self.cell_gate.W_hid, self.out_gate.W_hid], 1)
        b_stacked = T.concatenate([self.in_gate.b, self.forget_gate.b, self.cell_gate.b, self.out_gate.b], 0)

        def slice_w(x, n):
            s = x[:, n * self.num_units:(n + 1) * self.num_units]
            if self.num_units == 1:
                s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
            return s

        def step(input_n, cell_prev, hid_prev, *args):
            input_n = T.dot(input_n, W_in_stacked) + b_stacked
            gates = input_n + T.dot(hid_prev, W_hid_stacked)
            if self.grad_clip:
                gates = theano.gradient.grad_clip(gates, -self.grad_clip, self.grad_clip)

            in_gate = slice_w(gates, 0)
            forget_gate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            out_gate = slice_w(gates, 3)

            in_gate = self.in_gate.activation(in_gate, **self.kwargs)
            forget_gate = self.forget_gate.activation(forget_gate, **self.kwargs)
            cell_input = self.cell_gate.activation(cell_input, **self.kwargs)

            cell = forget_gate * cell_prev + in_gate * cell_input
            out_gate = self.out_gate.activation(out_gate, **self.kwargs)
            hid = out_gate * self.activation(cell)
            return cell, hid

        ones = T.ones((num_batch, 1), 'float32')
        non_seqs = [W_hid_stacked, W_in_stacked, b_stacked]
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)
        cell_out, hid_out = theano.scan(step, input, [cell_init, hid_init], go_backwards=self.backward,
                                        truncate_gradient=self.grad_step, non_sequences=non_seqs, strict=True)[0]
        hid_out = hid_out.dimshuffle(1, 0, 2)
        if self.backward:
            hid_out = hid_out[:, ::-1]
        return hid_out

    @property
    def output_shape(self):
        return self.input_shape[0], self.input_shape[1], self.num_units


class GRUCell(Layer):
    def __init__(self, input_shape, num_units, backwards=False, learn_init=False, grad_steps=-1, grad_clip=0, layer_name='GRUCell', **kwargs):
        assert isinstance(input_shape, (list, tuple)), 'input_shape must be a list or tuple, got %s.' % input_shape
        assert len(input_shape) == 3, 'input_shape must contain exactly 3 elements, got %d.' % len(input_shape)

        super(GRUCell, self).__init__(tuple(input_shape), layer_name)
        self.num_units = num_units
        self.backwards = backwards
        self.learn_init = learn_init
        self.grad_steps = grad_steps
        self.grad_clip = grad_clip
        self.kwargs = kwargs

        num_inputs = input_shape[-1]
        self.update_gate = Gate((num_inputs, num_units), layer_name='update_gate')
        self.reset_gate = Gate((num_inputs, num_units), layer_name='reset_gate')
        self.hidden_update = Gate((num_inputs, num_units), layer_name='hidden_update')
        self.params += self.update_gate.params + self.reset_gate.params + self.hidden_update.params
        self.trainable += self.update_gate.trainable + self.reset_gate.trainable + self.hidden_update.trainable
        self.regularizable += self.update_gate.regularizable + self.reset_gate.regularizable + self.hidden_update.regularizable

        self.hid_init = theano.shared(np.zeros((1, num_units), 'float32'), 'hid_init')
        self.params.append(self.hid_init)
        if learn_init:
            self.trainable.append(self.hid_init)
        self.descriptions = '%s GRUCell: shape = (%d, %d)' % (self.layer_name, num_inputs, num_units)

    def get_output(self, input):
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        W_in_stacked = T.concatenate([self.reset_gate.W_in, self.update_gate.W_in, self.hidden_update.W_in], 1)
        W_hid_stacked = T.concatenate([self.reset_gate.W_hid, self.update_gate.W_hid, self.hidden_update.W_hid], 1)
        b_stacked = T.concatenate([self.reset_gate.b, self.update_gate.b, self.hidden_update.b])

        def slice_w(x, n):
            s = x[:, n * self.num_units:(n + 1) * self.num_units]
            if self.num_units == 1:
                s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
            return s

        def step(input_n, hid_prev, *args):
            hid_input = T.dot(hid_prev, W_hid_stacked)

            if self.grad_clip:
                input_n = theano.gradient.grad_clip(input_n, -self.grad_clip, self.grad_clip)
                hid_input = theano.gradient.grad_clip(hid_input, -self.grad_clip, self.grad_clip)

            input_n = T.dot(input_n, W_in_stacked) + b_stacked

            reset_gate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            update_gate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            reset_gate = self.reset_gate.activation(reset_gate, **self.kwargs)
            update_gate = self.update_gate.activation(update_gate, **self.kwargs)

            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = hidden_update_in + reset_gate * hidden_update_hid
            if self.grad_clip:
                hidden_update = theano.gradient.grad_clip(hidden_update, -self.grad_clip, self.grad_clip)
            hidden_update = self.hidden_update.activation(hidden_update, **self.kwargs)
            return (1. - update_gate) * hid_prev + update_gate * hidden_update

        hid_init = T.dot(T.ones((num_batch, 1), 'float32'), self.hid_init)
        non_seqs = [W_hid_stacked, W_in_stacked, b_stacked]
        hid_out = theano.scan(step, input, [hid_init], non_seqs, go_backwards=self.backwards,
                              truncate_gradient=self.grad_steps, strict=True)[0]
        hid_out = hid_out.dimshuffle(1, 0, 2)
        if self.backwards:
            hid_out = hid_out[:, ::-1]
        return hid_out

    @property
    def output_shape(self):
        return self.input_shape[0], self.input_shape[1], self.num_units


class ConvLSTMCell(Layer):
    def __init__(self, input_shape, filter_shape, use_peephole=False, backward=False, learn_init=False, grad_step=-1,
                 grad_clip=0, activation='tanh', layer_name='ConvLSTMCell', **kwargs):
        assert isinstance(input_shape, (list, tuple)), 'input_shape must be a list or tuple, got %s.' % input_shape
        assert len(input_shape) == 5, 'input_shape must contain exactly 5 elements, got %d.' % len(input_shape)

        super(ConvLSTMCell, self).__init__(tuple(input_shape), layer_name)
        self.filter_shape = filter_shape
        self.use_peephole = use_peephole
        self.backward = backward
        self.learn_init = learn_init
        self.grad_step = grad_step
        self.grad_clip = grad_clip
        self.activation = utils.function[activation]
        self.kwargs = kwargs

        n_in = self.input_shape[-1]
        self.in_gate = Gate(filter_shape, W_cell=False, layer_name='in_gate')
        self.forget_gate = Gate(filter_shape, W_cell=False, layer_name='forget_gate')
        self.cell_gate = Gate(filter_shape, W_cell=False, activation='tanh', layer_name='cell_gate')
        self.out_gate = Gate(filter_shape, W_cell=False, layer_name='out_gate')

        self.cell_init = theano.shared(np.zeros(self.output_shape[1:], 'float32'), 'cell_init')
        self.hid_init = theano.shared(np.zeros(self.output_shape[1:], 'float32'), 'hid_init')
        self.params += self.in_gate.trainable + self.forget_gate.trainable + \
                       self.cell_gate.trainable + self.out_gate.trainable + [self.cell_init, self.hid_init]
        self.trainable += self.in_gate.trainable + self.forget_gate.trainable + \
                          self.cell_gate.trainable + self.out_gate.trainable
        self.regularizable += self.in_gate.regularizable + self.forget_gate.regularizable + \
                              self.cell_gate.regularizable + self.out_gate.regularizable
        if self.learn_init:
            self.trainable += [self.cell_init, self.hid_init]
        self.descriptions = '{} ConvLSTMCell: input shape = {} filter shape = {}'.format(self.layer_name, input_shape,
                                                                                     filter_shape)

    @property
    def output_shape(self):
        return tuple([self.input_shape[0], self.input_shape[1], self.filter_shape[0]] + list(self.input_shape[3:]))

    def get_output(self, input):
        input = input.dimshuffle(1, 0, 2, 3, 4)
        seq_len, num_batch, _, _, _ = input.shape
        conv = partial(T.nnet.conv2d, border_mode='half')

        def step(input_n, cell_prev, hid_prev, *args):
            It = self.in_gate.activation(conv(input_n, self.in_gate.W_in) + conv(hid_prev, self.in_gate.W_hid) + self.in_gate.b, **self.kwargs)
            Ft = self.forget_gate.activation(conv(input_n, self.forget_gate.W_in) + conv(hid_prev, self.forget_gate.W_hid) + self.forget_gate.b, **self.kwargs)
            Ot = self.out_gate.activation(conv(input_n, self.out_gate.W_in) + conv(hid_prev, self.out_gate.W_hid) + self.out_gate.b, **self.kwargs)
            Gt = self.cell_gate.activation(conv(input_n, self.cell_gate.W_in) + conv(hid_prev, self.cell_gate.W_hid) + self.cell_gate.b, **self.kwargs)
            Ct = Ft * cell_prev + It * Gt
            Ht = Ot * self.activation(Ct)
            return Ct, Ht

        non_seqs = [self.in_gate.W_in, self.in_gate.W_hid, self.in_gate.b, self.forget_gate.W_in, self.forget_gate.W_hid,
                    self.forget_gate.b, self.out_gate.W_in, self.out_gate.W_hid, self.out_gate.b, self.cell_gate.W_in,
                    self.cell_gate.W_hid, self.cell_gate.b]
        cell_out, hid_out = theano.scan(step, input, [self.cell_init, self.hid_init], go_backwards=self.backward,
                                        truncate_gradient=self.grad_step, non_sequences=non_seqs, strict=True)[0]
        hid_out = hid_out.dimshuffle(1, 0, 2, 3, 4)
        if self.backward:
            hid_out = hid_out[:, ::-1]
        return hid_out


class AttConvLSTMCell(Layer):
    def __init__(self, input_shape, num_filters, filter_size, steps, use_peephole=False, learn_init=False, grad_step=-1,
                 grad_clip=0, activation='tanh', layer_name='AttConvLSTMCell', **kwargs):
        assert isinstance(input_shape, (list, tuple)), 'input_shape must be a list or tuple, got %s.' % input_shape
        assert len(input_shape) == 4, 'input_shape must contain exactly 4 elements, got %d.' % len(input_shape)
        assert isinstance(filter_size, (list, tuple, int)), 'filter_size must be a list, tuple or int, got %s.' % type(filter_size)

        super(AttConvLSTMCell, self).__init__(tuple(input_shape), layer_name)
        filter_shape = (num_filters, input_shape[1], filter_size, filter_size) if isinstance(filter_size, int) \
            else tuple([num_filters, input_shape[1]] + list(filter_size))
        self.filter_shape = filter_shape
        self.steps = steps
        self.use_peephole = use_peephole
        self.learn_init = learn_init
        self.grad_step = grad_step
        self.grad_clip = grad_clip
        self.activation = utils.function[activation]
        self.kwargs = kwargs

        self.in_gate = Gate(filter_shape, W_cell=False, layer_name='in_gate')
        self.forget_gate = Gate(filter_shape, W_cell=False, layer_name='forget_gate')
        self.cell_gate = Gate(filter_shape, W_cell=False, activation='tanh', layer_name='cell_gate')
        self.out_gate = Gate(filter_shape, W_cell=False, layer_name='out_gate')
        self.att_gate = Gate(filter_shape, W_cell=False, activation='tanh', layer_name='att_gate')
        # self.Va = theano.shared(np.zeros((1, filter_shape[0], filter_shape[2], filter_shape[3]), 'float32'), 'att_kern')

        self.params += self.in_gate.params + self.forget_gate.params + self.cell_gate.params + \
                       self.out_gate.params + self.att_gate.params #+ [self.Va]
        self.trainable += self.in_gate.trainable + self.forget_gate.trainable + self.cell_gate.trainable + \
                          self.out_gate.trainable + self.att_gate.trainable #+ [self.Va]
        self.regularizable += self.in_gate.regularizable + self.forget_gate.regularizable + \
                              self.cell_gate.regularizable + self.out_gate.regularizable + self.att_gate.regularizable
        if self.learn_init:
            self.cell_init = theano.shared(
                np.zeros((1, filter_shape[0], input_shape[2], input_shape[3]), 'float32'), 'cell_init')
            self.hid_init = theano.shared(
                np.zeros((1, filter_shape[0], input_shape[2], input_shape[3]), 'float32'), 'hid_init')
            self.params += [self.cell_init, self.hid_init]
            self.trainable += [self.cell_init, self.hid_init]
        self.descriptions = '{} AttConvLSTMCell: input shape = {} filter shape = {}'.format(self.layer_name, input_shape,
                                                                                            filter_shape)

    @property
    def output_shape(self):
        return tuple([self.input_shape[0], self.filter_shape[0]] + list(self.input_shape[2:]))

    def get_output(self, input):
        conv = lambda x, y: T.nnet.conv2d(x, y, border_mode='half')

        num_batch, _, _, _ = input.shape
        if self.learn_init:
            cell_init = T.tile(self.cell_init, (num_batch, 1, 1, 1))
            hid_init = T.tile(self.hid_init, (num_batch, 1, 1, 1))
        else:
            cell_init = T.zeros((num_batch, self.filter_shape[0], self.input_shape[2], self.input_shape[3]), 'float32')
            hid_init = T.zeros((num_batch, self.filter_shape[0], self.input_shape[2], self.input_shape[3]), 'float32')

        def softmax(x):
            exp = T.exp(x)
            return exp / T.sum(exp, (2, 3), keepdims=True)

        def step(Xt, cell_prev, hid_prev, *args):
            It = self.in_gate.activation(
                conv(Xt, self.in_gate.W_in) + conv(hid_prev, self.in_gate.W_hid) + self.in_gate.b.dimshuffle('x', 0, 'x', 'x'), **self.kwargs)
            Ft = self.forget_gate.activation(
                conv(Xt, self.forget_gate.W_in) + conv(hid_prev, self.forget_gate.W_hid) + self.forget_gate.b.dimshuffle('x', 0, 'x', 'x'),
                **self.kwargs)
            Ot = self.out_gate.activation(
                conv(Xt, self.out_gate.W_in) + conv(hid_prev, self.out_gate.W_hid) + self.out_gate.b.dimshuffle('x', 0, 'x', 'x'),
                **self.kwargs)
            Gt = self.cell_gate.activation(
                conv(Xt, self.cell_gate.W_in) + conv(hid_prev, self.cell_gate.W_hid) + self.cell_gate.b.dimshuffle('x', 0, 'x', 'x'),
                **self.kwargs)
            Ct = Ft * cell_prev + It * Gt
            Ht = Ot * self.activation(Ct)

            # Zt = conv(self.att_gate.activation(conv(Xt, self.att_gate.W_in) + conv(hid_prev, self.att_gate.W_hid)
            #                                    + self.att_gate.b.dimshuffle('x', 0, 'x', 'x'), **self.kwargs), self.Va)
            #Xt = T.addbroadcast(softmax(Zt), 1) * Xt
            Zt = self.att_gate.activation(conv(Xt, self.att_gate.W_in) + conv(hid_prev, self.att_gate.W_hid)
                                          + self.att_gate.b.dimshuffle('x', 0, 'x', 'x'), **self.kwargs)
            Xt = softmax(Zt) * Xt
            return Xt, Ct, Ht

        non_seqs = [self.in_gate.W_in, self.in_gate.W_hid, self.in_gate.b, self.forget_gate.W_in, self.forget_gate.W_hid,
                    self.forget_gate.b, self.out_gate.W_in, self.out_gate.W_hid, self.out_gate.b, self.cell_gate.W_in,
                    self.cell_gate.W_hid, self.cell_gate.b, self.att_gate.W_in, self.att_gate.W_hid, self.att_gate.b]#, self.Va]
        X, cell_out, hid_out = theano.scan(step, outputs_info=[input, cell_init, hid_init], strict=True,
                                           truncate_gradient=self.grad_step, non_sequences=non_seqs, n_steps=self.steps)[0]
        return X[-1]


def set_training_status(training):
    Layer.set_training_status(training)


def GlobalAveragePoolingLayer(input_shape, layer_name='GlbAvgPooling'):
    return PoolingLayer(input_shape, input_shape[2:], True, (1, 1), (0, 0), 'average_exc_pad', layer_name)


def MaxPoolingLayer(input_shape, ws, ignore_border=True, stride=None, pad='valid', layer_name='MaxPooling'):
    return PoolingLayer(input_shape, ws, ignore_border, stride, pad, 'max', layer_name)


def AveragePoolingLayer(input_shape, ws, ignore_border=True, stride=None, pad='valid', layer_name='AvgPooling'):
    return PoolingLayer(input_shape, ws, ignore_border, stride, pad, 'average_exc_pad', layer_name)


def ConvNormAct(input_shape, num_filters, filter_size, init=HeNormal(gain=1.), no_bias=True, border_mode='half',
                stride=(1, 1), layer_name='ConvNormAct', activation='relu', dilation=(1, 1), epsilon=1e-4,
                running_average_factor=1e-1, axes='spatial', no_scale=False, normalization='bn', groups=32, **kwargs):
    assert normalization in ('bn', 'gn'), 'normalization can be either \'bn\' or \'gn\', got %s' % normalization

    block = Sequential(input_shape=input_shape, layer_name=layer_name)
    block.append(ConvolutionalLayer(block.output_shape, num_filters, filter_size, init, no_bias, border_mode, stride,
                                    dilation, layer_name+'_conv', 'linear', **kwargs))
    if normalization == 'bn':
        block.append(BatchNormLayer(block.output_shape, layer_name+'/bn', epsilon, running_average_factor, axes,
                                    activation, no_scale, **kwargs))
    else:
        block.append(GroupNormLayer(block.output_shape, layer_name+'/gn', groups, epsilon, activation, **kwargs))
    return block


def StackingConv(input_shape, num_layers, num_filters, filter_size=3, batch_norm=False, layer_name='StackingConv',
                 init=HeNormal(gain=1.), no_bias=True, border_mode='half', stride=1, dilation=(1, 1), activation='relu', **kwargs):
    assert num_layers > 1, 'num_layers must be greater than 1, got %d' % num_layers

    block = Sequential(input_shape=input_shape, layer_name=layer_name)
    conv_layer = ConvNormAct if batch_norm else ConvolutionalLayer
    for num in range(num_layers - 1):
        block.append(conv_layer(block.output_shape, num_filters, filter_size, init=init, no_bias=no_bias, border_mode=border_mode,
                                stride=(1, 1), dilation=dilation, layer_name=layer_name + '/conv_%d' % (num + 1),
                                activation=activation, **kwargs))
    block.append(conv_layer(block.output_shape, num_filters, filter_size, init=init, no_bias=no_bias, border_mode=border_mode,
                            dilation=dilation, stride=stride, activation=activation,
                            layer_name=layer_name + '/conv_%d' % num_layers, **kwargs))
    return block


def NetworkInNetworkBlock(input_shape, num_filters, filter_size, num_layers=2, num_nodes=(96, 96), activation='relu',
                          layer_name='NetworkInNetworkBlock', **kwargs):
        assert len(
            num_nodes) == num_layers, 'The number of element in num_nodes must be equal to num_layers, got %d and %d.' % (
            len(num_nodes), num_layers)

        block = Sequential(input_shape=input_shape, layer_name=layer_name)
        block.append(ConvolutionalLayer(input_shape, num_filters, filter_size, activation=activation,
                                        layer_name=layer_name + '/first_conv', **kwargs))
        for i in range(num_layers):
            block.append(ConvolutionalLayer(block.output_shape, num_nodes[i], 1, activation=activation,
                                            layer_name=layer_name + '/conv1_%d' % (i + 1), **kwargs))
        return block


def MeanPoolConvLayer(input_shape, num_filters, filter_size, activation='linear', ws=(2, 2), init=HeNormal(gain=1.),
                      no_bias=False, layer_name='Mean Pool Conv', **kwargs):
        assert input_shape[2] / 2 == input_shape[2] // 2 and input_shape[3] / 2 == input_shape[3] // 2, 'Input must have even shape.'

        block = Sequential(input_shape=input_shape, layer_name=layer_name)
        block.append(PoolingLayer(input_shape, ws, stride=ws, ignore_border=True, mode='average_exc_pad',
                                  layer_name=layer_name+'/meanpool'))
        block.append(ConvolutionalLayer(block.output_shape, num_filters, filter_size, init, no_bias,
                                        layer_name=layer_name+'/conv', activation=activation, **kwargs))
        return block


def ConvMeanPoolLayer(input_shape, num_filters, filter_size, activation='linear', ws=(2, 2), init=HeNormal(gain=1.),
                      no_bias=False, layer_name='Conv Mean Pool', **kwargs):
        assert input_shape[2] / 2 == input_shape[2] // 2 and input_shape[3] / 2 == input_shape[3] // 2, 'Input must have even shape.'

        block = Sequential(input_shape=input_shape, layer_name=layer_name)
        block.append(ConvolutionalLayer(block.output_shape, num_filters, filter_size, init, no_bias,
                                        layer_name=layer_name + '/conv', activation=activation, **kwargs))
        block.append(PoolingLayer(block.output_shape, ws, stride=ws, ignore_border=True, mode='average_exc_pad',
                                  layer_name=layer_name+'/meanpool'))
        return block


def SoftmaxLayer(input_shape, num_nodes, layer_name='Softmax'):
    return FullyConnectedLayer(input_shape, num_nodes, layer_name=layer_name, activation='softmax')


def SigmoidLayer(input_shape, layer_name='Sigmoid'):
    return FullyConnectedLayer(input_shape, 1, layer_name=layer_name, activation='sigmoid')


if __name__ == '__main__':
    from neuralnet import model_zoo
    X = T.tensor4('float32')
    net = model_zoo.resnet50((None, 3, 224, 224), 64, 10)
    Y_ = net(X)
    f = theano.function([X], Y_)
    x = np.random.rand(64, 3, 224, 224).astype('float32')
    y = f(x)
    print(y.shape)
