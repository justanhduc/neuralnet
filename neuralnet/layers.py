'''
Written and collected by Duc Nguyen
Last Modified by Duc Nguyen (theano version >= 0.8.1 required)
Updates on Sep 2017: added BatchNormDNNLayer from Lasagne


'''
__author__ = 'Duc Nguyen'

import math
import time
import numpy as np
import abc
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d as conv
from theano.tensor.signal.pool import pool_2d as pool

from neuralnet import utils


def validate(func):
    """make sure output shape is a list of ints"""
    def func_wrapper(self):
        out = [int(x) if x is not None else x for x in func(self)]
        return tuple(out)
    return func_wrapper


class Layer(metaclass=abc.ABCMeta):
    def __init__(self):
        self.rng = np.random.RandomState(int(time.time()))
        self.params = []
        self.trainable = []
        self.regularizable = []
        self.descriptions = ''

    def __str__(self):
        return self.descriptions

    def __call__(self, *args, **kwargs):
        return self.get_output(*args, **kwargs)

    @abc.abstractmethod
    def get_output(self, input):
        return

    @property
    @abc.abstractmethod
    def output_shape(self):
        return

    def init_he(self, shape, activation, sampling='uniform', lrelu_alpha=0.1):
        # He et al. 2015
        if activation in ['relu', 'elu']:  # relu or elu
            gain = np.sqrt(2)
        elif activation == 'lrelu':  # lrelu
            gain = np.sqrt(2 / (1 + lrelu_alpha ** 2))
        else:
            gain = 1.0

        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        if sampling == 'normal':
            std = gain * np.sqrt(1. / fan_in)
            return np.asarray(self.rng.normal(0., std, shape), dtype=theano.config.floatX)
        elif sampling == 'uniform':
            bound = gain * np.sqrt(3. / fan_in)
            return np.asarray(self.rng.uniform(-bound, bound, shape), dtype=theano.config.floatX)
        else:
            raise NotImplementedError

    def reset(self):
        pass


class Sequential(Layer):
    """
    Mimicking Pytorch Sequential class
    """
    def __init__(self, layer_list=[]):
        super(Sequential, self).__init__()
        assert isinstance(layer_list, (list, tuple, Sequential)), 'layer_list must be a list or tuple, got %s.' % type(layer_list)
        self.block = list(layer_list) if isinstance(layer_list, (list, tuple)) else list(layer_list.block)
        self.params = [p for layer in layer_list for p in layer.params] if isinstance(layer_list, (list, tuple)) else layer_list.params
        self.trainable = [p for layer in layer_list for p in layer.trainable] if isinstance(layer_list, (list, tuple)) else layer_list.trainable
        self.regularizable = [p for layer in layer_list for p in layer.regularizable] if isinstance(layer_list, (list, tuple)) else layer_list.regularizable
        self.__idx = 0
        self.__max = len(self.block) - 1

    def __iter__(self):
        self.__idx = 0
        self.__max = len(self.block) - 1
        return self

    def __next__(self):
        if self.__idx > self.__max:
            raise StopIteration
        self.__idx += 1
        return self.block[self.__idx - 1]

    def __len__(self):
        return len(self.block)

    def __getitem__(self, item):
        assert isinstance(item, int), 'index should be int, got %s.' % type(item)
        return self.block[item]

    def get_output(self, input):
        out = input
        for layer in self.block:
            out = layer(out)
        return out

    @property
    def output_shape(self):
        return self.block[-1].output_shape

    def append(self, layer):
        self.block.append(layer)
        self.params += layer.params
        self.trainable += layer.trainable
        self.regularizable += layer.regularizable

    def __add__(self, other):
        assert isinstance(other, Sequential), 'Cannot concatenate a Sequential object with a %s object.' % type(other)
        res = Sequential()
        res.block = self.block + other.block
        res.params = self.params + other.params
        res.trainable = self.trainable + other.trainable
        res.regularizable = self.regularizable + other.regularizable
        res.__idx = 0
        res.__max = len(res.block)
        return res

    def __str__(self):
        descriptions = ''
        for idx, layer in enumerate(self.block):
            descriptions += layer.descriptions if idx == len(self.block) - 1 else layer.descriptions + '\n'
        return descriptions

    def reset(self):
        for layer in self.block:
            layer.reset()


class ActivationLayer(Layer):
    def __init__(self, input_shape, activation='relu', layer_name='Activation', **kwargs):
        super(ActivationLayer, self).__init__()
        self.input_shape = tuple(input_shape)
        self.activation = utils.function[activation]
        self.layer_name = layer_name
        self.kwargs = kwargs
        self.descriptions = '{} Activation layer: {}'.format(self.layer_name, activation)

        if activation == 'prelu':
            self.alpha = theano.shared(np.float32(.1), 'alpha')
            self.params += [self.alpha]
            self.trainable += [self.alpha]
            self.kwargs['alpha'] = self.alpha

    def get_output(self, input):
        return self.activation(input, **self.kwargs)

    @property
    def output_shape(self):
        return tuple(self.input_shape)

    def reset(self):
        if self.alpha is utils.function['prelu']:
            self.alpha.set_value(np.float32(.1))


class PoolingLayer(Layer):
    def __init__(self, input_shape, ws=(2, 2), ignore_border=False, stride=(2, 2), pad='valid', mode='max', layer_name='Pooling'):
        """

        :param input_shape:
        :param ws:
        :param ignore_border:
        :param stride:
        :param pad:
        :param mode: {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        :param layer_name:
        """
        assert isinstance(input_shape, list) or isinstance(input_shape, tuple), \
            'input_shape must be list or tuple. Received %s' % type(input_shape)
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)
        assert mode in ('max', 'sum', 'average_inc_pad', 'average_exc_pad'), 'Invalid pooling mode. ' \
                                                                             'Mode should be \'max\', \'sum\', ' \
                                                                             '\'average_inc_pad\' or \'average_exc_pad\', ' \
                                                                             'got %s' % mode
        super(PoolingLayer, self).__init__()

        self.input_shape = tuple(input_shape)
        self.ws = ws
        self.ignore_border = ignore_border
        self.stride = stride
        self.mode = mode
        self.layer_name = layer_name
        if isinstance(pad, (list, tuple)):
            self.pad = tuple(pad)
        elif isinstance(pad, str):
            if pad == 'half':
                self.pad = (ws[0]//2, ws[1]//2)
            elif pad == 'valid':
                self.pad = (0, 0)
            elif pad == 'full':
                self.pad = (ws[0]-1, ws[1]-1)
            else:
                raise NotImplementedError
        else:
            raise TypeError
        self.descriptions = ''.join(('{} {} PoolingLayer: size: {}'.format(layer_name, mode, ws),
                                     ' stride: {}'.format(stride), ' {} -> {}'.format(input_shape, self.output_shape)))

    def get_output(self, input):
        return pool(input, self.ws, self.ignore_border, self.stride, self.pad, self.mode)

    @property
    @validate
    def output_shape(self):
        size = list(self.input_shape)
        if not self.ignore_border:
            size[2] = int(np.ceil((size[2] + 2 * self.pad[0] - self.ws[0]) / self.stride[0] + 1))
            size[3] = int(np.ceil((size[3] + 2 * self.pad[1] - self.ws[1]) / self.stride[1] + 1))
        else:
            size[2] = int(np.ceil((size[2] + 2 * self.pad[0] - self.ws[0]) / self.stride[0]))
            size[3] = int(np.ceil((size[3] + 2 * self.pad[1] - self.ws[1]) / self.stride[1]))
        return tuple(size)


class DropoutLayer(Layer):
    layers = []

    def __init__(self, input_shape, drop_prob=0.5, GaussianNoise=False, activation='relu', layer_name='Dropout', **kwargs):
        assert isinstance(input_shape, list) or isinstance(input_shape, tuple), \
            'input_shape must be list or tuple. Received %s' % type(input_shape)
        assert len(input_shape) == 2 or len(input_shape) == 4, \
            'input_shape must have 2 or 4 elements. Received %d' % len(input_shape)
        super(DropoutLayer, self).__init__()

        self.input_shape = tuple(input_shape)
        self.GaussianNoise = GaussianNoise
        self.activation = utils.function[activation]
        self.layer_name = layer_name
        self.srng = theano.sandbox.rng_mrg.MRG_RandomStreams(self.rng.randint(1, int(time.time())))
        self.keep_prob = T.as_tensor_variable(np.float32(1. - drop_prob))
        self.training_flag = False
        self.kwargs = kwargs
        self.descriptions = '{} Dropout Layer: p={:.2f} activation: {}'.format(layer_name, 1. - drop_prob, activation)
        DropoutLayer.layers.append(self)

    def get_output(self, input):
        mask = self.srng.normal(input.shape) + 1. if self.GaussianNoise else self.srng.binomial(n=1, p=self.keep_prob, size=input.shape)
        output_on = mask * input if self.GaussianNoise else input * T.cast(mask, theano.config.floatX)
        output_off = input if self.GaussianNoise else input * self.keep_prob
        return self.activation(output_on if self.training_flag else output_off, **self.kwargs)

    @property
    @validate
    def output_shape(self):
        return tuple(self.input_shape)

    @staticmethod
    def set_training(training):
        for layer in DropoutLayer.layers:
            layer.training_flag = training


class FullyConnectedLayer(Layer):
    def __init__(self, input_shape, num_nodes, He_init=None, He_init_gain=None, no_bias=False, layer_name='fc',
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
        assert isinstance(input_shape, list) or isinstance(input_shape, tuple), \
            'input_shape must be list or tuple. Received %s' % type(input_shape)
        assert len(input_shape) == 2 or len(input_shape) == 4, \
            'input_shape must have 2 or 4 elements. Received %d' % len(input_shape)
        super(FullyConnectedLayer, self).__init__()

        self.input_shape = tuple(input_shape) if len(input_shape) == 2 else (input_shape[0], np.prod(input_shape[1:]))
        self.num_nodes = num_nodes
        self.He_init = He_init
        self.He_init_gain = He_init_gain if He_init_gain is not None else activation
        self.activation = utils.function[activation]
        self.no_bias = no_bias
        self.layer_name = layer_name
        self.keep_dims = keep_dims
        self.kwargs = kwargs

        if self.He_init:
            self.W_values = self.init_he((self.input_shape[1], num_nodes), self.He_init_gain, self.He_init)
        else:
            W_bound = np.sqrt(6. / (self.input_shape[1] + num_nodes)) * 4 if self.activation is utils.function['sigmoid'] \
                else np.sqrt(6. / (self.input_shape[1] + num_nodes))
            self.W_values = np.asarray(self.rng.uniform(low=-W_bound, high=W_bound, size=(self.input_shape[1], num_nodes)),
                                       dtype=theano.config.floatX)
        self.W = theano.shared(value=np.copy(self.W_values), name=self.layer_name + '_W', borrow=True)#, target=self.target)
        self.trainable.append(self.W)
        self.params.append(self.W)
        self.regularizable.append(self.W)

        if not self.no_bias:
            self.b_values = np.zeros((num_nodes,), dtype=theano.config.floatX)
            self.b = theano.shared(value=np.copy(self.b_values), name=self.layer_name + '_b', borrow=True)
            self.trainable.append(self.b)
            self.params.append(self.b)

        if activation == 'prelu':
            self.alpha = theano.shared(np.float32(.1), layer_name + '_alpha')
            self.params += [self.alpha]
            self.trainable += [self.alpha]
            self.kwargs['alpha'] = self.alpha

        self.regularizable.append(self.W)
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
    def __init__(self, input_shape, num_filters, filter_size, He_init=None, He_init_gain=None, no_bias=True, border_mode='half',
                 stride=(1, 1), dilation=(1, 1), layer_name='conv', activation='relu', target='dev0', **kwargs):
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
        assert isinstance(input_shape, list) or isinstance(input_shape, tuple), \
            'input_shape must be list or tuple. Received %s' % type(input_shape)
        assert len(input_shape) == 2 or len(input_shape) == 4, \
            'input_shape must have 2 or 4 elements. Received %d' % len(input_shape)
        assert isinstance(num_filters, int) and isinstance(filter_size, (int, list, tuple))
        assert isinstance(border_mode, (int, list, tuple, str)), 'border_mode should be either \'int\', ' \
                                                                 '\'list\', \'tuple\' or \'str\', got {}'.format(type(border_mode))
        assert isinstance(stride, (int, list, tuple))
        super(ConvolutionalLayer, self).__init__()

        self.input_shape = tuple(input_shape)
        self.filter_shape = (num_filters, input_shape[1], filter_size[0], filter_size[1]) if isinstance(filter_size, (list, tuple)) \
            else (num_filters, input_shape[1], filter_size, filter_size)
        self.no_bias = no_bias
        self.activation = utils.function[activation]
        self.He_init = He_init
        self.He_init_gain = He_init_gain if He_init_gain is not None else activation
        self.layer_name = layer_name
        self.border_mode = border_mode
        self.subsample = tuple(stride) if isinstance(stride, (tuple, list)) else (stride, stride)
        self.dilation = dilation
        self.target = target
        self.kwargs = kwargs

        if He_init:
            self.W_values = self.init_he(self.filter_shape, self.He_init_gain, He_init)
        else:
            fan_in = np.prod(self.filter_shape[1:])
            fan_out = (self.filter_shape[0] * np.prod(self.filter_shape[2:]))
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W_values = np.asarray(self.rng.uniform(low=-W_bound, high=W_bound, size=self.filter_shape),
                                       dtype=theano.config.floatX)
        self.W = theano.shared(np.copy(self.W_values), name=self.layer_name + '_W', borrow=True)
        self.trainable.append(self.W)
        self.params.append(self.W)

        if not self.no_bias:
            self.b_values = np.zeros(self.filter_shape[0], dtype=theano.config.floatX)
            self.b = theano.shared(np.copy(self.b_values), self.layer_name + '_b', borrow=True)
            self.trainable.append(self.b)
            self.params.append(self.b)

        if activation == 'prelu':
            self.alpha = theano.shared(np.float32(.1), layer_name + '_alpha')
            self.params += [self.alpha]
            self.trainable += [self.alpha]
            self.kwargs['alpha'] = self.alpha

        self.regularizable += [self.W]
        self.descriptions = ''.join(('{} Conv Layer: '.format(self.layer_name), 'border mode: {} '.format(border_mode),
                                     'subsampling: {} dilation {} '.format(stride, dilation), 'input shape: {} x '.format(input_shape),
                                     'filter shape: {} '.format(self.filter_shape), '-> output shape {} '.format(self.output_shape),
                                     'activation: {} '.format(activation)))

    def get_output(self, input):
        output = conv(input=input, filters=self.W, border_mode=self.border_mode, subsample=self.subsample)
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


class StackingConv(Layer):
    def __init__(self, input_shape, num_layers, num_filters, filter_size=3, batch_norm=False, layer_name='StackingConv',
                 He_init=None, He_init_gain=None, no_bias=True, border_mode='half', stride=1, dilation=(1, 1),
                 activation='relu', **kwargs):
        assert num_layers > 1, 'num_layers must be greater than 1, got %d' % num_layers
        super(StackingConv, self).__init__()
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.layer_name = layer_name
        self.descriptions = '{} Stacking {} Convolutional Blocks {} filters size {} batchnorm {} ' \
                            'stride {} {} {}'.format(layer_name, num_layers, num_filters, filter_size, batch_norm,
                                                     stride,
                                                     activation,
                                                     ' '.join([' '.join((k, str(v))) for k, v in kwargs.items()]))
        self.block = []
        shape = tuple(self.input_shape)
        conv_layer = ConvNormAct if batch_norm else ConvolutionalLayer
        for num in range(num_layers - 1):
            self.block.append(conv_layer(shape, num_filters, filter_size, He_init=He_init, He_init_gain=He_init_gain,
                                         no_bias=no_bias, border_mode=border_mode, stride=(1, 1), dilation=dilation,
                                         layer_name=self.layer_name + '_conv_%d' % (num + 1), activation=activation,
                                         **kwargs))
            shape = self.block[-1].output_shape
            self.params += self.block[-1].params
            self.trainable += self.block[-1].trainable
            self.regularizable += self.block[-1].regularizable
        self.block.append(conv_layer(shape, num_filters, filter_size, He_init=He_init, He_init_gain=He_init_gain,
                                     no_bias=no_bias, border_mode=border_mode, dilation=dilation, stride=stride,
                                     activation=activation, layer_name=self.layer_name + '_conv_%d' % num_layers,
                                     **kwargs))
        self.params += self.block[-1].params
        self.trainable += self.block[-1].trainable
        self.regularizable += self.block[-1].regularizable

    def get_output(self, input):
        output = input
        for layer in self.block:
            output = layer(output)
        return output

    @property
    def output_shape(self):
        return self.block[-1].output_shape

    def reset(self):
        for layer in self.block:
            layer.reset()


class StackingDeconv(Layer):
    def __init__(self, input_shape, num_layers, num_filters, stride_first=True, output_shape=None, He_init=None,
                 layer_name='transconv', padding='half', stride=(2, 2), activation='relu'):
        super(StackingDeconv, self).__init__()
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.layer_name = layer_name
        self.stride_first = stride_first
        self.block = []

        if self.stride_first:
            o_shape = (input_shape[0], input_shape[1], stride[0] * input_shape[2], stride[1] * input_shape[3]) \
                if output_shape is not None else output_shape
            f_shape = (input_shape[1], num_filters, 3, 3)
            self.block.append(TransposedConvolutionalLayer(input_shape, f_shape, o_shape, He_init, stride=stride,
                                                           padding=padding, activation=activation,
                                                           layer_name=layer_name + '_deconv0'))
            shape = self.block[-1].output_shape
            self.params += self.block[-1].params
            self.trainable += self.block[-1].trainable
            self.regularizable += self.block[-1].regularizable
        else:
            shape = tuple(self.input_shape)
        for num in range(num_layers - 1):
            o_shape = (input_shape[0], num_filters, shape[2], shape[3])
            f_shape = (shape[1], num_filters, 3, 3)
            self.block.append(
                TransposedConvolutionalLayer(shape, f_shape, o_shape, He_init, stride=(1, 1), padding=padding,
                                             activation=activation, layer_name=layer_name + '_deconv%d' % (num + 1)))
            shape = self.block[-1].output_shape
            self.params += self.block[-1].params
            self.trainable += self.block[-1].trainable
            self.regularizable += self.block[-1].regularizable
        if not stride_first:
            o_shape = (input_shape[0], num_filters, shape[2] * stride[0], shape[3] * stride[1])
            f_shape = (shape[1], num_filters, 3, 3)
            self.block.append(
                TransposedConvolutionalLayer(shape, f_shape, o_shape, He_init, layer_name + '_deconv_last',
                                             stride=stride, activation=activation))
            self.params += self.block[-1].params
            self.trainable += self.block[-1].trainable
            self.regularizable += self.block[-1].regularizable

    def get_output(self, input):
        output = input
        for layer in self.block:
            output = layer(output)
        return output

    @property
    def output_shape(self):
        return self.block[-1].output_shape

    def reset(self):
        for layer in self.layers:
            layer.reset()


class DilatedConvModule(Layer):
    def __init__(self, input_shape, num_filters, filter_size, dilation_scale=3, He_init=None, He_init_gain=None, W=None,
                 b=None,
                 no_bias=True, border_mode='half', stride=(1, 1), layer_name='conv', activation='relu', target='dev0'):
        super(DilatedConvModule, self).__init__()
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_shape = filter_size
        self.dilation_scale = dilation_scale

        self.module = [[] for i in range(dilation_scale)]
        for i in range(dilation_scale):
            self.module[i].append(ConvNormAct(input_shape, num_filters, filter_size, 'normal', border_mode=border_mode,
                                              stride=(1, 1), dilation=(i + 1, i + 1), activation=activation,
                                              layer_name=layer_name + '_branch1'))
            self.module[i].append(ConvNormAct(self.module[i][-1].output_shape, num_filters, filter_size, 'normal',
                                              border_mode=border_mode, stride=(1, 1), dilation=(i + 1, i + 1),
                                              activation=activation,
                                              layer_name=layer_name + '_branch1_conv2'))
            self.module[i].append(ConvNormAct(self.module[i][-1].output_shape, num_filters, filter_size, 'normal',
                                              border_mode=border_mode, stride=stride, dilation=(i + 1, i + 1),
                                              activation=activation,
                                              layer_name=layer_name + '_branch1_conv3'))

        self.params += [p for block in self.module for layer in block for p in layer.params]
        self.trainable += [p for block in self.module for layer in block for p in layer.trainable]
        self.regularizable += [p for block in self.module for layer in block for p in layer.regularizable]

    def get_output(self, input):
        output = [utils.inference(input, block) for block in self.module]
        return T.concatenate(output, 1)

    @property
    def output_shape(self):
        shape = self.module[0][-1].output_shape
        return (self.input_shape[0], self.num_filters * self.dilation_scale, shape[2], shape[3])

    def reset(self):
        for block in self.module:
            for layer in block:
                layer.reset()


class InceptionModule1(Layer):
    def __init__(self, input_shape, num_filters=48, border_mode='half', stride=(1, 1), activation='relu',
                 layer_name='inception_mixed1'):
        super(InceptionModule1, self).__init__()
        self.input_shape = input_shape
        self.border_mode = border_mode
        self.stride = stride
        self.activation = activation
        self.layer_name = layer_name

        self.module = [[], [], [], []]
        self.module[0].append(ConvNormAct(input_shape, num_filters, (1, 1), 'normal', border_mode=border_mode,
                                          stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '_branch1_conv1x1'))
        self.module[0].append(ConvNormAct(self.module[0][-1].output_shape, num_filters * 4 // 3, 3,
                                        'normal', border_mode=border_mode, stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '_branch1_conv3x3'))
        self.module[0].append(ConvNormAct(self.module[0][-1].output_shape, num_filters * 4 // 3, 3,
                                        'normal', border_mode=border_mode, stride=stride, activation=activation,
                                          layer_name=layer_name + '_branch1_conv3x3'))

        self.module[1].append(ConvNormAct(input_shape, num_filters * 4 // 3, 1, 'normal', border_mode=border_mode,
                                          stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '_branch2_conv1x1'))
        self.module[1].append(ConvNormAct(self.module[1][-1].output_shape, num_filters * 2, 3,
                                        'normal', border_mode=border_mode, stride=stride, activation=activation,
                                          layer_name=layer_name + '_branch2_conv3x3'))

        self.module[2].append(PoolingLayer(input_shape, (3, 3), stride=stride, mode='average_exc_pad', pad='half',
                                           ignore_border=True, layer_name=layer_name + '_branch3_pooling'))
        self.module[2].append(ConvNormAct(self.module[2][-1].output_shape, num_filters * 2 // 3, 1,
                                        'normal', border_mode=border_mode, stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '_branch3_conv1x1'))

        self.module[3].append(ConvNormAct(input_shape, num_filters * 4 // 3, 1, 'normal', border_mode=border_mode,
                                          stride=stride, activation=activation,
                                          layer_name=layer_name + '_branch4_conv1x1'))

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
    def __init__(self, input_shape, num_filters=128, filter_size=7, border_mode='half', stride=(1, 1),
                 activation='relu',
                 layer_name='inception_mixed1'):
        super(InceptionModule2, self).__init__()
        self.input_shape = input_shape
        self.filter_size = filter_size
        self.border_mode = border_mode
        self.stride = stride
        self.activation = activation
        self.layer_name = layer_name

        self.module = [[], [], [], []]
        self.module[0].append(ConvNormAct(input_shape, num_filters, (1, 1), 'normal', border_mode=border_mode,
                                          stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '_branch1_conv1x1'))
        self.module[0].append(ConvNormAct(self.module[0][-1].output_shape, num_filters, (filter_size, 1),
                                        'normal', border_mode=border_mode, stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '_branch1_conv7x1_1'))
        self.module[0].append(ConvNormAct(self.module[0][-1].output_shape, num_filters, (1, filter_size),
                                        'normal', border_mode=border_mode, stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '_branch1_conv1x7_1'))
        self.module[0].append(ConvNormAct(self.module[0][-1].output_shape, num_filters, (filter_size, 1),
                                        'normal', border_mode=border_mode, stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '_branch1_conv7x1_2'))
        self.module[0].append(ConvNormAct(self.module[0][-1].output_shape, num_filters * 3 // 2, (1, filter_size),
                                        'normal', border_mode=border_mode, stride=stride, activation=activation,
                                          layer_name=layer_name + '_branch1_conv1x7_2'))

        self.module[1].append(ConvNormAct(input_shape, 64, (1, 1), 'normal', border_mode=border_mode,
                                          stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '_branch2_conv1x1'))
        self.module[1].append(ConvNormAct(self.module[1][-1].output_shape, num_filters, (filter_size, 1),
                                        'normal', border_mode=border_mode, stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '_branch2_conv7x1'))
        self.module[1].append(ConvNormAct(self.module[1][-1].output_shape, num_filters * 3 // 2, (1, filter_size),
                                        'normal', border_mode=border_mode, stride=stride, activation=activation,
                                          layer_name=layer_name + '_branch2_conv1x7'))

        self.module[2].append(PoolingLayer(input_shape, (3, 3), stride=stride, mode='average_exc_pad', pad='half',
                                           ignore_border=True, layer_name=layer_name + '_branch3_pooling'))
        self.module[2].append(ConvNormAct(self.module[2][-1].output_shape, num_filters * 3 // 2, (1, 1),
                                        'normal', border_mode=border_mode, stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '_branch3_conv1x1'))

        self.module[3].append(ConvNormAct(input_shape, num_filters * 3 // 2, (1, 1), 'normal', border_mode=border_mode,
                                          stride=stride, activation=activation,
                                          layer_name=layer_name + '_branch4_conv1x1'))

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
                 layer_name='inception_mixed1'):
        super(InceptionModule3, self).__init__()
        self.input_shape = input_shape
        self.border_mode = border_mode
        self.stride = stride
        self.activation = activation
        self.layer_name = layer_name

        self.module = [[], [], [], []]
        self.module[0].append(ConvNormAct(input_shape, num_filters * 7 // 5, (1, 1), 'normal', border_mode=border_mode,
                                          stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '_branch1_conv1x1'))
        self.module[0].append(ConvNormAct(self.module[0][-1].output_shape, num_filters * 6 // 5, (3, 3),
                                        'normal', border_mode=border_mode, stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '_branch1_conv3x3'))
        self.module[0].append([[], []])
        self.module[0][-1][0].append(ConvNormAct(self.module[0][1].output_shape, num_filters * 6 // 5, (3, 1),
                                               'normal', border_mode=border_mode, stride=stride, activation=activation,
                                                 layer_name=layer_name + '_branch1_conv3x1'))
        self.module[0][-1][1].append(ConvNormAct(self.module[0][1].output_shape, num_filters * 6 // 5, (3, 1),
                                               'normal', border_mode=border_mode, stride=stride, activation=activation,
                                                 layer_name=layer_name + '_branch1_conv1x3'))

        self.module[1].append(ConvNormAct(input_shape, num_filters * 7 // 5, (1, 1), 'normal', border_mode=border_mode,
                                          stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '_branch2_conv1x1'))
        self.module[1].append([[], []])
        self.module[1][-1][0].append(ConvNormAct(self.module[1][0].output_shape, num_filters * 6 // 5, (3, 1),
                                               'normal', border_mode=border_mode, stride=stride, activation=activation,
                                                 layer_name=layer_name + '_branch2_conv3x1'))
        self.module[1][-1][1].append(ConvNormAct(self.module[1][0].output_shape, num_filters * 6 // 5, (3, 1),
                                               'normal', border_mode=border_mode, stride=stride, activation=activation,
                                                 layer_name=layer_name + '_branch2_conv1x3'))

        self.module[2].append(PoolingLayer(input_shape, (3, 3), stride=stride, mode='average_exc_pad', pad='half',
                                           ignore_border=True, layer_name=layer_name + '_branch3_pooling'))
        self.module[2].append(ConvNormAct(self.module[2][-1].output_shape, num_filters * 2 // 3, (1, 1),
                                        'normal', border_mode=border_mode, stride=(1, 1), activation=activation,
                                          layer_name=layer_name + '_branch3_conv1x1'))

        self.module[3].append(ConvNormAct(input_shape, num_filters * 4 // 3, (1, 1), 'normal', border_mode=border_mode,
                                          stride=stride, activation=activation,
                                          layer_name=layer_name + '_branch4_conv1x1'))

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


class ConvNormAct(Layer):
    def __init__(self, input_shape, num_filters, filter_size, He_init=None, He_init_gain=None, no_bias=True, border_mode='half',
                 stride=(1, 1), layer_name='convbnact', activation='relu', dilation=(1, 1), epsilon=1e-4, running_average_factor=1e-1,
                 axes='spatial', no_scale=False, normalization='bn', groups=32, **kwargs):
        """

        :param input_shape:
        :param num_filters:
        :param filter_size:
        :param He_init:
        :param He_init_gain:
        :param no_bias:
        :param border_mode:
        :param stride:
        :param layer_name:
        :param activation:
        :param dilation:
        :param epsilon:
        :param running_average_factor:
        :param axes:
        :param no_scale:
        :param normalization:
        :param kwargs:
        """
        super(ConvNormAct, self).__init__()
        assert normalization in ('bn', 'gn'), 'normalization can be either \'bn\' or \'gn\', got %s' % normalization
        self.layer_type = 'Conv BN Act Block {} filters size {} padding {} stride {} {} {}'. \
            format(num_filters, filter_size, border_mode, stride, activation,
                   ' '.join([' '.join((k, str(v))) for k, v in kwargs.items()]))
        self.Conv = ConvolutionalLayer(input_shape, num_filters, filter_size, He_init=He_init,
                                       He_init_gain=He_init_gain,
                                       no_bias=no_bias, border_mode=border_mode, stride=stride, dilation=dilation,
                                       layer_name=layer_name + '_conv', activation='linear')
        self.Norm = BatchNormLayer(self.Conv.output_shape, layer_name + '_bn', epsilon, running_average_factor, axes,
                                   activation, no_scale, **kwargs) if normalization == 'bn' \
            else GroupNormLayer(self.Conv.output_shape, layer_name+'_gn', groups, epsilon, activation, **kwargs)
        self.block = [self.Conv, self.Norm]
        self.trainable += self.Conv.trainable + self.Norm.trainable
        self.regularizable += self.Conv.regularizable + self.Norm.regularizable
        self.params += self.Conv.params + self.Norm.params
        self.descriptions = '{} Conv BN Act: {} -> {}'.format(layer_name, input_shape, self.output_shape)

    def get_output(self, input):
        output = utils.inference(input, self.block)
        return output

    @property
    @validate
    def output_shape(self):
        return self.block[-1].output_shape

    def reset(self):
        for layer in self.block:
            layer.reset()


class TransposedConvolutionalLayer(Layer):
    def __init__(self, input_shape, num_filters, filter_size, output_shape=None, He_init=None, layer_name='transconv',
                 padding='half', stride=(2, 2), activation='relu', target='dev0', **kwargs):
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
        super(TransposedConvolutionalLayer, self).__init__()
        self.filter_shape = (input_shape[1], num_filters, filter_size[0], filter_size[1]) if isinstance(filter_size, (list, tuple)) \
            else (input_shape[1], num_filters, filter_size, filter_size)
        self.input_shape = tuple(input_shape)
        self.output_shape_tmp = (input_shape[0], num_filters, output_shape[0], output_shape[1]) \
            if output_shape is not None else output_shape
        self.padding = padding
        self.stride = stride
        self.activation = utils.function[activation]
        self.layer_name = layer_name
        self.target = target
        self.kwargs = kwargs

        self.b_values = np.zeros((self.filter_shape[1],), dtype=theano.config.floatX)
        if He_init:
            self.W_values = self.init_he(self.filter_shape, 'relu', He_init)
        else:
            fan_in = np.prod(self.filter_shape[1:])
            fan_out = (self.filter_shape[0] * np.prod(self.filter_shape[2:]))
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W_values = np.asarray(self.rng.uniform(low=-W_bound, high=W_bound, size=self.filter_shape),
                                       dtype=theano.config.floatX)
        self.W = theano.shared(np.copy(self.W_values), self.layer_name + '_W', borrow=True)
        self.b = theano.shared(np.copy(self.b_values), self.layer_name + '_b', borrow=True)
        self.params += [self.W, self.b]
        self.trainable += [self.W, self.b]
        self.regularizable.append(self.W)

        if activation == 'prelu':
            self.alpha = theano.shared(np.float32(.1), layer_name + '_alpha')
            self.params += [self.alpha]
            self.trainable += [self.alpha]
            self.kwargs['alpha'] = self.alpha

        self.descriptions = '{} Transposed Conv Layer: '.format(layer_name), \
                            'shape: {} '.format(input_shape), 'filter shape: {} '.format(self.filter_shape), \
                            '-> {} '.format(self.output_shape), 'padding: {}'.format(self.padding), \
                            'stride: {}'.format(self.stride), 'activation: {}'.format(activation)

    def _get_deconv_filter(self):
        """
        This function is collected
        :param f_shape: self.filter_shape
        :return: an initializer for get_variable
        """
        width = self.filter_shape[2]
        height = self.filter_shape[3]
        f = math.ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
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
        if self.padding == 'half':
            p = (self.filter_shape[2] // 2, self.filter_shape[3] // 2)
        elif self.padding == 'valid':
            p = (0, 0)
        elif self.padding == 'full':
            p = (self.filter_shape[2] - 1, self.filter_shape[3] - 1)
        else:
            raise NotImplementedError
        if self.output_shape_tmp is None:
            in_shape = output.shape
            h = ((in_shape[2] - 1) * self.stride[0]) + self.filter_shape[2] + \
                T.mod(in_shape[2]+2*p[0]-self.filter_shape[2], self.stride[0]) - 2*p[0]
            w = ((in_shape[3] - 1) * self.stride[1]) + self.filter_shape[3] + \
                T.mod(in_shape[3]+2*p[1]-self.filter_shape[3], self.stride[1]) - 2*p[1]
            self.output_shape_tmp = [self.output_shape_tmp[0], self.filter_shape[1], h, w]

        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(imshp=self.output_shape_tmp, kshp=self.filter_shape,
                                                            subsample=self.stride, border_mode=self.padding)
        input = op(self.W, output, self.output_shape_tmp[-2:])
        input = input + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.activation(input, **self.kwargs)

    @property
    @validate
    def output_shape(self):
        return tuple(self.output_shape_tmp)

    def reset(self):
        self.W.set_value(np.copy(self.W_values))
        self.b.set_value(np.copy(self.b_values))
        if self.activation is utils.function['prelu']:
            self.alpha.set_value(np.float32(.1))


class PixelShuffleLayer(Layer):
    def __init__(self, input_shape, num_filters, filter_size, rate=(2, 2), activation='relu', he_init=True,
                 biases=True, layer_name='Upsample Conv', **kwargs):
        super(PixelShuffleLayer, self).__init__()
        self.input_shape = tuple(input_shape)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.rate = rate
        self.activation = activation
        self.he_init = he_init
        self.biases = biases
        self.layer_name = layer_name

        self.shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]*rate[0], self.input_shape[3]*rate[1])
        self.conv = ConvolutionalLayer(self.shape, num_filters, filter_size, 'uniform' if self.he_init else None, activation=self.activation,
                                       layer_name=self.layer_name, no_bias=not self.biases, **kwargs)
        self.params += self.conv.params
        self.trainable += self.conv.trainable
        self.regularizable += self.conv.regularizable
        self.descriptions = '{} Upsample Conv: {} -> {}'.format(layer_name, self.input_shape, self.output_shape)

    def get_output(self, input):
        output = input
        output = T.concatenate([output for i in range(np.sum(self.rate))], 1)
        output = T.transpose(output, (0, 2, 3, 1))
        output = T.reshape(output, (-1, self.shape[2], self.shape[3], self.shape[1]))
        output = T.transpose(output, (0, 3, 1, 2))
        return self.conv(output)

    @property
    def output_shape(self):
        return (self.shape[0], self.num_filters, self.shape[2], self.shape[3])

    def reset(self):
        self.conv.reset()


class MeanPoolConvLayer(Layer):
    def __init__(self, input_shape, num_filters, filter_size, activation='relu', ws=(2, 2), he_init=True, biases=True, layer_name='Mean Pool Conv', **kwargs):
        assert input_shape[2] / 2 == input_shape[2] // 2 and input_shape[3] / 2 == input_shape[3] // 2, 'Input must have even shape.'
        super(MeanPoolConvLayer, self).__init__()
        self.input_shape = tuple(input_shape)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.activation = activation
        self.ws = ws
        self.he_init = he_init
        self.biases = biases
        self.layer_name = layer_name

        self.shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]//2, self.input_shape[3]//2)
        self.conv = ConvolutionalLayer(self.shape, num_filters, filter_size, 'uniform' if self.he_init else None,
                                       activation=self.activation, layer_name=self.layer_name, no_bias=not self.biases, **kwargs)
        self.params += self.conv.params
        self.trainable += self.conv.trainable
        self.regularizable += self.conv.regularizable
        self.descriptions = '{} Mean Pool Conv layer: {} -> {}'.format(layer_name, self.input_shape, self.output_shape)

    def get_output(self, input):
        output = input
        # output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output[:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4.
        output = T.signal.pool.pool_2d(output, self.ws, ignore_border=True, mode='average_exc_pad')
        return self.conv(output)

    @property
    def output_shape(self):
        return (self.shape[0], self.num_filters, self.shape[2], self.shape[3])

    def reset(self):
        self.conv.reset()


class ConvMeanPoolLayer(Layer):
    def __init__(self, input_shape, num_filters, filter_size, activation='relu', he_init=True, biases=True, layer_name='Mean Pool Conv', **kwargs):
        assert input_shape[2] / 2 == input_shape[2] // 2 and input_shape[3] / 2 == input_shape[3] // 2, 'Input must have even shape.'
        super(ConvMeanPoolLayer, self).__init__()
        self.input_shape = tuple(input_shape)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.activation = activation
        self.he_init = he_init
        self.biases = biases
        self.layer_name = layer_name

        self.conv = ConvolutionalLayer(self.input_shape, num_filters, filter_size, 'normal' if self.he_init else None,
                                       activation=self.activation, layer_name=self.layer_name, no_bias=not self.biases, **kwargs)
        self.params += self.conv.params
        self.trainable += self.conv.trainable
        self.regularizable += self.conv.regularizable
        self. descriptions = '{} Conv Mean Pool: {} -> {}'.format(layer_name, self.input_shape, self.output_shape)

    def get_output(self, input):
        output = input
        output = self.conv(output)
        return (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output[:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4.

    @property
    def output_shape(self):
        return (self.input_shape[0], self.num_filters, self.input_shape[2]//2, self.input_shape[3]//2)

    def reset(self):
        self.conv.reset()


class ResNetBlockWGAN(Layer):
    def __init__(self, input_shape, num_filters, filter_size, activation='relu', resample=None, he_init=True, layer_name='ResNet Block WGAN'):
        super(ResNetBlockWGAN, self).__init__()
        self.input_shape = tuple(input_shape)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.activation = activation
        self.resample = resample
        self.he_init = he_init
        self.layer_name = layer_name

        self.block = []
        self.proj = []
        if resample == 'down':
            self.proj.append(MeanPoolConvLayer(input_shape, num_filters, 1, 'linear', False, True, layer_name + '_shortcut'))

            self.block.append(GroupNormLayer(self.input_shape, layer_name + '_ln1', activation=activation))
            self.block.append(ConvolutionalLayer(self.block[-1].output_shape, input_shape[1], filter_size, 'uniform', no_bias=True,
                                                 layer_name=layer_name+'_conv1', activation='linear'))
            self.block.append(GroupNormLayer(self.block[-1].output_shape, layer_name + '_ln2', activation=activation))
            self.block.append(ConvMeanPoolLayer(self.block[-1].output_shape, num_filters, filter_size, 'linear',
                                                he_init, True, layer_name + '_ConvMeanPool'))
        elif resample == 'up':
            self.proj.append(PixelShuffleLayer(input_shape, num_filters, 1, 'linear', False, True, layer_name + '_shortcut'))

            self.block.append(BatchNormLayer(self.input_shape, layer_name + '_bn1', activation=activation))
            self.block.append(PixelShuffleLayer(self.block[-1].output_shape, num_filters, filter_size, 'linear',
                                                he_init, False, layer_name + '_UpConv'))
            self.block.append(BatchNormLayer(self.block[-1].output_shape, layer_name+'_bn2', activation=activation))
            self.block.append(ConvolutionalLayer(self.block[-1].output_shape, num_filters, filter_size, 'uniform', no_bias=False,
                                                 activation='linear', layer_name=layer_name+'_conv2'))
        elif resample is None:
            if input_shape[1] == num_filters:
                self.proj.append(IdentityLayer(input_shape, layer_name+'_shortcut'))
            else:
                f_shape = (num_filters, input_shape[1], 1, 1)
                self.proj.append(ConvolutionalLayer(input_shape, f_shape, None, activation='linear', no_bias=False,
                                                    layer_name=layer_name+'_shortcut'))

            f_shape = (input_shape[1], self.block[-1].output_shape[1], filter_size, filter_size)
            self.block.append(ConvolutionalLayer(self.block[-1].output_shape, f_shape, 'uniform', activation='linear',
                                                 layer_name=layer_name+'_conv1'))
            self.block.append(BatchNormLayer(self.block[-1].output_shape, layer_name+'_bn2', activation=activation))
            f_shape = (num_filters, self.block[-1].output_shape[1], filter_size, filter_size)
            self.block.append(ConvolutionalLayer(self.block[-1].output_shape, f_shape, 'uniform', activation='linear',
                                                 layer_name=layer_name+'_conv2'))
        else:
            raise NotImplementedError
        self.params += [p for layer in self.block for p in layer.params]
        self.params += [p for layer in self.proj for p in layer.params]
        self.trainable += [p for layer in self.block for p in layer.trainable]
        self.trainable += [p for layer in self.proj for p in layer.trainable]
        self.regularizable += [p for layer in self.block for p in layer.regularizable]
        self.regularizable += [p for layer in self.proj for p in layer.regularizable]

    @property
    def output_shape(self):
        return self.block[-1].output_shape

    def get_output(self, input):
        output = input
        for layer in self.block:
            output = layer(output)

        shortcut = input
        for layer in self.proj:
            shortcut = layer(shortcut)
        return shortcut + output


class ResNetBlock(Layer):
    def __init__(self, input_shape, num_filters, stride=(1, 1), dilation=(1, 1), activation='relu', left_branch=False,
                 layer_name='ResBlock', normalization='bn', groups=32, **kwargs):
        '''

        :param input_shape: (
        :param down_dim:
        :param up_dim:
        :param stride:
        :param activation:
        :param layer_name:
        :param branch1_conv:
        :param target:
        '''
        assert left_branch or (input_shape[1] == num_filters), 'Cannot have identity branch when input dim changes.'
        super(ResNetBlock, self).__init__()

        self.input_shape = tuple(input_shape)
        self.num_filters = num_filters
        self.stride = stride
        self.dilation = dilation
        self.layer_name = layer_name
        self.activation = activation
        self.left_branch = left_branch
        self.normalization = normalization
        self.groups = groups
        self.kwargs = kwargs
        self.descriptions = '{} ResNet Block 1 {} filters stride {} dilation {} left branch {} {} {}'.\
            format(layer_name, num_filters, stride, dilation, left_branch, activation, ' '.join([' '.join((k, str(v))) for k, v in kwargs.items()]))

        self.block = list(self._build_simple_block(block_name=layer_name + '_1', no_bias=True))
        self.params += [p for layer in self.block for p in layer.params]
        self.trainable += [p for layer in self.block for p in layer.trainable]
        self.regularizable += [p for layer in self.block for p in layer.regularizable]

        if activation == 'prelu':
            self.alpha = theano.shared(np.float32(.1), layer_name + '_alpha')
            self.params += [self.alpha]
            self.trainable += [self.alpha]
            self.kwargs['alpha'] = self.alpha

        if self.left_branch:
            self.shortcut = []
            self.shortcut.append(ConvolutionalLayer(self.input_shape, num_filters, 3, 'normal',
                                                    stride=stride, layer_name=layer_name+'_2', activation='linear'))
            self.shortcut.append(BatchNormLayer(self.shortcut[-1].output_shape, layer_name=layer_name + '_2_bn',
                                                activation='linear') if normalization == 'bn'
                                 else GroupNormLayer(self.shortcut[-1].output_shape, layer_name=layer_name + '_2_gn',
                                                     groups=groups, activation='linear'))
            self.params += [p for layer in self.shortcut for p in layer.params]
            self.trainable += [p for layer in self.shortcut for p in layer.trainable]
            self.regularizable += [p for layer in self.shortcut for p in layer.regularizable]

    def _build_simple_block(self, block_name, no_bias):
        block = []
        block.append(ConvolutionalLayer(self.input_shape, self.num_filters, 3, border_mode='half', stride=self.stride,
                                     dilation=self.dilation, layer_name=block_name + '_conv1', no_bias=no_bias, activation='linear'))
        block.append(BatchNormLayer(block[-1].output_shape, activation=self.activation,
                                     layer_name=block_name + '_conv1_bn', **self.kwargs) if self.normalization == 'bn'
                     else GroupNormLayer(block[-1].output_shape, activation=self.activation,
                                         layer_name=block_name+'_conv1_gn', groups=self.groups, **self.kwargs))

        block.append(ConvolutionalLayer(block[-1].output_shape, self.num_filters, 3, border_mode='half', dilation=self.dilation,
                                         layer_name=block_name + '_conv2', no_bias=no_bias, activation='linear'))
        block.append(BatchNormLayer(block[-1].output_shape, layer_name=block_name + '_conv2_bn', activation='linear')
                     if self.normalization == 'bn' else GroupNormLayer(block[-1].output_shape, activation='linear',
                                                                       layer_name=block_name + '_conv2_gn', groups=self.groups))
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


class ResNetBlock2(Layer):
    layers = []

    def __init__(self, input_shape, ratio_n_filter=1.0, stride=1, upscale_factor=4, dilation=(1, 1),
                 activation='relu', left_branch=False, layer_name='ResBlock', **kwargs):
        '''

        :param input_shape: (
        :param down_dim:
        :param up_dim:
        :param stride:
        :param activation:
        :param layer_name:
        :param branch1_conv:
        :param target:
        '''
        super(ResNetBlock2, self).__init__()

        self.input_shape = tuple(input_shape)
        self.ratio_n_filter = ratio_n_filter
        self.stride = stride
        self.upscale_factor = upscale_factor
        self.dilation = dilation
        self.layer_name = layer_name
        self.activation = activation
        self.left_branch = left_branch
        self.kwargs = kwargs
        self.descriptions = 'ResNet Block 2 ratio {} stride {} upscale {} dilation {} left branch {} {}'.\
            format(ratio_n_filter, stride, upscale_factor, dilation, left_branch, activation)

        self.block = list(self._build_simple_block(block_name=layer_name + '_1', no_bias=True))
        self.params += [p for layer in self.block for p in layer.params]
        self.trainable += [p for layer in self.block for p in layer.trainable]
        self.regularizable += [p for layer in self.block for p in layer.regularizable]

        if activation == 'prelu':
            self.alpha = theano.shared(np.float32(.1), layer_name + '_alpha')
            self.params += [self.alpha]
            self.trainable += [self.alpha]
            self.kwargs['alpha'] = self.alpha

        if self.left_branch:
            self.shortcut = []
            self.shortcut.append(ConvNormAct(self.input_shape, int(input_shape[1] * 4 * ratio_n_filter), 1, 'normal',
                                             stride=stride, layer_name=layer_name+'_2', activation='linear', **self.kwargs))
            self.params += [p for layer in self.shortcut for p in layer.params]
            self.trainable += [p for layer in self.shortcut for p in layer.trainable]
            self.regularizable += [p for layer in self.shortcut for p in layer.regularizable]

    def _build_simple_block(self, block_name, no_bias):
        layers = []
        layers.append(ConvNormAct(self.input_shape, int(self.input_shape[1] * self.ratio_n_filter), 1, 'normal',
                                  stride=self.stride, no_bias=no_bias, activation=self.activation, layer_name=block_name+'_conv_bn_act_1', **self.kwargs))

        layers.append(ConvNormAct(layers[-1].output_shape, layers[-1].output_shape[1], 3, stride=(1, 1), border_mode='half',
                                  activation=self.activation, layer_name=block_name+'_conv_bn_act_2', no_bias=no_bias, **self.kwargs))

        layers.append(ConvNormAct(layers[-1].output_shape, layers[-1].output_shape[1] * self.upscale_factor, 1, stride=1,
                                  activation='linear', layer_name=block_name+'_conv_bn_act_3', no_bias=no_bias, **self.kwargs))
        return layers

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

        for layer in self.shortcut:
            layer.reset()

        if self.activation is utils.function['prelu']:
            self.alpha.set_value(np.float32(.1))


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
        super(DenseBlock, self).__init__()
        assert normlization == 'bn' or normlization == 'dbn', \
            'normalization should be either \'bn\' or \'dbn\', got %s' % normlization

        self.input_shape = tuple(input_shape)
        self.transit = transit
        self.num_conv_layer = num_conv_layer
        self.growth_rate = growth_rate
        self.activation = activation
        self.dropout = dropout
        self.pool_transition = pool_transition
        self.layer_name = layer_name
        self.target = target
        self.normalization = BatchNormLayer if normlization == 'bn' else DecorrBatchNormLayer
        self.kwargs = kwargs

        if not self.transit:
            self.block = self._dense_block(self.input_shape, self.num_conv_layer, self.growth_rate, self.dropout,
                                           self.activation, self.layer_name)
        else:
            self.block = self._transition(self.input_shape, self.dropout, self.activation,
                                          self.layer_name + '_transition')

        self.descriptions = '{} Dense Block: {} -> {} {} conv layers growth rate {} transit {} dropout {} {}'.\
            format(layer_name, input_shape, self.output_shape, num_conv_layer, growth_rate, transit, dropout, activation)

    def _bn_act_conv(self, input_shape, num_filters, filter_size, dropout, activation, stride=1, layer_name='bn_re_conv'):
        block = [
            self.normalization(input_shape, activation=activation, layer_name=layer_name + '_bn', **self.kwargs),
            ConvolutionalLayer(input_shape, num_filters, filter_size, He_init='normal', stride=stride,
                               activation='linear', layer_name=layer_name + '_conv')
        ]
        for layer in block:
            self.params += layer.params
            self.trainable += layer.trainable
            self.regularizable += layer.regularizable
        if dropout:
            block.append(DropoutLayer(block[-1].output_shape, dropout, activation='linear',
                                      layer_name=layer_name + 'dropout'))
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
    layers = []

    def __init__(self, input_shape, layer_name='BN', epsilon=1e-4, running_average_factor=1e-1, axes='spatial',
                 activation='relu', no_scale=False, **kwargs):
        '''

        :param input_shape: (int, int, int, int) or (int, int)
        :param layer_name: str
        :param epsilon: float
        :param running_average_factor: float
        :param axes: 'spatial' or 'per-activation'
        '''
        super(BatchNormLayer, self).__init__()

        self.layer_name = layer_name
        self.input_shape = tuple(input_shape)
        self.epsilon = np.float32(epsilon)
        self.running_average_factor = running_average_factor
        self.activation = utils.function[activation]
        self.no_scale = no_scale
        self.training_flag = False
        self.axes = (0,) + tuple(range(2, len(input_shape))) if axes == 'spatial' else (0,)
        self.shape = (self.input_shape[1],) if axes == 'spatial' else self.input_shape[1:]
        self.kwargs = kwargs

        self.gamma_values = np.ones(self.shape, dtype=theano.config.floatX)
        self.gamma = theano.shared(np.copy(self.gamma_values), name=layer_name + '_gamma', borrow=True)

        self.beta_values = np.zeros(self.shape, dtype=theano.config.floatX)
        self.beta = theano.shared(np.copy(self.beta_values), name=layer_name + '_beta', borrow=True)

        self.running_mean = theano.shared(np.zeros(self.shape, dtype=theano.config.floatX),
                                          name=layer_name + '_running_mean', borrow=True)
        self.running_var = theano.shared(np.zeros(self.shape, dtype=theano.config.floatX),
                                         name=layer_name + '_running_var', borrow=True)

        self.params += [self.gamma, self.beta, self.running_mean, self.running_var]
        self.trainable += [self.beta] if self. no_scale else [self.beta, self.gamma]
        self.regularizable += [self.gamma] if not self.no_scale else []

        if activation == 'prelu':
            self.alpha = theano.shared(np.float32(.1), layer_name + '_alpha')
            self.params += [self.alpha]
            self.trainable += [self.alpha]
            self.kwargs['alpha'] = self.alpha

        self.descriptions = '{} BatchNorm Layer: shape: {} -> {} running_average_factor = {:.4f} activation: {}'\
            .format(layer_name, self.input_shape, self.output_shape, self.running_average_factor, activation)
        BatchNormLayer.layers.append(self)

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

    @staticmethod
    def set_training(training):
        for layer in BatchNormLayer.layers:
            layer.training_flag = training


class DecorrBatchNormLayer(Layer):
    """
    From the paper "Decorrelated Batch Normalization" - Lei Huang, Dawei Yang, Bo Lang, Jia Deng
    """
    layers = []

    def __init__(self, input_shape, layer_name='DBN', epsilon=1e-4, running_average_factor=1e-1, activation='relu',
                 no_scale=False, **kwargs):
        '''

        :param input_shape: (int, int, int, int) or (int, int)
        :param layer_name: str
        :param epsilon: float
        :param running_average_factor: float
        :param axes: 'spatial' or 'per-activation'
        '''
        super(DecorrBatchNormLayer, self).__init__()

        self.layer_name = layer_name
        self.input_shape = tuple(input_shape)
        self.epsilon = np.float32(epsilon)
        self.running_average_factor = running_average_factor
        self.activation = utils.function[activation]
        self.no_scale = no_scale
        self.training_flag = False
        self.axes = (0,) #+ tuple(range(2, len(input_shape)))
        self.shape = (self.input_shape[1],)
        self.kwargs = kwargs

        self.gamma_values = np.ones(self.shape, dtype=theano.config.floatX)
        self.gamma = theano.shared(np.copy(self.gamma_values), name=layer_name + '_gamma', borrow=True)

        self.beta_values = np.zeros(self.shape, dtype=theano.config.floatX)
        self.beta = theano.shared(np.copy(self.beta_values), name=layer_name + '_beta', borrow=True)

        self.running_mean = theano.shared(np.zeros(self.shape, dtype=theano.config.floatX),
                                          name=layer_name + '_running_mean', borrow=True)
        self.running_var = theano.shared(np.zeros(self.shape, dtype=theano.config.floatX),
                                         name=layer_name + '_running_var', borrow=True)

        self.params += [self.gamma, self.beta, self.running_mean, self.running_var]
        self.trainable += [self.beta] if self. no_scale else [self.beta, self.gamma]
        self.regularizable += [self.gamma] if not self.no_scale else []

        self.descriptions = '{} DecorrelatedBatchNorm Layer: shape: {} -> {} running_average_factor = {:.4f} activation: {}'\
            .format(layer_name, self.input_shape, self.output_shape, self.running_average_factor, activation)
        DecorrBatchNormLayer.layers.append(self)

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

    @staticmethod
    def set_training(training):
        for layer in BatchNormLayer.layers:
            layer.training_flag = training


class GroupNormLayer(Layer):
    """
    Implementation of the paper "Group Normalization" - Wu et al.
    group = 1 -> Insntance Normalization
    group = input_shape[1] -> Layer Normalization
    """
    def __init__(self, input_shape, layer_name='GN', groups=32, epsilon=1e-4, activation='relu', **kwargs):
        super(GroupNormLayer, self).__init__()
        assert input_shape[1] / groups == input_shape[1] // groups, 'groups must divide the number of input channels.'

        self.layer_name = layer_name
        self.input_shape = tuple(input_shape)
        self.groups = groups
        self.epsilon = np.float32(epsilon)
        self.activation = utils.function[activation]
        self.kwargs = kwargs
        self.gamma_values = np.ones(self.input_shape[1], dtype=theano.config.floatX)
        self.gamma = theano.shared(np.copy(self.gamma_values), name=layer_name + '_gamma', borrow=True)

        self.beta_values = np.zeros(self.input_shape[1], dtype=theano.config.floatX)
        self.beta = theano.shared(np.copy(self.beta_values), name=layer_name + '_beta', borrow=True)

        self.params += [self.gamma, self.beta]
        self.trainable += [self.gamma, self.beta]
        self.regularizable += [self.gamma]

        if activation == 'prelu':
            self.alpha = theano.shared(np.float32(.1), layer_name + '_alpha')
            self.params += [self.alpha]
            self.trainable += [self.alpha]
            self.kwargs['alpha'] = self.alpha

        self.descriptions = '{} GroupNorm Layer: shape: {} -> {} activation: {}'\
            .format(layer_name, self.input_shape, self.output_shape, activation)

    def get_output(self, input):
        n, c, h, w = T.shape(input)
        input = T.reshape(input, (n, self.groups, -1, h, w))
        mean = T.mean(input, (2, 3, 4), keepdims=True)
        var = T.var(input, (2, 3, 4), keepdims=True)
        gamma = self.gamma.dimshuffle(('x', 0, 'x', 'x'))
        beta = self.beta.dimshuffle(('x', 0, 'x', 'x'))
        input = (input - mean) / T.sqrt(var + self.epsilon)
        input = T.reshape(input, (n, c, h, w))
        output = gamma * input + beta
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
    layers = []

    def __init__(self, input_shape, layer_name='BRN', epsilon=1e-4, r_max=1, d_max=0, running_average_factor=0.1,
                 axes='spatial', activation='relu'):
        '''

        :param input_shape: (int, int, int, int) or (int, int)
        :param layer_name: str
        :param epsilon: float
        :param running_average_factor: float
        :param axes: 'spatial' or 'per-activation'
        '''
        super(BatchRenormLayer, self).__init__()

        self.layer_name = layer_name
        self.input_shape = tuple(input_shape)
        self.epsilon = epsilon
        self.running_average_factor = running_average_factor
        self.activation = utils.function[activation]
        self.training_flag = False
        self.r_max = theano.shared(np.float32(r_max), name=layer_name + 'rmax')
        self.d_max = theano.shared(np.float32(d_max), name=layer_name + 'dmax')
        self.axes = (0,) + tuple(range(2, len(input_shape))) if axes == 'spatial' else (0,)
        self.shape = (self.input_shape[1],) if axes == 'spatial' else self.input_shape[1:]

        self.gamma_values = np.ones(self.shape, dtype=theano.config.floatX)
        self.beta_values = np.zeros(self.shape, dtype=theano.config.floatX)
        self.gamma = theano.shared(self.gamma_values, name=layer_name + '_gamma', borrow=True)
        self.beta = theano.shared(self.beta_values, name=layer_name + '_beta', borrow=True)
        self.running_mean = theano.shared(np.zeros(self.shape, dtype=theano.config.floatX),
                                          name=layer_name + '_running_mean', borrow=True)
        self.running_var = theano.shared(np.zeros(self.shape, dtype=theano.config.floatX),
                                         name=layer_name + '_running_var', borrow=True)
        self.params += [self.gamma, self.beta, self.running_mean, self.running_var]
        self.trainable += [self.gamma, self.beta]
        self.regularizable.append(self.gamma)
        self.descriptions = '{} Batch Renorm Layer: running_average_factor = {:.4f}'.format(layer_name, self.running_average_factor)
        BatchRenormLayer.layers.append(self)

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

    @staticmethod
    def set_training(training):
        for layer in BatchRenormLayer.layers:
            layer.training_flag = training


class TransformerLayer(Layer):
    """Implementation of the bilinear interpolation transformer layer in https://arxiv.org/abs/1506.020250. Based on
    the implementation in Lasagne.
    coordinates is a tensor of shape (n, 2, h, w). coordinates[:, 0, :, :] is the horizontal coordinates and the other
    is the vertical coordinates.
    """

    def __init__(self, input_shapes, downsample_factor=1, border_mode='nearest', layer_name='Warping', **kwargs):
        assert isinstance(input_shapes, (list, tuple)), 'input_shapes must be a list a tuple of shapes, got %s.' % type(input_shapes)

        super(TransformerLayer, self).__init__()
        self.input_shape, self.transform_shape = tuple(input_shapes)
        self.layer_name = layer_name
        self.downsample_factor = (downsample_factor, downsample_factor) if isinstance(downsample_factor, int) else tuple(downsample_factor)
        self.border_mode = border_mode
        self.kwargs = kwargs
        self.descriptions = '%s Warping layer.' % layer_name

    @property
    def output_shape(self):
        shape = self.input_shape
        factors = self.downsample_factor
        return (shape[:2] + tuple(None if s is None else int(s // f) for s, f in zip(shape[2:], factors)))

    def get_output(self, inputs):
        input, theta = inputs
        return utils.transform_affine(theta, input, self.downsample_factor, self.border_mode)


class IdentityLayer(Layer):
    def __init__(self, input_shape, layer_name='Identity'):
        super(IdentityLayer, self).__init__()
        self.input_shape = tuple(input_shape)
        self.layer_name = layer_name
        self.descriptions = '%s Identity layer.' % layer_name

    @property
    def output_shape(self):
        return tuple(self.input_shape)

    def get_output(self, input):
        return input


class ResizingLayer(Layer):
    def __init__(self, input_shape, ratio=None, frac_ratio=None, layer_name='Upsampling'):
        super(ResizingLayer, self).__init__()
        if ratio != int(ratio):
            raise NotImplementedError
        if ratio and frac_ratio:
            raise NotImplementedError
        self.input_shape = tuple(input_shape)
        self.ratio = ratio
        self.frac_ratio = frac_ratio
        self.layer_name = layer_name
        self.descriptions = '{} x{} Resizing Layer {} -> {}'.format(layer_name, self.ratio, self.input_shape, self.output_shape)

    def get_output(self, input):
        return T.nnet.abstract_conv.bilinear_upsampling(input, ratio=self.ratio) if self.frac_ratio is None \
            else T.nnet.abstract_conv.bilinear_upsampling(input, frac_ratio=self.frac_ratio)

    @property
    @validate
    def output_shape(self):
        return (self.input_shape[0], self.input_shape[1], self.input_shape[2] * self.ratio, self.input_shape[3] * self.ratio)


class ReshapingLayer(Layer):
    def __init__(self, input_shape, new_shape, layer_name='reshape'):
        super(ReshapingLayer, self).__init__()
        self.input_shape = tuple(input_shape)
        self.new_shape = tuple(new_shape)
        self.layer_name = layer_name
        self.descriptions = 'Reshaping Layer: {} -> {}'.format(self.input_shape, self.output_shape)

    def get_output(self, input):
        return T.reshape(input, self.new_shape)

    @property
    @validate
    def output_shape(self):
        if self.new_shape[0] == -1:
            output = list(self.new_shape)
            output[0] = None
            return tuple(output)
        else:
            prod_shape = np.prod(self.input_shape[1:])
            prod_new_shape = np.prod(self.new_shape) * -1
            shape = [x if x != -1 else prod_shape // prod_new_shape for x in self.input_shape]
            return tuple(shape)


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
        super(SlicingLayer, self).__init__()
        self.input_shape = tuple(input_shape)
        self.from_idx = from_idx
        self.to_idx = to_idx
        self.axes = axes
        self.layer_name = layer_name
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
        super(ConcatLayer, self).__init__()
        self.input_shapes = tuple(input_shapes)
        self.axis = axis
        self.layer_name = layer_name
        self.descriptions = ''.join(('%s Concat Layer: axis %d' % (layer_name, axis), ' '.join([str(x) for x in input_shapes]),
                                     ' -> {}'.format(self.output_shape)))

    def get_output(self, input):
        return T.concatenate(input, self.axis)

    @property
    def output_shape(self):
        shape = np.array(self.input_shapes)
        depth = np.sum(shape[:, 1])
        return (self.input_shapes[0][0], depth, self.input_shapes[0][2], self.input_shapes[0][3])


class SumLayer(Layer):
    def __init__(self, input_shapes, weight=1., layer_name='SumLayer'):
        super(SumLayer, self).__init__()
        self.input_shapes = tuple(input_shapes)
        self.weight = weight
        self.layer_name = layer_name
        self.descriptions = '%s Sum Layer: weight %d' % (layer_name, weight)

    def get_output(self, input):
        assert isinstance(input, (list, tuple)), 'Input must be a list or tuple of same-sized tensors.'
        return sum(input) * np.float32(self.weight)

    @property
    def output_shape(self):
        return tuple(self.input_shapes[0])


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
        super(ScalingLayer, self).__init__()

        self.input_shape = tuple(input_shape)
        if shared_axes == 'auto':
            # default: share scales over all but the second axis
            shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(shared_axes, int):
            shared_axes = (shared_axes,)
        self.shared_axes = shared_axes
        self.layer_name = layer_name
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
        return tuple(self.input_shape)


class Gate(object):
    def __init__(self, n_in, n_hid, use_peephole=False, W_hid=None, W_cell=False, bias_init_range=(-.5, .5),
                 layer_name='gate'):
        super(Gate, self).__init__()

        self.n_in = n_in
        self.n_hid = n_hid
        self.bias_init_range = bias_init_range
        self.layer_name = layer_name
        self.use_peephole = use_peephole
        if not use_peephole:
            self.W_hid = theano.shared(self.sample_weights(n_in + n_hid, n_hid), name=layer_name + '_Whid') if W_hid is None else W_hid
        else:
            self.W_hid = theano.shared(self.sample_weights(n_hid, n_hid), name=layer_name + '_Whid_ph') if isinstance(W_cell, int) else W_cell
        self.b = theano.shared(np.cast[theano.config.floatX](np.random.uniform(bias_init_range[0], bias_init_range[1],
                                                                               size=n_hid)), name=layer_name + '_b')

        self.trainable = [self.W_in, self.W_hid, self.W_cell, self.b] if W_cell else [self.W_in, self.W_hid, self.b]
        self.regularizable = [self.W_in, self.W_hid, self.W_cell, self.b] if W_cell else [self.W_in, self.W_hid, self.b]

    def sample_weights(self, sizeX, sizeY):
        values = np.ndarray([sizeX, sizeY], dtype=theano.config.floatX)
        for dx in range(sizeX):
            vals = np.random.uniform(low=-1., high=1., size=(sizeY,))
            # vals_norm = np.sqrt((vals**2).sum())
            # vals = vals / vals_norm
            values[dx, :] = vals
        _, svs, _ = np.linalg.svd(values)
        # svs[0] is the largest singular value
        values = values / svs[0]
        return values


class LSTMcell(Layer):
    def __init__(self, n_in, n_hid, use_peephole=False, tensordot=False, gradien_clipping=False, layer_name='lstmcell'):
        super(LSTMcell, self).__init__()

        self.n_in = n_in
        self.n_hid = n_hid
        self.use_peephole = False
        self.tensordot = tensordot
        self.gradient_clipping = gradien_clipping
        self.layer_name = layer_name
        self.in_gate = Gate(n_in, n_hid, W_cell=True, layer_name='in_gate')
        self.forget_gate = Gate(n_hid, n_hid, W_cell=True, bias_init_range=(0., 1.), layer_name='forget_gate')
        self.cell_gate = Gate(n_hid, n_hid, W_cell=False, layer_name='cell_gate')
        self.out_gate = Gate(n_hid, n_hid, W_cell=True, layer_name='out_gate')
        self.c0 = theano.shared(np.zeros((n_hid, ), dtype=theano.config.floatX), 'first_cell_state')
        self.h0 = utils.function['tanh'](self.c0)
        self.trainable += self.in_gate.trainable + self.forget_gate.trainable + self.cell_gate.trainable + self.out_gate.trainable + \
                          [self.c0]
        self.regularizable += self.in_gate.regularizable + self.forget_gate.regularizable + \
                              self.cell_gate.regularizable + self.out_gate.regularizable + [self.c0]
        print('@ %s LSTMCell: shape = (%d, %d)' % (self.layer_name, n_in, n_hid))

    def get_output_onestep(self, x_t, h_tm1, c_tm1, W_i, b_i, W_f, b_f, W_c, b_c, W_o, b_o):
        from theano import dot
        inputs = T.concatenate((x_t, h_tm1), 1)
        W = T.concatenate((W_i, W_f, W_c, W_o), 1)


        def slice_w(x, n):
            s = x[:, n*self.n_hid:(n+1)*self.n_hid]
            if self.n_hid == 1:
                s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
            return s

        xt_dot_W = dot(x_t, W_x)
        htm1_dot_W = dot(h_tm1, W_htm1)
        ctm1_dot_W = dot(c_tm1, W_ctm1)
        i_t = utils.function['sigmoid'](_slice(xt_dot_W, 0, self.n_hid) + dot(h_tm1, W_hi) + dot(c_tm1, W_ci) + b_i)
        f_t = utils.function['sigmoid'](dot(x_t, W_xf) + dot(h_tm1, W_hf) + dot(c_tm1, W_cf) + b_f)
        c_t = f_t * c_tm1 + i_t * utils.function['tanh'](dot(x_t, W_xc) + dot(h_tm1, W_hc) + b_c)
        o_t = utils.function['sigmoid'](dot(x_t, W_xo) + dot(h_tm1, W_ho) + dot(c_t, W_co) + b_o)
        h_t = o_t * utils.function['tanh'](c_t)
        return h_t, c_t

    def get_output(self, input):
        non_sequences = list(self.trainable)
        non_sequences.pop()
        [h_vals, _], _ = theano.scan(LSTMcell.get_output_onestep, sequences=dict(input=input, taps=[0]),
                                     outputs_info=[self.h0, self.c0], non_sequences=non_sequences)
        return h_vals

    def output_shape(self):
        return None, self.n_in


def set_training_status(training):
    DropoutLayer.set_training(training)
    BatchNormLayer.set_training(training)
    BatchRenormLayer.set_training(training)
    DecorrBatchNormLayer.set_training(training)
