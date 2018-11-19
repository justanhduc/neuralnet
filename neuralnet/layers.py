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
from theano.tensor.nnet import conv3d as conv3
from collections import OrderedDict
from functools import partial

from neuralnet import utils
from neuralnet.init import *

__all__ = ['Layer', 'Sequential', 'ConvolutionalLayer', 'FullyConnectedLayer', 'SpatialTransformerLayer',
           'TransposedConvolutionalLayer', 'DenseBlock', 'SumLayer', 'StackingConv', 'ScalingLayer',
           'SlicingLayer', 'LSTMCell', 'ActivationLayer', 'AttConvLSTMCell', 'ConcatLayer', 'ConvLSTMCell',
           'ConvNormAct', 'RecursiveResNetBlock', 'ResNetBlock', 'ResNetBottleneckBlock', 'GRUCell',
           'IdentityLayer', 'DropoutLayer', 'InceptionModule1', 'InceptionModule2', 'InceptionModule3',
           'NetworkInNetworkBlock', 'SoftmaxLayer', 'TransposingLayer', 'set_training_status', 'WarpingLayer',
           'NoiseResNetBlock', 'set_training_on', 'set_training_off', 'LambdaLayer', 'Conv2DLayer',
           'Deconv2DLayer', 'FCLayer', 'SqueezeAndExcitationBlock', 'Conv3DLayer', 'RNNBlockDNN']


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


set_training_on = partial(Layer.set_training_status, True)
set_training_off = partial(Layer.set_training_status, False)


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
            keys = keys[item]
            sl = [self[k] for k in keys]
            return Sequential(sl, layer_name=self.layer_name)
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

        if isinstance(other, (Layer, Sequential)):
            if other.layer_name not in self.keys():
                self[other.layer_name] = other
            else:
                raise NameError('Name %s already existed.' % other.layer_name)
        else:
            raise TypeError('Cannot update a Sequential instance with a %s instance.' % type(other))

    def extend(self, others):
        if isinstance(others, (list, tuple, Sequential)):
            for layer in others:
                self.append(layer)
        else:
            raise TypeError('Cannot extend a Sequential instance with a %s instance.' % type(others))

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


class LambdaLayer(Layer):
    def __init__(self, input_shape, function, layer_name='Lambda Layer', **kwargs):
        assert callable(function), 'The provided function must be callable.'

        super(LambdaLayer, self).__init__(input_shape, layer_name)
        self.function = function
        self.kwargs = kwargs
        self.descriptions = '{} Lambda Layer'.format(layer_name)

    def get_output(self, input):
        return self.function(input, **self.kwargs)

    @property
    def output_shape(self):
        input_shape = (0 if s is None else s for s in self.input_shape)
        X = theano.tensor.alloc(0, *input_shape)
        output_shape = self.function(X, **self.kwargs).shape.eval()
        output_shape = tuple(s if s else None for s in output_shape)
        return output_shape


class ActivationLayer(Layer):
    def __init__(self, input_shape, activation='relu', layer_name='Activation', **kwargs):
        super(ActivationLayer, self).__init__(input_shape, layer_name)
        self.activation = utils.function[activation] if not callable(activation) else activation
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


class DropoutLayer(Layer):
    def __init__(self, input_shape, drop_prob=0.5, gaussian=False, position='per-activation', layer_name='Dropout'):
        assert len(input_shape) == 2 or len(input_shape) == 4, \
            'input_shape must have 2 or 4 elements. Received %d' % len(input_shape)
        assert type in ('per-activation', 'per-channel'), 'Unknown dropout position.'

        super(DropoutLayer, self).__init__(input_shape, layer_name)
        self.gaussian = gaussian
        self.srng = theano.sandbox.rng_mrg.MRG_RandomStreams(self.rng.randint(1, int(time.time())))
        self.dropout_prob = drop_prob
        self.position = position
        self.descriptions = '{} Dropout Layer: p={:.2f}'.format(layer_name, 1. - drop_prob)

    def get_output(self, input):
        keep_prob = T.constant((1 - self.dropout_prob) if self.training_flag and not self.gaussian else 1, 'keep_prob',
                               dtype=theano.config.floatX)
        if self.training_flag:
            shape = input.shape if self.position == 'per-activation' else input.shape[:2]
            mask = self.srng.normal(shape) + 1. if self.gaussian \
                else self.srng.binomial(n=1, p=keep_prob, size=shape, dtype=theano.config.floatX)
            if self.position == 'per-channel':
                mask = mask.dimshuffle(0, 1, 'x', 'x')
            input = input * mask
        return input / keep_prob

    @property
    @utils.validate
    def output_shape(self):
        return tuple(self.input_shape)


class FullyConnectedLayer(Layer):
    def __init__(self, input_shape, num_nodes, init=HeNormal(gain=1.), no_bias=False, layer_name='FC Layer',
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
        self.activation = utils.function[activation] if not callable(activation) else activation
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
        input = T.unbroadcast(input, 0)
        output = T.dot(input.flatten(2), self.W) + self.b if not self.no_bias else T.dot(input.flatten(2), self.W)
        return self.activation(output, **self.kwargs) if self.keep_dims else T.squeeze(
            self.activation(output, **self.kwargs))

    @property
    @utils.validate
    def output_shape(self):
        return (self.input_shape[0], self.num_nodes // self.kwargs.get('maxout_size')) if self.activation is 'maxout' \
            else (self.input_shape[0], self.num_nodes)

    def reset(self):
        self.W.set_value(np.copy(self.W_values))
        if not self.no_bias:
            self.b.set_value(np.copy(self.b_values))
        if self.activation is utils.function['prelu']:
            self.alpha.set_value(np.float32(.1))


FCLayer = FullyConnectedLayer


class SqueezeAndExcitationBlock(Sequential):
    def __init__(self, input_shape, ratio=4, activation='relu', layer_name='SE Block', **kwargs):
        super(SqueezeAndExcitationBlock, self).__init__(input_shape=input_shape, layer_name=layer_name)
        from neuralnet.resizing import GlobalAveragePoolingLayer
        self.ratio = ratio
        self.activation = activation
        self.kwargs = kwargs
        self.descriptions = '{} SE Block: ratio {} act {}'.format(layer_name, ratio, activation)
        block = Sequential(input_shape=input_shape, layer_name=layer_name+'/scaling')
        block.append(GlobalAveragePoolingLayer(block.output_shape, block.layer_name+'/gap'))
        block.append(FCLayer(block.output_shape, input_shape[1] // ratio, activation=activation,
                             layer_name=block.layer_name + '/fc1'))
        block.append(
            FCLayer(block.output_shape, input_shape[1], activation='sigmoid', layer_name=block.layer_name + '/fc2'))
        self.append(block)

    def get_output(self, input):
        scale = self[self.layer_name+'/scaling'](input)
        return input * scale.dimshuffle(0, 1, 'x', 'x')

    @property
    @utils.validate
    def output_shape(self):
        return self.input_shape


class ConvolutionalLayer(Layer):
    def __init__(self, input_shape, num_filters, filter_size, init=HeNormal(gain=1.), no_bias=True, border_mode='half',
                 stride=1, dilation=1, layer_name='Conv Layer', activation='relu', **kwargs):
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
        assert len(input_shape) in (4, 5), 'input_shape must have 4 or 5 elements. Received %d' % len(input_shape)
        assert isinstance(num_filters, int) and isinstance(filter_size, (int, list, tuple))
        assert isinstance(border_mode, (int, list, tuple, str)), 'border_mode should be either \'int\', ' \
                                                                 '\'list\', \'tuple\' or \'str\', got {}'.format(type(border_mode))
        assert isinstance(stride, (int, list, tuple))

        super(ConvolutionalLayer, self).__init__(input_shape, layer_name)
        if len(input_shape) == 4:
            self.filter_shape = (num_filters, input_shape[1], filter_size[0], filter_size[1]) if isinstance(filter_size, (
                list, tuple)) else (num_filters, input_shape[1], filter_size, filter_size)
        else:
            self.filter_shape = (
                num_filters, input_shape[1], filter_size[0], filter_size[1], filter_size[2]) if isinstance(filter_size, (
                list, tuple)) else (num_filters, input_shape[1], filter_size, filter_size, filter_size)
        self.no_bias = no_bias
        self.activation = utils.function[activation] if not callable(activation) else activation
        self.border_mode = border_mode
        self.subsample = tuple(stride) if isinstance(stride, (tuple, list)) else (stride, stride) if len(
            input_shape) == 4 else (stride, stride, stride)
        self.dilation = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation) if len(
            input_shape) == 4 else (dilation, dilation, dilation)
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
        if self.border_mode in ('ref', 'rep'):
            assert len(self.input_shape) == 4, '\'ref\' and \'rep\' padding modes support only 4D input'
            if self.border_mode == 'ref':
                output = utils.reflect_pad(input, (self.filter_shape[2] >> 1, self.filter_shape[3] >> 1))
            else:
                output = utils.replication_pad(input, (self.filter_shape[2] >> 1, self.filter_shape[3] >> 1))
            output = conv(input=output, filters=self.W, border_mode='valid', subsample=self.subsample,
                          filter_flip=self.filter_flip, filter_shape=self.filter_shape)
        else:
            output = conv(input=input, filters=self.W, border_mode=self.border_mode, subsample=self.subsample,
                          filter_flip=self.filter_flip, filter_shape=self.filter_shape) if len(
                self.input_shape) == 4 else conv3(input, self.W, border_mode=self.border_mode, subsample=self.subsample,
                                                  filter_flip=self.filter_flip, filter_dilation=self.dilation)
        if not self.no_bias:
            output += self.b.dimshuffle(('x', 0, 'x', 'x')) if len(self.input_shape) == 4 else self.b.dimshuffle(
                ('x', 0, 'x', 'x', 'x'))
        return self.activation(output, **self.kwargs)

    @property
    @utils.validate
    def output_shape(self):
        shape = list(self.input_shape)
        ks = [fs + (fs - 1) * (d - 1) for fs, d in zip(self.filter_shape[2:], self.dilation)]

        if isinstance(self.border_mode, str):
            if self.border_mode in ('half', 'ref', 'rep'):
                p = [k >> 1 for k in ks]
            elif self.border_mode == 'valid':
                p = [0] * len(ks)
            elif self.border_mode == 'full':
                p = [k - 1 for k in ks]
            else:
                raise NotImplementedError
        elif isinstance(self.border_mode, (list, tuple)):
            p = tuple(self.border_mode)
        elif isinstance(self.border_mode, int):
            p = [self.border_mode] * len(ks)
        else:
            raise NotImplementedError

        shape[2:] = [(s - ks[idx] + 2 * p[idx]) // self.subsample[idx] + 1 for idx, s in enumerate(shape[2:])]
        shape[1] = self.filter_shape[0] // self.kwargs.get('maxout_size') if self.activation == utils.maxout \
            else self.filter_shape[0]
        return tuple(shape)

    def reset(self):
        self.W.set_value(np.copy(self.W_values))
        if not self.no_bias:
            self.b.set_value(np.copy(self.b_values))
        if self.activation is utils.function['prelu']:
            self.alpha.set_value(np.float32(.1))


Conv2DLayer = ConvolutionalLayer
Conv3DLayer = ConvolutionalLayer


class PerturbativeLayer(Layer):
    def __init__(self, input_shape, num_filters, init=HeNormal(), noise_level=.1, activation='relu', no_bias=True,
                 layer_name='Perturbative Layer', **kwargs):
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(PerturbativeLayer, self).__init__(input_shape, layer_name)
        self.num_filters = num_filters
        self.noise_level = noise_level
        self.activation = utils.function[activation] if not callable(activation) else activation
        self.no_bias = no_bias
        self.kwargs = kwargs

        self.noise = (2. * T.as_tensor_variable(np.random.rand(*input_shape[1:]).astype(theano.config.floatX)) - 1.) * noise_level
        self.W_values = init((num_filters, input_shape[1]))
        self.W = theano.shared(self.W_values, layer_name+'/W', borrow=True)
        self.params += [self.W]
        self.trainable += [self.W]
        self.regularizable += [self.W]

        if not no_bias:
            self.b = theano.shared(np.zeros((num_filters,), theano.config.floatX), layer_name+'/b', borrow=True)
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
        output = conv(input, kern, border_mode='half')
        return output if self.no_bias else output + self.b.dimshuffle('x', 0, 'x', 'x')

    @property
    def output_shape(self):
        return (self.input_shape[0],) + (self.num_filters,) + self.input_shape[2:]

    def reset(self):
        self.W.set_value(self.W_values)
        if self.activation is utils.function['prelu']:
            self.alpha.set_value(.1)
        if not self.no_bias:
            self.b.set_value(np.zeros((self.num_filters,), theano.config.floatX))


class InceptionModule1(Layer):
    def __init__(self, input_shape, num_filters=48, border_mode='half', stride=(1, 1), activation='relu',
                 layer_name='Inception Type 1'):
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(InceptionModule1, self).__init__(input_shape, layer_name)
        from .resizing import PoolingLayer
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
                 layer_name='Inception Type 2'):
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(InceptionModule2, self).__init__(input_shape, layer_name)
        from .resizing import PoolingLayer
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
                 layer_name='Inception Type 3'):
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(InceptionModule3, self).__init__(input_shape, layer_name)
        from .resizing import PoolingLayer
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
                 layer_name='Deconv2D', padding='half', stride=(2, 2), activation='relu', **kwargs):
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
        self.activation = utils.function[activation] if not callable(activation) else activation
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
    @utils.validate
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


Deconv2DLayer = TransposedConvolutionalLayer


class ResNetBlock(Sequential):
    upscale_factor = 1

    def __init__(self, input_shape, num_filters, stride=(1, 1), dilation=(1, 1), activation='relu', downsample=None,
                 layer_name='Res Block', normalization='bn', groups=32, block=None, se_block=False, **kwargs):
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
        from neuralnet.normalization import BatchNormLayer, GroupNormLayer
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
        self.se_block = se_block
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

        if se_block:
            ratio = kwargs.pop('ratio', 4)
            self.append(SqueezeAndExcitationBlock(self.output_shape, ratio, activation, layer_name+'/se', **kwargs))

        self.descriptions = '{} ResNet Basic Block {} -> {} {} filters stride {} dilation {} {} {}'.\
            format(layer_name, self.input_shape, self.output_shape, num_filters, stride, dilation, activation,
                   ' '.join([' '.join((k, str(v))) for k, v in kwargs.items()]))

    def _build_simple_block(self, block_name):
        from neuralnet.normalization import BatchNormLayer, GroupNormLayer
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

        if self.se_block:
            output = self[self.layer_name+'/se'](output)
        return utils.function[self.activation](output + res, **self.kwargs)

    def reset(self):
        super(ResNetBlock, self).reset()
        if self.activation == 'prelu':
            self.alpha.set_value(np.float32(.1))


class ResNetBottleneckBlock(Sequential):
    upscale_factor = 4

    def __init__(self, input_shape, num_filters, stride=1, dilation=(1, 1), activation='relu', downsample=False,
                 layer_name='Res Bottleneck Block', normalization='bn', block=None, **kwargs):
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
                 layer_name='Noise Res Block', normalization='bn', groups=32, **kwargs):
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
        from neuralnet.normalization import BatchNormLayer, GroupNormLayer
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
        from neuralnet.normalization import BatchNormLayer, GroupNormLayer
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
    @utils.validate
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
                 layer_name='Recursive Res Block', normalization='bn', groups=32, **kwargs):
        assert normalization in (
        None, 'bn', 'gn'), 'normalization must be either None, \'bn\' or \'gn\', got %s.' % normalization
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(RecursiveResNetBlock, self).__init__(input_shape, layer_name)
        from neuralnet.normalization import BatchNormLayer, GroupNormLayer
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
        from neuralnet.normalization import BatchNormLayer
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
                 layer_name='Dense Block', pool_transition=True, normlization='bn', target='dev0', **kwargs):
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
        from neuralnet.normalization import BatchNormLayer, GroupNormLayer
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
        from .resizing import PoolingLayer
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
    @utils.validate
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


class SpatialTransformerLayer(Layer):
    """Implementation of the bilinear interpolation transformer layer in https://arxiv.org/abs/1506.020250. Based on
    the implementation in Lasagne.
    coordinates is a tensor of shape (n, 2, h, w). coordinates[:, 0, :, :] is the horizontal coordinates and the other
    is the vertical coordinates.
    """

    def __init__(self, input_shape, downsample_factor=1, border_mode='nearest', layer_name='Transformer', dnn=True,
                 **kwargs):
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)

        super(SpatialTransformerLayer, self).__init__(input_shape, layer_name)
        self.downsample_factor = (downsample_factor, downsample_factor) if isinstance(downsample_factor,
                                                                                      int) else tuple(downsample_factor)
        self.border_mode = border_mode
        self.dnn = dnn
        self.kwargs = kwargs
        self.descriptions = '%s Transformer layer.' % layer_name

    @property
    def output_shape(self):
        shape = self.input_shape
        factors = self.downsample_factor
        return tuple(list(shape[:2]) + [None if s is None else int(s // f) for s, f in zip(shape[2:], factors)])

    def get_output(self, inputs):
        input, theta = inputs
        return theano.gpuarray.dnn.dnn_spatialtf(input, theta, 1. / np.float32(self.downsample_factor[0]),
                                                 1. / np.float32(self.downsample_factor[
                                                                     1])) if self.dnn else utils.transform_affine(theta,
                                                                                                                  input,
                                                                                                                  self.downsample_factor,
                                                                                                                  self.border_mode)


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
                            T.as_tensor_variable(np.linspace(-1.0, 1.0, width, dtype=theano.config.floatX)).dimshuffle(0, 'x'),
                            [1, 0]))
            y_t = T.dot(T.as_tensor_variable(np.linspace(-1.0, 1.0, height, dtype=theano.config.floatX)).dimshuffle(0, 'x'),
                        T.ones(shape=[1, width]))
            x_t_flat = T.reshape(x_t, (1, -1))
            y_t_flat = T.reshape(y_t, (1, -1))
            grid_x = T.reshape(x_t_flat, [1, height, width])
            grid_y = T.reshape(y_t_flat, [1, height, width])
            return grid_x, grid_y

        gx, gy = meshgrid(self.input_shape[2], self.input_shape[3])
        gx = T.as_tensor_variable(gx, ndim=2).astype(theano.config.floatX).dimshuffle('x', 0, 1)
        gy = T.as_tensor_variable(gy, ndim=2).astype(theano.config.floatX).dimshuffle('x', 0, 1)
        x_coor = gx + flow[:, 0]
        y_coor = gy + flow[:, 1]
        output = utils.interpolate_bilinear(image, x_coor, y_coor, border_mode=self.border_mode)
        return output

    @property
    @utils.validate
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


class SlicingLayer(Layer):
    def __init__(self, input_shape, to_idx, from_idx=(0, 0), axes=(2, 3), layer_name='Slicing Layer'):
        '''

        :param input_shape: (int, int, int, int)
        :param to_idx:
        :param from_idx:
        :param axes:
        :param layer_name:
        '''
        assert isinstance(to_idx, (int, list, tuple)) and isinstance(to_idx, (int, list, tuple)) and isinstance(to_idx,
                                                                                                                (int,
                                                                                                                 list,
                                                                                                                 tuple))
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
    @utils.validate
    def output_shape(self):
        shape = list(self.input_shape)
        for idx, axis in enumerate(self.axes):
            shape[axis] = self.to_idx[idx] - self.from_idx[idx]
        return shape


class ConcatLayer(Layer):
    def __init__(self, input_shapes, axis=1, layer_name='Concat Layer'):
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
    def __init__(self, input_shape, transpose, layer_name='Transpose Layer'):
        super(TransposingLayer, self).__init__(input_shape, layer_name)
        self.transpose = transpose
        self.descriptions = '{} Transposing layer: {} -> {}'.format(layer_name, [i for i in range(len(input_shape))],
                                                                    transpose)

    def get_output(self, input):
        return T.transpose(input, self.transpose)

    @property
    @utils.validate
    def output_shape(self):
        return tuple([self.input_shape[i] if isinstance(i, int) else 1 for i in self.transpose])


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
    def __init__(self, input_shape, scales=1, shared_axes='auto', layer_name='Scale Layer'):
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
    @utils.validate
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
        self.activation = utils.function[activation] if not callable(activation) else activation


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
        self.activation = utils.function[activation] if not callable(activation) else activation
        self.kwargs = kwargs

        n_in = self.input_shape[-1]
        self.in_gate = Gate((n_in, num_units), W_cell=False, layer_name='in_gate')
        self.forget_gate = Gate((n_in, num_units), W_cell=False, layer_name='forget_gate')
        self.cell_gate = Gate((n_in, num_units), W_cell=False, layer_name='cell_gate')
        self.out_gate = Gate((n_in, num_units), W_cell=False, layer_name='out_gate')

        self.cell_init = theano.shared(np.zeros((1, num_units), theano.config.floatX), 'cell_init')
        self.hid_init = theano.shared(np.zeros((1, num_units), theano.config.floatX), 'hid_init')
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

        ones = T.ones((num_batch, 1), theano.config.floatX)
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


class RNNBlockDNN(Layer):
    def __init__(self, input_shape, num_units, depth=1, type='lstm', direction='unidirectional', learn_init=False,
                 layer_name='RNNBlock'):
        """

        :param input_shape: (timesteps, batch_size, input_dim)
        :param num_units:
        :param depth:
        :param type:
        :param direction:
        :param learn_init:
        :param layer_name:
        """
        super(RNNBlockDNN, self).__init__(input_shape, layer_name)
        self.num_units = num_units
        self.depth = depth
        self.type = type
        self.direction = direction
        self.learn_init = learn_init
        self.rnn = theano.gpuarray.dnn.RNNBlock(theano.config.floatX, num_units, depth, type)

        n_in = int(np.prod(input_shape[2:]))
        psize = self.rnn.get_param_size((input_shape[1], n_in))
        self.W = theano.shared(np.zeros(psize, theano.config.floatX), layer_name + '_' + type + '/W')
        self.params.append(self.W)
        self.trainable.append(self.W)
        self.regularizable.append(self.W)

        self.init = [theano.shared(np.zeros((1, 1, num_units), theano.config.floatX), layer_name + '_' + type + '/hid_init')]
        if type == 'lstm':
            self.init.append(
                theano.shared(np.zeros((1, 1, num_units), theano.config.floatX), layer_name + '_' + type + '/cell_init'))
        if learn_init:
            self.trainable += self.init

        self.descriptions = '%s RNNBlock: %s shape = (%d, %d)' % (self.layer_name, type, n_in, num_units)

    def get_output(self, input, **kwargs):
        return_hid = kwargs.get('hid', False)
        return self.rnn.apply(self.W, input, *self.init) if return_hid else self.rnn.apply(self.W, input, *self.init)[0]

    @property
    def output_shape(self):
        return tuple([self.input_shape[0], self.num_units])


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

        self.hid_init = theano.shared(np.zeros((1, num_units), theano.config.floatX), 'hid_init')
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

        hid_init = T.dot(T.ones((num_batch, 1), theano.config.floatX), self.hid_init)
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
                 grad_clip=0, activation='tanh', layer_name='Conv LSTMCell', **kwargs):
        assert isinstance(input_shape, (list, tuple)), 'input_shape must be a list or tuple, got %s.' % input_shape
        assert len(input_shape) == 5, 'input_shape must contain exactly 5 elements, got %d.' % len(input_shape)

        super(ConvLSTMCell, self).__init__(tuple(input_shape), layer_name)
        self.filter_shape = filter_shape
        self.use_peephole = use_peephole
        self.backward = backward
        self.learn_init = learn_init
        self.grad_step = grad_step
        self.grad_clip = grad_clip
        self.activation = utils.function[activation] if not callable(activation) else activation
        self.kwargs = kwargs

        n_in = self.input_shape[-1]
        self.in_gate = Gate(filter_shape, W_cell=False, layer_name='in_gate')
        self.forget_gate = Gate(filter_shape, W_cell=False, layer_name='forget_gate')
        self.cell_gate = Gate(filter_shape, W_cell=False, activation='tanh', layer_name='cell_gate')
        self.out_gate = Gate(filter_shape, W_cell=False, layer_name='out_gate')

        self.cell_init = theano.shared(np.zeros(self.output_shape[1:], theano.config.floatX), 'cell_init')
        self.hid_init = theano.shared(np.zeros(self.output_shape[1:], theano.config.floatX), 'hid_init')
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
                 grad_clip=0, activation='tanh', layer_name='AttConv LSTMCell', **kwargs):
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
        self.activation = utils.function[activation] if not callable(activation) else activation
        self.kwargs = kwargs

        self.in_gate = Gate(filter_shape, W_cell=False, layer_name='in_gate')
        self.forget_gate = Gate(filter_shape, W_cell=False, layer_name='forget_gate')
        self.cell_gate = Gate(filter_shape, W_cell=False, activation='tanh', layer_name='cell_gate')
        self.out_gate = Gate(filter_shape, W_cell=False, layer_name='out_gate')
        self.att_gate = Gate(filter_shape, W_cell=False, activation='tanh', layer_name='att_gate')
        # self.Va = theano.shared(np.zeros((1, filter_shape[0], filter_shape[2], filter_shape[3]), theano.config.floatX), 'att_kern')

        self.params += self.in_gate.params + self.forget_gate.params + self.cell_gate.params + \
                       self.out_gate.params + self.att_gate.params #+ [self.Va]
        self.trainable += self.in_gate.trainable + self.forget_gate.trainable + self.cell_gate.trainable + \
                          self.out_gate.trainable + self.att_gate.trainable #+ [self.Va]
        self.regularizable += self.in_gate.regularizable + self.forget_gate.regularizable + \
                              self.cell_gate.regularizable + self.out_gate.regularizable + self.att_gate.regularizable
        if self.learn_init:
            self.cell_init = theano.shared(
                np.zeros((1, filter_shape[0], input_shape[2], input_shape[3]), theano.config.floatX), 'cell_init')
            self.hid_init = theano.shared(
                np.zeros((1, filter_shape[0], input_shape[2], input_shape[3]), theano.config.floatX), 'hid_init')
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
            cell_init = T.zeros((num_batch, self.filter_shape[0], self.input_shape[2], self.input_shape[3]), theano.config.floatX)
            hid_init = T.zeros((num_batch, self.filter_shape[0], self.input_shape[2], self.input_shape[3]), theano.config.floatX)

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


@utils.deprecated(message='Please use \'set_training_on\' and \'set_training_off\' instead.')
def set_training_status(training):
    Layer.set_training_status(training)


def ConvNormAct(input_shape, num_filters, filter_size, init=HeNormal(gain=1.), no_bias=True, border_mode='half',
                stride=(1, 1), layer_name='Conv Norm Act', activation='relu', dilation=(1, 1), epsilon=1e-4,
                running_average_factor=1e-1, axes='spatial', no_scale=False, normalization='bn', groups=32, **kwargs):
    assert normalization in ('bn', 'gn'), 'normalization can be either \'bn\' or \'gn\', got %s' % normalization

    from neuralnet.normalization import BatchNormLayer, GroupNormLayer
    block = Sequential(input_shape=input_shape, layer_name=layer_name)
    block.append(ConvolutionalLayer(block.output_shape, num_filters, filter_size, init, no_bias, border_mode, stride,
                                    dilation, layer_name+'_conv', 'linear', **kwargs))
    if normalization == 'bn':
        block.append(BatchNormLayer(block.output_shape, layer_name+'/bn', epsilon, running_average_factor, axes,
                                    activation, no_scale, **kwargs))
    else:
        block.append(GroupNormLayer(block.output_shape, layer_name+'/gn', groups, epsilon, activation, **kwargs))
    return block


def StackingConv(input_shape, num_layers, num_filters, filter_size=3, batch_norm=False, layer_name='Stacking Conv',
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
                          layer_name='NetworkInNetwork', **kwargs):
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


def SoftmaxLayer(input_shape, num_nodes, layer_name='Softmax'):
    return FullyConnectedLayer(input_shape, num_nodes, layer_name=layer_name, activation='softmax')


def SigmoidLayer(input_shape, layer_name='Sigmoid'):
    return FullyConnectedLayer(input_shape, 1, layer_name=layer_name, activation='sigmoid')
