'''
Written and collected by Duc Nguyen
Last Modified by Duc Nguyen (theano version >= 0.8.1 required)
Updates on Sep 2017: added BatchNormDNNLayer from Lasagne


'''
__author__ = 'Duc Nguyen'

import math
import time
import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d as conv
from theano.tensor.signal.pool import pool_2d as pool
if theano.sandbox.cuda.cuda_enabled:
    from theano.sandbox.cuda import dnn
elif theano.gpuarray.dnn.dnn_present():
    from theano.gpuarray import dnn

from neuralnet import utils

rng = np.random.RandomState(int(time.time()))


class Layer(object):
    def __init__(self):
        self.rng = np.random.RandomState(int(time.time()))
        self.params = []
        self.trainable = []
        self.regularizable = []

    def __call__(self, *args, **kwargs):
        return self.get_output(*args, **kwargs)

    def get_output(self, input):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

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

    @staticmethod
    def reset():
        raise NotImplementedError


class PoolingLayer(Layer):
    def __init__(self, input_shape, ws=(2, 2), ignore_border=False, stride=(2, 2), pad=(0, 0), mode='max', layer_name='Pooling'):
        assert isinstance(input_shape, list) or isinstance(input_shape, tuple), \
            'input_shape must be list or tuple. Received %s' % type(input_shape)
        assert len(input_shape) == 4, 'input_shape must have 4 elements. Received %d' % len(input_shape)
        super(PoolingLayer, self).__init__()

        self.input_shape = tuple(input_shape)
        self.ws = ws
        self.ignore_border = ignore_border
        self.stride = stride
        self.pad = pad
        self.mode = mode
        self.layer_name = layer_name
        print('@ {} {}PoolingLayer: size: {}'.format(self.layer_name, self.mode, self.ws),
              ' stride: {}'.format(self.stride))

    def get_output(self, input):
        return pool(input, self.ws, self.ignore_border, self.stride, self.pad, self.mode)

    @property
    def output_shape(self):
        size = list(self.input_shape)
        size[2] = int(np.ceil(float(size[2] + 2 * self.pad[0] - self.ws[0]) / self.stride[0] + 1))
        size[3] = int(np.ceil(float(size[3] + 2 * self.pad[1] - self.ws[1]) / self.stride[1] + 1))
        return tuple(size)


class DropoutLayer(Layer):
    layers = []

    def __init__(self, input_shape, drop_prob=0.5, GaussianNoise=False, activation='relu', layer_name='Dropout'):
        assert isinstance(input_shape, list) or isinstance(input_shape, tuple), \
            'input_shape must be list or tuple. Received %s' % type(input_shape)
        assert len(input_shape) == 2 or len(input_shape) == 4, \
            'input_shape must have 2 or 4 elements. Received %d' % len(input_shape)
        super(DropoutLayer, self).__init__()

        self.input_shape = tuple(input_shape)
        self.GaussianNoise = GaussianNoise
        self.activation = utils.function[activation]
        self.layer_name = layer_name
        self.srng = utils.RandomStreams(rng.randint(1, int(time.time())))
        self.keep_prob = 1. - drop_prob
        self.training_flag = False
        print('@ {} DropoutLayer: p={:.2f} activation: {}'.format(self.layer_name, self.keep_prob, activation))
        DropoutLayer.layers.append(self)

    def get_output(self, input):
        mask = self.srng.normal(input.shape) + 1. if self.GaussianNoise else self.srng.binomial(n=1, p=self.keep_prob, size=input.shape)
        output_on = mask * input if self.GaussianNoise else input * T.cast(mask, theano.config.floatX)
        output_off = input if self.GaussianNoise else input * self.keep_prob
        return self.activation(output_on if self.training_flag else output_off)

    @property
    def output_shape(self):
        return tuple(self.input_shape)

    @staticmethod
    def set_training(training):
        for layer in DropoutLayer.layers:
            layer.training_flag = training


class FullyConnectedLayer(Layer):
    layers = []

    def __init__(self, input_shape, num_nodes, He_init=None, He_init_gain='relu', W=None, b=None, no_bias=False, layer_name='fc',
                 activation='relu', target='dev0'):
        '''

        :param n_in:
        :param num_nodes:
        :param He_init: 'normal', 'uniform' or None (default)
        :param He_init_gain:
        :param W:
        :param b:
        :param no_bias:
        :param layer_name:
        :param activation:
        :param target:
        '''
        assert isinstance(input_shape, list) or isinstance(input_shape, tuple), \
            'input_shape must be list or tuple. Received %s' % type(input_shape)
        assert len(input_shape) == 2 or len(input_shape) == 4, \
            'input_shape must have 2 or 4 elements. Received %d' % len(input_shape)
        super(FullyConnectedLayer, self).__init__()

        self.input_shape = tuple(input_shape) if len(input_shape) == 2 else (input_shape[0], np.prod(input_shape[1:]))
        self.num_nodes = num_nodes
        self.He_init = He_init
        self.He_init_gain = He_init_gain
        self.activation = utils.function[activation]
        self.no_bias = no_bias
        self.layer_name = layer_name
        self.target = target

        if W is None:
            if self.He_init:
                self.W_values = self.init_he((self.input_shape[1], num_nodes), self.He_init_gain, self.He_init)
            else:
                W_bound = np.sqrt(6. / (self.input_shape[1] + num_nodes)) * 4 if self.activation is utils.function['sigmoid'] \
                    else np.sqrt(6. / (self.input_shape[1] + num_nodes))
                self.W_values = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(self.input_shape[1], num_nodes)),
                                      dtype=theano.config.floatX) if W is None else W
        else:
            self.W_values = W
        self.W = theano.shared(value=self.W_values, name=self.layer_name + '_W', borrow=True)#, target=self.target)
        self.trainable.append(self.W)
        self.params.append(self.W)

        if not self.no_bias:
            if b is None:
                self.b_values = np.zeros((num_nodes,), dtype=theano.config.floatX) if b is None else b
            else:
                self.b_values = b
            self.b = theano.shared(value=self.b_values, name=self.layer_name + '_b', borrow=True) if not no_bias else None#, target=self.target)
            self.trainable.append(self.b)
            self.params.append(self.b)

        self.regularizable.append(self.W)
        print('@ {} FC: in_shape = {} out_shape = {} activation: {}'.format(self.layer_name, self.input_shape,
                                                                            self.output_shape, activation))
        FullyConnectedLayer.layers.append(self)

    def get_output(self, input):
        output = T.dot(input.flatten(2), self.W) + self.b if not self.no_bias else T.dot(input.flatten(2), self.W)
        return self.activation(output)

    @property
    def output_shape(self):
        return (self.input_shape[0], int(self.num_nodes/2)) if self.activation is 'maxout' \
            else (self.input_shape[0], self.num_nodes)

    @staticmethod
    def reset():
        for layer in FullyConnectedLayer.layers:
            layer.W.set_value(layer.W_values)
            layer.b.set_value(layer.b_values)


class ConvolutionalLayer(Layer):
    layers = []

    def __init__(self, input_shape, filter_shape, He_init=None, He_init_gain='relu', W=None, b=None, no_bias=True,
                 border_mode='half', subsample=(1, 1), layer_name='conv', activation='relu', pool=False,
                 pool_size=(2, 2), pool_mode='max', pool_stride=(2, 2), pool_pad=(0, 0), target='dev0'):
        '''

        :param input_shape: (B, C, H, W)
        :param filter_shape: (num filters, num input channels, H, W)
        :param He_init:
        :param He_init_gain:
        :param W:
        :param b:
        :param no_bias: True (default) or False
        :param border_mode:
        :param subsample:
        :param layer_name:
        :param activation:
        :param pool:
        :param pool_size:
        :param pool_mode:
        :param pool_stride:
        :param pool_pad:
        :param target:
        '''
        assert isinstance(input_shape, list) or isinstance(input_shape, tuple), \
            'input_shape must be list or tuple. Received %s' % type(input_shape)
        assert len(input_shape) == 2 or len(input_shape) == 4, \
            'input_shape must have 2 or 4 elements. Received %d' % len(input_shape)
        assert len(input_shape) == len(filter_shape) == 4, \
            'Filter shape and input shape must be 4-dimensional. Received %d and %d' % \
            (len(input_shape), len(filter_shape))
        super(ConvolutionalLayer, self).__init__()

        self.input_shape = tuple(input_shape)
        self.filter_shape = tuple(filter_shape)
        self.no_bias = no_bias
        self.activation = utils.function[activation]
        self.He_init = He_init
        self.He_init_gain = He_init_gain
        self.layer_name = layer_name
        self.border_mode = border_mode
        self.subsample = subsample
        self.pool = pool
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.pool_stride = pool_stride
        self.pool_pad = pool_pad
        self.target = target

        if W is None:
            if He_init:
                self.W_values = self.init_he(self.filter_shape, self.He_init_gain, He_init)
            else:
                fan_in = np.prod(filter_shape[1:])
                fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))
                W_bound = np.sqrt(6. / (fan_in + fan_out))
                self.W_values = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                           dtype=theano.config.floatX)
        else:
            self.W_values = W
        self.W = theano.shared(self.W_values, name=self.layer_name + '_W', borrow=True)#, target=self.target)
        self.trainable.append(self.W)
        self.params.append(self.W)

        if not self.no_bias:
            if b is None:
                self.b_values = np.zeros(filter_shape[0], dtype=theano.config.floatX)
            else:
                self.b_values = b
            self.b = theano.shared(self.b_values, self.layer_name + '_b', borrow=True)
            self.trainable.append(self.b)
            self.params.append(self.b)

        self.regularizable += [self.W]
        print('@ {} ConvLayer: '.format(self.layer_name), 'border mode: {} '.format(border_mode), end=' ')
        print('shape: {} , '.format(input_shape), end=' ')
        print('filter shape: {} '.format(filter_shape), end=' ')
        print('activation: {} '.format(activation), end=' ')
        print('{} pool {}'.format(self.pool_mode, self.pool))
        ConvolutionalLayer.layers.append(self)

    def get_output(self, input):
        output = conv(input=input, filters=self.W, border_mode=self.border_mode, subsample=self.subsample)
        if not self.no_bias:
            output += self.b.dimshuffle(('x', 0, 'x', 'x'))
        output = output if not self.pool else pool(input=output, ws=self.pool_size, ignore_border=False,
                                                   mode=self.pool_mode, pad=self.pool_pad, stride=self.pool_stride)
        return self.activation(T.clip(output, 1e-7, 1.0 - 1e-7)) if self.activation is utils.function['sigmoid'] \
            else self.activation(output)

    @property
    def output_shape(self):
        size = list(self.input_shape)
        assert len(size) == 4, "Shape must consist of 4 elements only"

        if self.border_mode == 'half':
            p = (self.filter_shape[2] // 2, self.filter_shape[3] // 2)
        elif self.border_mode == 'valid':
            p = (0, 0)
        elif self.border_mode == 'full':
            p = (self.filter_shape[2] - 1, self.filter_shape[3] - 1)
        else:
            raise NotImplementedError

        size[2] = (size[2] - self.filter_shape[2] + 2 * p[0]) // self.subsample[0] + 1
        size[3] = (size[3] - self.filter_shape[3] + 2 * p[1]) // self.subsample[1] + 1
        if self.pool:
            size[2] = int(np.ceil((size[2] + 2 * self.pool_pad[0] - self.pool_size[0]) / self.pool_stride[0] + 1))
            size[3] = int(np.ceil((size[3] + 2 * self.pool_pad[1] - self.pool_size[1]) / self.pool_stride[1] + 1))

        size[1] = self.filter_shape[0] // 2 if self.activation is 'maxout' else self.filter_shape[0]
        return tuple(size)

    @staticmethod
    def reset():
        for layer in ConvolutionalLayer.layers:
            layer.W.set_value(layer.W_values)
            layer.b.set_value(layer.b_values)


class TransposedConvolutionalLayer(Layer):
    layers = []

    def __init__(self, filter_shape, output_shape, layer_name='transconv', W=None, b=None, padding='valid', stride=(2, 2),
                 activation='relu', target='dev0'):
        '''

        :param filter_shape:(input channels, output channels, filter rows, filter columns)
        :param output_shape:
        :param layer_name:
        :param W:
        :param b:
        :param padding:
        :param stride:
        :param activation:
        :param target:
        '''
        super(TransposedConvolutionalLayer, self).__init__()

        self.filter_shape = tuple(filter_shape)
        self.input_shape = tuple(output_shape)
        self.padding = padding
        self.stride = stride
        self.activation = utils.function[activation]
        self.layer_name = layer_name
        self.target = target

        self.b_values = np.zeros((filter_shape[1],), dtype=theano.config.floatX) if b is None else b
        self.W_values = self.get_deconv_filter(filter_shape) if W is None else W
        self.W = theano.shared(self.W_values, self.layer_name + '_W', borrow=True)#, target=self.target)
        self.b = theano.shared(value=self.b_values, name=self.layer_name + '_b', borrow=True)#, target=self.target)
        self.params += [self.W, self.b]
        self.trainable += [self.W, self.b]
        self.regularizable.append(self.W)
        print('@ {} TransposedConv: padding: {} '.format(layer_name, padding), end=' ')
        print('shape: {} '.format(output_shape), end=' ')
        print('filter shape: {} '.format(filter_shape), end=' ')
        print('activation: {}'.format(activation))
        TransposedConvolutionalLayer.layers.append(self)

    def get_deconv_filter(self, f_shape):
        """
        This function is collected
        :param f_shape: self.filter_shape
        :return: an initializer for get_variable
        """
        width = f_shape[2]
        height = f_shape[3]
        f = math.ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[2], f_shape[3]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for j in range(f_shape[1]):
            for i in range(f_shape[0]):
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
        if None in self.input_shape:
            in_shape = output.shape
            h = ((in_shape[2] - 1) * self.stride[0]) + self.filter_shape[2] + \
                T.mod(in_shape[2]+2*p[0]-self.filter_shape[2], self.stride[0]) - 2*p[0]
            w = ((in_shape[3] - 1) * self.stride[1]) + self.filter_shape[3] + \
                T.mod(in_shape[3]+2*p[1]-self.filter_shape[3], self.stride[1]) - 2*p[1]
            self.input_shape = [self.input_shape[0], self.filter_shape[1], h, w]
        else:
            self.input_shape = [self.input_shape[0], self.input_shape[1], self.input_shape[2], self.filter_shape[2]]
        input = theano.tensor.nnet.conv2d_transpose(output, self.W, self.input_shape, self.filter_shape,
                                                    border_mode=self.padding, input_dilation=self.stride)
        input = input + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.activation(T.clip(input, 1e-7, 1.0 - 1e-7)) if self.activation is utils.function['sigmoid'] \
            else self.activation(input)

    @property
    def output_shape(self):
        return tuple(self.input_shape)

    @staticmethod
    def reset():
        for layer in TransposedConvolutionalLayer.layers:
            layer.W.set_value(layer.W_values)
            layer.b.set_value(layer.b_values)


class ResNetBlock(Layer):
    def __init__(self, input_shape, down_dim, up_dim, stride=(1, 1), activation='relu', layer_name='ResNetBlock',
                 branch1_conv=False, target='dev0'):
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
        super(ResNetBlock, self).__init__()

        self.input_shape = tuple(input_shape)
        self.down_dim = down_dim
        self.up_dim = up_dim
        self.stride = stride
        self.layer_name = layer_name
        self.activation = utils.function[activation]
        self.branch1_conv = branch1_conv
        self.target = target
        self.block = []

        if self.branch1_conv:
            self.conv1_branch1 = ConvolutionalLayer(self.input_shape, (self.up_dim, self.input_shape[1], 1, 1),
                                                    subsample=self.stride, target=self.target)

        self.conv1_branch2 = ConvolutionalLayer(self.input_shape, (self.down_dim, self.input_shape[1], 1, 1),
                                                subsample=self.stride, layer_name=layer_name + '_conv1_branch2',
                                                activation='linear', target=self.target)
        self.block.append(self.conv1_branch2)
        self.conv1_branch2_bn = BatchNormLayer(self.conv1_branch2.output_shape, layer_name + '_conv1_branch2_bn',
                                               activation=self.activation)
        self.block.append(self.conv1_branch2_bn)
        self.conv2_branch2 = ConvolutionalLayer(self.conv1_branch2_bn.output_shape, (self.down_dim, self.down_dim, 3, 3),
                                                subsample=(1, 1), layer_name=layer_name + '_conv2_branch2',
                                                activation='linear', target=self.target)
        self.block.append(self.conv2_branch2)
        self.conv2_branch2_bn = BatchNormLayer(self.conv2_branch2.output_shape, layer_name + '_conv2_branch2_bn',
                                               activation=self.activation)
        self.block.append(self.conv2_branch2_bn)
        self.conv3_branch2 = ConvolutionalLayer(self.conv2_branch2_bn.output_shape, (self.up_dim, self.down_dim, 1, 1),
                                                subsample=(1, 1), layer_name=layer_name + '_conv3_branch2',
                                                activation='linear', target=self.target)
        self.block.append(self.conv3_branch2)
        self.conv3_branch2_bn = BatchNormLayer(self.conv3_branch2.output_shape, layer_name + '_conv3_branch2_bn',
                                               activation='linear')
        self.block.append(self.conv3_branch2_bn)

    def get_output(self, input):
        output = self.conv1_branch1.get_output(input) + utils.inference(input, self.block) if self.branch1_conv \
            else input + utils.inference(input, self.block)
        return self.activation(output)

    @property
    def output_shape(self):
        output_shape = self.block[-1].output_shape
        return (self.input_shape[0], self.up_dim, output_shape[2], output_shape[3])


class DenseBlock(Layer):
    def __init__(self, input_shape, transit=False, num_conv_layer=6, growth_rate=32, dropout=False,
                 layer_name='DenseBlock', target='dev0'):
        '''

        :param input_shape: (int, int, int, int)
        :param num_conv_layer: int
        :param growth_rate: int
        :param layer_name: str
        :param target: str
        '''
        super(DenseBlock, self).__init__()

        self.input_shape = tuple(input_shape)
        self.transit = transit
        self.num_conv_layer = num_conv_layer
        self.growth_rate = growth_rate
        self.dropout = dropout
        self.layer_name = layer_name
        self.target = target

        if not self.transit:
            self.block = self._dense_block(self.input_shape, self.num_conv_layer, self.growth_rate, self.dropout, self.layer_name)
            pass
        else:
            self.block = self._transition(self.input_shape, self.dropout, self.layer_name + '_transition')

    def _bn_relu_conv(self, input_shape, filter_shape, dropout, layer_name='bn_re_conv'):
        block = [
            BatchNormLayer(input_shape, activation='relu', layer_name=layer_name + '_bn'),
            ConvolutionalLayer(input_shape, filter_shape, He_init='normal', He_init_gain='relu', activation='linear',
                               layer_name=layer_name + '_conv')
        ]
        for layer in block:
            self.params += layer.params
            self.trainable += layer.trainable
            self.regularizable += layer.regularizable
        if dropout:
            block.append(DropoutLayer(block[-1].output_shape, dropout, activation='linear',
                                      layer_name=layer_name + 'dropout'))
        return block

    def _transition(self, input_shape, dropout, layer_name='transition'):
        filter_shape = (input_shape[1], input_shape[1], 1, 1)
        block = self._bn_relu_conv(input_shape, filter_shape, dropout, layer_name)
        block.append(PoolingLayer(block[-1].output_shape, (2, 2), mode='average_inc_pad',
                                  layer_name=layer_name + 'pooling'))
        return block

    def _dense_block(self, input_shape, num_layers, growth_rate, dropout, layer_name='dense_block'):
        block, input_channels = [], input_shape[1]
        i_shape = list(input_shape)
        for n in range(num_layers):
            filter_shape = (growth_rate, input_channels, 3, 3)
            block.append(self._bn_relu_conv(i_shape, filter_shape, dropout, layer_name + '_%d' % n))
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
    def output_shape(self):
        if not self.transit:
            shape = (self.input_shape[0], self.input_shape[1] + self.growth_rate * self.num_conv_layer,
                     self.input_shape[2], self.input_shape[3])
        else:
            shape = self.block[-1].output_shape
        return tuple(shape)


class DenseBlockBRN(Layer):
    def __init__(self, input_shape, transit=False, num_conv_layer=6, growth_rate=32, dropout=False,
                 layer_name='DenseBlock', target='dev0'):
        '''

        :param input_shape: (int, int, int, int)
        :param num_conv_layer: int
        :param growth_rate: int
        :param layer_name: str
        :param target: str
        '''
        super(DenseBlockBRN, self).__init__()

        self.input_shape = tuple(input_shape)
        self.transit = transit
        self.num_conv_layer = num_conv_layer
        self.growth_rate = growth_rate
        self.dropout = dropout
        self.layer_name = layer_name
        self.target = target

        if not self.transit:
            self.block = self._dense_block(self.input_shape, self.num_conv_layer, self.growth_rate, self.dropout, self.layer_name)
            pass
        else:
            self.block = self._transition(self.input_shape, self.dropout, self.layer_name + '_transition')

    def _brn_relu_conv(self, input_shape, filter_shape, dropout, layer_name='bn_re_conv'):
        block = [
            BatchRenormLayer(input_shape, activation='relu', layer_name=layer_name + '_bn'),
            ConvolutionalLayer(input_shape, filter_shape, He_init='normal', He_init_gain='relu', activation='linear',
                               layer_name=layer_name + '_conv')
        ]
        for layer in block:
            self.params += layer.params
            self.trainable += layer.trainable
            self.regularizable += layer.regularizable
        if dropout:
            block.append(DropoutLayer(block[-1].output_shape, dropout, activation='linear',
                                      layer_name=layer_name + 'dropout'))
        return block

    def _transition(self, input_shape, dropout, layer_name='transition'):
        filter_shape = (input_shape[1], input_shape[1], 1, 1)
        block = self._brn_relu_conv(input_shape, filter_shape, dropout, layer_name)
        block.append(PoolingLayer(block[-1].output_shape, (2, 2), mode='average_inc_pad',
                                  layer_name=layer_name + 'pooling'))
        return block

    def _dense_block(self, input_shape, num_layers, growth_rate, dropout, layer_name='dense_block'):
        block, input_channels = [], input_shape[1]
        i_shape = list(input_shape)
        for n in range(num_layers):
            filter_shape = (growth_rate, input_channels, 3, 3)
            block.append(self._brn_relu_conv(i_shape, filter_shape, dropout, layer_name + '_%d' % n))
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
    def output_shape(self):
        if not self.transit:
            shape = (self.input_shape[0], self.input_shape[1] + self.growth_rate * self.num_conv_layer,
                     self.input_shape[2], self.input_shape[3])
        else:
            shape = self.block[-1].output_shape
        return tuple(shape)


class BatchNormLayer(Layer):
    layers = []

    def __init__(self, input_shape, layer_name='BN', epsilon=1e-4, running_average_factor=1e-3, axes='spatial',
                 activation='relu'):
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
        self.epsilon = epsilon
        self.running_average_factor = running_average_factor
        self.activation = utils.function[activation]
        self.training_flag = False
        self.axes = (0,) + tuple(range(2, len(input_shape))) if axes == 'spatial' else (0,)
        shape = (self.input_shape[1],) if axes == 'spatial' else self.input_shape[1:]
        self.gamma = theano.shared(np.ones(shape, dtype=theano.config.floatX), name=layer_name + '_gamma', borrow=True)
        self.beta = theano.shared(np.zeros(shape, dtype=theano.config.floatX), name=layer_name + '_beta', borrow=True)
        self.running_mean = theano.shared(np.zeros(shape, dtype=theano.config.floatX),
                                          name=layer_name + '_running_mean', borrow=True)
        self.running_var = theano.shared(np.zeros(shape, dtype=theano.config.floatX),
                                         name=layer_name + '_running_var', borrow=True)
        self.params += [self.gamma, self.beta, self.running_mean, self.running_var]
        self.trainable += [self.gamma, self.beta]
        self.regularizable.append(self.gamma)
        print('@ {} BatchNormLayer: running_average_factor = {:.4f}'.format(layer_name, self.running_average_factor))
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
                               else self.batch_normalization_test(input))

    @property
    def output_shape(self):
        return tuple(self.input_shape)

    @staticmethod
    def set_training(training):
        for layer in BatchNormLayer.layers:
            layer.training_flag = training

    @staticmethod
    def reset():
        for layer in BatchNormLayer.layers:
            layer.gamma.set_value(layer.gamma_values)
            layer.beta.set_value(layer.beta_values)


class BatchRenormLayer(Layer):
    layers = []

    def __init__(self, input_shape, layer_name='BN', epsilon=1e-4, r_max=1, d_max=0, running_average_factor=0.1,
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
        shape = (self.input_shape[1],) if axes == 'spatial' else self.input_shape[1:]
        self.gamma_values = np.ones(shape, dtype=theano.config.floatX)
        self.beta_values = np.zeros(shape, dtype=theano.config.floatX)
        self.gamma = theano.shared(self.gamma_values, name=layer_name + '_gamma', borrow=True)
        self.beta = theano.shared(self.beta_values, name=layer_name + '_beta', borrow=True)
        self.running_mean = theano.shared(np.zeros(shape, dtype=theano.config.floatX),
                                          name=layer_name + '_running_mean', borrow=True)
        self.running_var = theano.shared(np.zeros(shape, dtype=theano.config.floatX),
                                         name=layer_name + '_running_var', borrow=True)
        self.params += [self.gamma, self.beta, self.running_mean, self.running_var]
        self.trainable += [self.gamma, self.beta]
        self.regularizable.append(self.gamma)
        print('@ {} BatchRenormLayer: running_average_factor = {:.4f}' % (layer_name, self.running_average_factor))
        BatchNormLayer.layers.append(self)

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
    def output_shape(self):
        return tuple(self.input_shape)

    @staticmethod
    def set_training(training):
        for layer in BatchRenormLayer.layers:
            layer.training_flag = training

    @staticmethod
    def reset():
        for layer in BatchRenormLayer.layers:
            layer.gamma.set_value(layer.gamma_values)
            layer.beta.set_value(layer.beta_values)


class BatchNormDNNLayer(Layer):
    layers = []
    """
    lasagne.layers.BatchNormDNNLayer(incoming, axes='auto', epsilon=1e-4,
    alpha=0.1, beta=lasagne.init.Constant(0), gamma=lasagne.init.Constant(1),
    mean=lasagne.init.Constant(0), inv_std=lasagne.init.Constant(1), **kwargs)

    Batch Normalization

    This layer implements batch normalization of its inputs:

    .. math::
        y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta

    This is a drop-in replacement for :class:`lasagne.layers.BatchNormLayer`
    that uses cuDNN for improved performance and reduced memory usage.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    axes : 'auto', int or tuple of int
        The axis or axes to normalize over. If ``'auto'`` (the default),
        normalize over all axes except for the second: this will normalize over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers. Only supports ``'auto'``
        and the equivalent axes list, or ``0`` and ``(0,)`` to normalize over
        the minibatch dimension only.
    epsilon : scalar
        Small constant :math:`\\epsilon` added to the variance before taking
        the square root and dividing by it, to avoid numerical problems. Must
        not be smaller than ``1e-5``.
    alpha : scalar
        Coefficient for the exponential moving average of batch-wise means and
        standard deviations computed during training; the closer to one, the
        more it will depend on the last batches seen
    beta : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\beta`. Must match
        the incoming shape, skipping all axes in `axes`. Set to ``None`` to fix
        it to 0.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    gamma : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\gamma`. Must
        match the incoming shape, skipping all axes in `axes`. Set to ``None``
        to fix it to 1.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    mean : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`\\mu`. Must match
        the incoming shape, skipping all axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    inv_std : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`1 / \\sqrt{
        \\sigma^2 + \\epsilon}`. Must match the incoming shape, skipping all
        axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer should be inserted between a linear transformation (such as a
    :class:`DenseLayer`, or :class:`Conv2DLayer`) and its nonlinearity. The
    convenience function :func:`batch_norm_dnn` modifies an existing layer to
    insert cuDNN batch normalization in front of its nonlinearity.

    For further information, see :class:`lasagne.layers.BatchNormLayer`. This
    implementation is fully compatible, except for restrictions on the `axes`
    and `epsilon` arguments.

    See also
    --------
    batch_norm_dnn : Convenience function to apply batch normalization
    """
    def __init__(self, input_shape, axes='auto', epsilon=1e-4, alpha=0.1, beta=0., gamma=1., mean=0., inv_std=1.,
                 activation='relu', layer_name='BatchNormDNN'):
        '''

        :param input_shape:
        :param axes:
        :param epsilon:
        :param alpha:
        :param beta:
        :param gamma:
        :param mean:
        :param inv_std:
        '''
        super(BatchNormDNNLayer, self).__init__()

        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(input_shape)
        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)

        self.axes = axes
        self.epsilon = epsilon
        self.alpha = alpha
        self.activation = utils.function[activation]
        self.training_flag = False
        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(self.input_shape) if axis not in self.axes]
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        if beta is None:
            self.beta = None
        else:
            self.beta = theano.shared(beta * np.ones(shape, theano.config.floatX), borrow=True, name=layer_name + '_beta')

        if gamma is None:
            self.gamma = None
        else:
            self.gamma = theano.shared(gamma * np.ones(shape, theano.config.floatX), borrow=True, name=layer_name + '_gamma')

        self.mean = theano.shared(mean * np.ones(shape, theano.config.floatX), borrow=True, name=layer_name + '_mean')
        self.inv_std = theano.shared(inv_std * np.ones(shape, theano.config.floatX), borrow=True, name=layer_name + '_inv_std')
        self.params += [self.gamma, self.beta, self.mean, self.inv_std]
        self.trainable += [self.beta, self.gamma]
        self.regularizable += [self.gamma]
        all_but_second_axis = (0,) + tuple(range(2, len(self.input_shape)))
        if self.axes not in ((0,), all_but_second_axis):
            raise ValueError("BatchNormDNNLayer only supports normalization "
                             "across the first axis, or across all but the "
                             "second axis, got axes=%r" % (axes,))
        BatchNormDNNLayer.layers.append(self)
        print('@ {} BatchNormDNNLayer: running_average_factor = {:.4f}' % (layer_name, self.alpha))

    def get_output(self, input, batch_norm_use_averages=None, batch_norm_update_averages=None):
        # Decide whether to use the stored averages or mini-batch statistics
        if batch_norm_use_averages is None:
            batch_norm_use_averages = self.training_flag
        use_averages = batch_norm_use_averages

        # Decide whether to update the stored averages
        if batch_norm_update_averages is None:
            batch_norm_update_averages = not self.training_flag
        update_averages = batch_norm_update_averages

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(list(range(input.ndim - len(self.axes))))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]
        # and prepare the converse pattern removing those broadcastable axes
        unpattern = [d for d in range(input.ndim) if d not in self.axes]

        # call cuDNN if needed, obtaining normalized outputs and statistics
        if not use_averages or update_averages:
            # cuDNN requires beta/gamma tensors; create them if needed
            shape = tuple(s for (d, s) in enumerate(input.shape)
                          if d not in self.axes)
            gamma = self.gamma or theano.tensor.ones(shape)
            beta = self.beta or theano.tensor.zeros(shape)
            mode = 'per-activation' if self.axes == (0,) else 'spatial'
            (normalized, input_mean, input_inv_std) = dnn.dnn_batch_normalization_train(
                input, gamma.dimshuffle(pattern), beta.dimshuffle(pattern), mode, self.epsilon)

        # normalize with stored averages, if needed
        if use_averages:
            mean = self.mean.dimshuffle(pattern)
            inv_std = self.inv_std.dimshuffle(pattern)
            gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
            beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
            normalized = (input - mean) * (gamma * inv_std) + beta

        # update stored averages, if needed
        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean.dimshuffle(unpattern))
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std.dimshuffle(unpattern))
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            dummy = 0 * (running_mean + running_inv_std).dimshuffle(pattern)
            normalized = normalized + dummy

        return self.activation(normalized)

    @staticmethod
    def set_training(training):
        for layer in BatchNormDNNLayer.layers:
            layer.training_flag = training


class ScaleLayer(Layer):
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

    >>> layer = ScaleLayer((20, 30, 40, 50), shared_axes=(0, 2))
    >>> layer.scales.get_value().shape
    (30, 50)
    """
    def __init__(self, input_shape, scales=1, shared_axes='auto', layer_name='ScaleLayer'):
        super(ScaleLayer, self).__init__()

        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(input_shape)
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

    def get_output(self, input):
        axes = iter(list(range(self.scales.ndim)))
        pattern = ['x' if input_axis in self.shared_axes
                   else next(axes) for input_axis in range(input.ndim)]
        return input * self.scales.dimshuffle(*pattern)


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


def reset_training():
    FullyConnectedLayer.reset()
    ConvolutionalLayer.reset()
    TransposedConvolutionalLayer.reset()
    BatchNormLayer.reset()
    BatchRenormLayer.reset()


def set_training_status(training):
    DropoutLayer.set_training(training)
    BatchNormLayer.set_training(training)
    BatchNormDNNLayer.set_training(training)
    BatchRenormLayer.set_training(training)
