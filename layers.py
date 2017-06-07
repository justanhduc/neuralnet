'''
Written and collected by Duc Nguyen
Last Modified by Duc Nguyen (theano version >= 0.8.1 required)
Last modification (May. 2017)


'''

import theano
from theano.tensor.signal.pool import pool_2d as pool
from theano.tensor.nnet import conv2d as conv
from theano.tensor.nnet import relu
import math
import time
import warnings

from utils import function
rng = np.random.RandomState(int(time.time()))


class DropoutLayer(object):
    layers = []

    def __init__(self, p=0.5, layer_name=None):
        self.layer_name = 'Dropout' if layer_name is None else layer_name
        self.srng = RandomStreams(rng.randint(1, int(time.time())))
        self.p = p
        self.dropout_on = theano.shared(np.cast[theano.config.floatX](1.0), borrow=True)
        print('  # %s (DO): p = %f' % (self.layer_name, p))
        DropoutLayer.layers.append(self)

    def get_output(self, input):
        mask = self.srng.binomial(n=1, p=self.p, size=input.shape)
        output_on = input * T.cast(mask, theano.config.floatX)
        output_off = input * self.p
        return self.dropout_on * output_on + (1.0 - self.dropout_on) * output_off

    @staticmethod
    def turn_dropout_on(training):
        for layer in DropoutLayer.layers:
            layer.dropout_on.set_value(float(training))


class DropoutGaussianLayer(object):
    layers = []

    def __init__(self, layer_name=None):
        self.layer_name = 'Dropout_Gaussian' if layer_name is None else layer_name
        self.srng = RandomStreams(rng.randint(1, int(time.time())))
        self.dropout_on = theano.shared(np.cast[theano.config.floatX](1.0), borrow=True)
        print('  # %s (DO-Gauss)' % self.layer_name)
        DropoutGaussianLayer.layers.append(self)

    def get_output(self, input):
        mask = self.srng.normal(input.shape) + 1.
        output = mask * input
        return self.dropout_on * output + (1.0 - self.dropout_on) * input

    @staticmethod
    def turn_dropout_on(training):
        for layer in DropoutGaussianLayer.layers:
            layer.dropout_on.set_value(float(training))


class FullyConnectedLayer(object):
    layers = []

    def __init__(self, n_in, n_out, W=None, b=None, layer_name=None, activation=relu, batch_norm=False, drop_out=False,
                 p=0.5, dropout_gauss=False):
        """
        :type n_in: int
        :param n_in: dimensionality of input
        :type n_out: int
        :param n_out: number of hidden units
        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer
        """
        assert isinstance(n_in, int), 'n_in must be an integer.'
        assert isinstance(n_out, int), 'n_in must be an integer.'
        if sum([dropout_gauss, drop_out, batch_norm]) > 1:
            warnings.warn('Dropout, dropout Gaussian and batch normalization are used together', UserWarning)

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.batch_norm = batch_norm
        self.layer_name = 'FC' if layer_name is None else layer_name
        self.drop_out = drop_out
        self.dropout_gauss = dropout_gauss
        FullyConnectedLayer.layers.append(self)

        b_values = np.zeros((n_out,), dtype=theano.config.floatX) if b is None else b
        W_bound = np.sqrt(6. / (n_in + n_out)) * 4 if self.activation is function['sigmoid'] \
            else np.sqrt(6. / (n_in + n_out))
        W_values = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(n_in, n_out)),
                              dtype=theano.config.floatX) if W is None else W
        self.W = theano.shared(value=W_values, name=self.layer_name + '_W', borrow=True)
        self.b = theano.shared(value=b_values, name=self.layer_name + '_b', borrow=True)
        self.params = [self.W, self.b]

        if self.batch_norm:
            self.bn_layer = BatchNormLayer((n_out,), 'bn_'+self.layer_name)
            self.params.pop(-1)
            self.params += self.bn_layer.params
        if self.drop_out:
            self.p = p
            self.do_layer = DropoutLayer(p=self.p, layer_name='%s_dropout' % self.layer_name)
        if self.dropout_gauss:
            self.do_gauss_layer = DropoutGaussianLayer(layer_name='%s_dropout_gaussian' % self.layer_name)

        print('  # %s (FC): in = %d -> out = %d' % (self.layer_name, n_in, n_out)),
        print('/ BN: %s /DO: %s /DO-GAUSSIAN: %s' % (self.batch_norm, self.drop_out, self.dropout_gauss))

    def get_output(self, input):
        output = T.dot(input, self.W) + self.b
        if self.batch_norm:
            output = self.bn_layer.get_output(output)
        if self.drop_out:
            output = self.do_layer.get_output(output)
        if self.dropout_gauss:
            output = self.do_gauss_layer.get_output(output)
        return self.activation(output)

    def get_output_shape(self):
        return int(self.n_out/2) if self.activation is maxout else self.n_out

    def reset(self):
        b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)
        W_bound = np.sqrt(6. / (self.n_in + self.n_out)) * 4 if self.activation is function['sigmoid'] \
            else np.sqrt(6. / (self.n_in + self.n_out))
        W_values = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(self.n_in, self.n_out)), dtype=theano.config.floatX)
        self.W.set_value(W_values)
        self.b.set_value(b_values)


class ConvolutionalLayer(object):
    layers = []

    def __init__(self, input_shape, filter_shape, W=None, border_mode='half', subsample=(1, 1), layer_name=None,
                 activation=function['relu'], pool=False, pool_size=(2, 2), pool_mode='max', pool_stride=(2, 2), pool_pad=(0, 0),
                 batch_norm=False, drop_out=False, p=0.5, dropout_gauss=False):
        """
        filter_shape: (number of filters, number of previous channels, filter height, filter width)
        image_shape: (batch size, num channels, image height, image width)
        """
        assert len(input_shape) == len(filter_shape) == 4, 'Filter shape and input shape must have 4 elements.'
        if sum([dropout_gauss, drop_out, batch_norm]) > 1:
            warnings.warn('Dropout, dropout Gaussian and batch normalization are used together', UserWarning)

        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.activation = activation
        self.batch_norm = batch_norm
        self.drop_out = drop_out
        self.layer_name = 'CONV' if layer_name is None else layer_name
        self.border_mode = border_mode
        self.subsample = subsample
        self.pool = pool
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.pool_stride = pool_stride
        self.pool_pad = pool_pad
        self.dropout_gauss = dropout_gauss

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        W_values = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                              dtype=theano.config.floatX) if W is None else W
        self.W = theano.shared(W_values, name=self.layer_name + '_W', borrow=True)
        self.params = [self.W]

        if self.batch_norm:
            self.bn_layer = BatchNormLayer(filter_shape, 'BN_'+self.layer_name)
            self.params += self.bn_layer.params

        if self.drop_out:
            self.p = p
            self.do_layer = DropoutLayer(p=self.p, layer_name='%s_dropout' % self.layer_name)

        if self.dropout_gauss:
            self.do_gauss_layer = DropoutGaussianLayer(layer_name='%s_dropout_gaussian' % self.layer_name)

        print('  # %s (Conv-%s):' % (layer_name, border_mode)),
        print('flt.(%s),' % ', '.join([str(i) for i in self.filter_shape])),
        print('/BN: %s /DO: %s /DO-GAUSSIAN: %s' % (self.batch_norm, self.drop_out, self.dropout_gauss))
        print('  # %s pool %s %s' % (self.layer_name, self.pool, self.pool_mode))
        ConvolutionalLayer.layers.append(self)

    def get_output(self, input):
        output = conv(input=input, filters=self.W, border_mode=self.border_mode, subsample=self.subsample)
        output = output if not self.pool else pool(input=output, ws=self.pool_size, ignore_border=False,
                                                   mode=self.pool_mode, pad=self.pool_pad)
        if self.batch_norm:
            output = self.bn_layer.get_output(output)
        if self.drop_out:
            output = self.do_layer.get_output(output)
        if self.dropout_gauss:
            output = self.do_gauss_layer.get_output(output)
        return self.activation(T.clip(output, 1e-7, 1.0 - 1e-7)) if self.activation is function['sigmoid'] \
            else self.activation(output)

    def get_output_shape(self, flatten=False):
        size = list(self.input_shape)
        assert len(size) == 4, "Shape must consist of 4 elements only"

        if self.border_mode == 'valid':
            size[2] = size[2] - self.filter_shape[2] + 1
            size[3] = size[3] - self.filter_shape[3] + 1
        elif self.border_mode == 'full':
            size[2] = size[2] + self.filter_shape[2] - 1
            size[3] = size[3] + self.filter_shape[3] - 1

        if self.pool:
            size[2] = int(np.ceil(float(size[2] - self.pool_size[0]) / self.pool_stride[0] + 1))
            size[3] = int(np.ceil(float(size[3] - self.pool_size[1]) / self.pool_stride[1] + 1))

        size[1] = int(self.filter_shape[0]/2) if self.activation is function['maxout'] else self.filter_shape[0]
        return (size[0], size[1] * size[2] * size[3]) if flatten else size

    def reset(self):
        fan_in = np.prod(self.filter_shape[1:])
        fan_out = (self.filter_shape[0] * np.prod(self.filter_shape[2:]))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        W_values = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=self.filter_shape), dtype=theano.config.floatX)
        self.W.set_value(W_values)


class ConvolutionalTransposedLayer(object):
    layers = []

    def __init__(self, filter_shape, output_shape, layer_name='ConvTransposed', W=None, b=None, padding='valid', stride=(2, 2),
                 activation=function['relu']):
        self.filter_shape = filter_shape
        self.output_shape = output_shape
        self.padding = padding
        self.stride = stride
        self.activation = activation
        self.layer_name = layer_name

        b_values = np.zeros((filter_shape[1],), dtype=theano.config.floatX) if b is None else b
        W_values = self.get_deconv_filter(filter_shape) if W is None else W
        self.W = theano.shared(W_values, self.layer_name + 'W', borrow=True)
        self.b = theano.shared(value=b_values, name=self.layer_name + '_b', borrow=True)
        self.params = [self.W, self.b]
        print('  # %s (ConvTransposed-%s):' % (layer_name, padding)),
        print('flt.(%s),' % ', '.join([str(i) for i in self.filter_shape]))

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
        for x in xrange(width):
            for y in xrange(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for j in xrange(f_shape[1]):
            for i in xrange(f_shape[0]):
                weights[i, j, :, :] = bilinear
        return weights.astype(theano.config.floatX)

    def get_output(self, output):
        if self.padding == 'half':
            p = (self.filter_shape[2] / 2, self.filter_shape[3] / 2)
        elif self.padding == 'valid':
            p = (0, 0)
        elif self.padding == 'full':
            p = (self.filter_shape[2] - 1, self.filter_shape[3] - 1)
        else:
            raise NotImplementedError
        if None in self.output_shape:
            in_shape = output.shape
            h = ((in_shape[2] - 1) * self.stride[0]) + self.filter_shape[2] + \
                T.mod(in_shape[2]+2*p[0]-self.filter_shape[2], self.stride[0]) - 2*p[0]
            w = ((in_shape[3] - 1) * self.stride[1]) + self.filter_shape[3] + \
                T.mod(in_shape[3]+2*p[1]-self.filter_shape[3], self.stride[1]) - 2*p[1]
            self.input_shape = [self.output_shape[0], self.filter_shape[1], h, w]
        else:
            self.input_shape = [self.output_shape[0], self.output_shape[1], self.output_shape[2], self.filter_shape[2]]
        input = theano.tensor.nnet.conv2d_transpose(output, self.W, self.input_shape, self.filter_shape,
                                                    border_mode=self.padding, input_dilation=self.stride)
        input = input + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.activation(T.clip(input, 1e-7, 1.0 - 1e-7)) if self.activation is function['sigmoid'] \
            else self.activation(input)

    def get_output_shape(self, flatten=False):
        return list(self.input_shape)


class BatchNormLayer(object):
    layers = []

    def __init__(self, input_shape, layer_name=None, epsilon=1e-4, running_average_factor=0.1):
        self.layer_name = 'BN' if layer_name is None else layer_name
        self.input_shape = input_shape
        self.epsilon = epsilon
        self.running_average_factor = running_average_factor
        self.training = theano.shared(np.cast['int8'](1), borrow=True)
        self.gamma = theano.shared(np.ones(input_shape[0], dtype=theano.config.floatX), name=layer_name + '_GAMMA',
                                   borrow=True)
        self.beta = theano.shared(np.zeros(input_shape[0], dtype=theano.config.floatX), name=layer_name + '_BETA',
                                  borrow=True)
        self.running_mean = T.zeros_like(self.gamma, dtype=theano.config.floatX)
        self.running_var = T.ones_like(self.gamma, dtype=theano.config.floatX)
        self.params = [self.gamma, self.beta]
        BatchNormLayer.layers.append(self)

    def batch_normalization_train(self, input):
        axes = (0,) + tuple(range(2, input.ndim))
        mean = input.mean(axes, keepdims=True)
        var = input.var(axes, keepdims=True)
        invstd = T.inv(T.sqrt(var + self.epsilon))
        gamma = self.gamma.dimshuffle('x', 0, 'x', 'x') if len(self.input_shape) == 4 else self.gamma
        beta = self.beta.dimshuffle('x', 0, 'x', 'x') if len(self.input_shape) == 4 else self.beta
        out = (input - mean) * gamma * invstd + beta
        m = T.cast(T.prod(input.shape) / T.prod(mean.shape), 'float32')
        self.running_mean = self.running_mean * (1 - self.running_average_factor) + mean * self.running_average_factor
        self.running_var = self.running_var * (1 - self.running_average_factor) + (m / (m - 1)) * var * self.running_average_factor
        return out

    def batch_normalization_test(self, input):
        gamma = self.gamma.dimshuffle('x', 0, 'x', 'x') if len(self.input_shape) == 4 else self.gamma
        beta = self.beta.dimshuffle('x', 0, 'x', 'x') if len(self.input_shape) == 4 else self.beta
        out = (input - self.running_mean) * gamma / T.sqrt(self.running_var + self.epsilon) + beta
        return out

    def get_output(self, input):
        return T.switch(T.eq(self.training, 1), self.batch_normalization_train(input), self.batch_normalization_test(input))

    @staticmethod
    def set_training(training):
        for layer in BatchNormLayer.layers:
            layer.training.set_value(training)


def set_training_status(training):
    DropoutGaussianLayer.turn_dropout_on(training)
    DropoutLayer.turn_dropout_on(training)
    BatchNormLayer.set_training(training)
