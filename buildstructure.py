"""
Written by Duc
April, 2016
Updated Feb 3rd, 2017
"""

from utils import *
import layers


class Structure(object):
    def __init__(self, config_file, **kwargs):
        super(Structure, self).__init__()
        self.config = load_configuration(config_file)
        self.layers = []
        self.index = 0
        self.output = None
        try:
            self.structure = self.config['architecture']['structure']
            self.sharing = self.config['architecture']['sharing']
            self.existing_model = self.config['architecture']['existing_model']
            self.param_file = self.config['architecture']['param_file']
            self.model_scope = self.config['architecture']['model_scope']
            self.num_layers = len(self.structure)
            self.batch_size = self.config['training']['batch_size']
            self.input_shape = self.config['layers']['input_size']
            self.input_size = [self.batch_size] + self.input_shape
            self.nkerns = self.config['layers']["nkerns"]
            self.kern_size = self.config['layers']['kern_size']
            self.activation = self.config['layers']['activation'] \
                if len(self.config['layers']['activation']) == self.num_layers \
                else [self.config['layers']['activation'][0]] * (self.num_layers - 1) \
                     + [self.config['layers']['activation'][-1]]
            self.conv_padding = self.config['layers']['conv_padding'] \
                if len(self.config['layers']['conv_padding']) == self.num_layers \
                else [self.config['layers']['conv_padding'][0]] * self.num_layers
            self.conv_stride = self.config['layers']['conv_stride']
            self.deconv_stride = self.config['layers']['deconv_stride']
            self.pooling = self.config['layers']['pooling']
            self.pooling_method = self.config['layers']['pooling_method'] \
                if len(self.config['layers']['pooling_method']) == self.num_layers \
                else [self.config['layers']['pooling_method'][0]] * self.num_layers
            self.pool_size = self.config['layers']['pool_size']
            self.pool_stride = self.config['layers']['pool_stride']
            self.pool_type = self.config['layers']['pool_type'] \
                if len(self.config['layers']['pool_type']) == self.num_layers \
                else [self.config['layers']['pool_type'][0]] * self.num_layers
            self.batch_norm = self.config['layers']['batch_norm'] \
                if len(self.config['layers']['batch_norm']) == self.num_layers \
                else [self.config['layers']['batch_norm'][0]] * self.num_layers
            self.dropout = self.config['layers']['dropout'] \
                if len(self.config['layers']['dropout']) == self.num_layers \
                else [self.config['layers']['dropout'][0]] * self.num_layers
            self.keep_prob = self.config['layers']['keep_prob']
            self.dropout_gauss = self.config['layers']['dropout_gauss'] \
                if len(self.config['layers']['dropout_gauss']) == self.num_layers \
                else [self.config['layers']['dropout_gauss'][0]] * self.num_layers
        except ValueError:
            raise ValueError('Some config value is invalid')

    def build_convolutional_layer(self, idx, feed_shape, prev_layer_shape, params=(None, None), layer_name=None):
        filter_shape = (self.nkerns[idx], prev_layer_shape, self.kern_size[idx], self.kern_size[idx])
        if idx != self.num_layers - 1:
            flatten = True if self.structure[idx + 1].lower() == 'fc' else False
        else:
            flatten = False
        conv_layer = layers.ConvolutionalLayer(input_shape=list(feed_shape), filter_shape=filter_shape, W=params[0],
                                               border_mode=self.conv_padding[idx], dropout_gauss=self.dropout_gauss[idx],
                                               subsample=self.conv_stride, batch_norm=self.batch_norm[idx],
                                               layer_name=layer_name, drop_out=self.dropout[idx], p=self.keep_prob,
                                               pool=self.pooling[idx], pool_size=self.pool_size, pool_stride=self.pool_stride,
                                               pool_mode=self.pooling_method[idx], pool_pad=self.pool_type[idx],
                                               activation=function[self.activation[idx].lower()])
        feed_shape = conv_layer.get_output_shape(flatten)
        prev_channel = self.nkerns[idx]
        return conv_layer, feed_shape, prev_channel

    def build_fully_connected_layer(self, idx, feed_shape, prev_layer_shape, params=(None, None), layer_name=None):
        fc_layer = layers.FullyConnectedLayer(n_in=feed_shape[1], n_out=self.nkerns[idx], W=params[0], b=params[1],
                                              batch_norm=self.batch_norm[idx], layer_name=layer_name, drop_out=self.dropout[idx],
                                              activation=function[self.activation[idx].lower()], dropout_gauss=self.dropout_gauss[idx])
        feed_shape = (feed_shape[0], fc_layer.get_output_shape())
        num_prev_layer_nodes = self.nkerns[idx]
        return fc_layer, feed_shape, num_prev_layer_nodes

    def build_transposed_convolutional_layer(self, idx, feed_shape, prev_layer_shape, params=(None, None), layer_name=None):
        transconv_layer = layers.TransposedConvolutionalLayer((prev_layer_shape, self.nkerns[idx], self.kern_size[idx],
                                                               self.kern_size[idx]), (feed_shape[0], None, None, None),
                                                              activation=self.activation[idx], layer_name=layer_name,
                                                              W=params[0], b=params[1])
        if idx != self.num_layers - 1:
            flatten = True if self.structure[idx + 1].lower() == 'fc' else False
        else:
            flatten = False
        feed_shape = transconv_layer.get_output_shape(flatten)
        num_prev_layer_nodes = self.nkerns[idx]
        return transconv_layer, feed_shape, num_prev_layer_nodes

    def build_model(self, **kwargs):
        if len(self.layers) > 0: self.layers = []
        feed_shape = kwargs.get('input_size', list(self.input_size))
        prev_channel = kwargs.get('num_channels', self.input_shape[0])
        model_scope = kwargs.get('model_scope', self.model_scope)
        params = kwargs.get('parameters', [(None, None)] * self.num_layers)
        layer = {'conv': lambda idx, feed_shape, prev_layer_shape, params, layer_name=None:
                 self.build_convolutional_layer(idx, feed_shape, prev_layer_shape, params, layer_name),
                 'fc': lambda idx, feed_shape, prev_layer_shape, params, layer_name=None:
                 self.build_fully_connected_layer(idx, feed_shape, prev_layer_shape, params, layer_name),
                 'transconv': lambda idx, feed_shape, prev_layer_shape, params, layer_name=None:
                 self.build_transposed_convolutional_layer(idx, feed_shape,prev_layer_shape, params, layer_name=None)}
        for i in xrange(self.num_layers):
            layer_name = '%s_%s%d' % (model_scope, self.structure[i].upper(), i)
            l, feed_shape, prev_channel = layer[self.structure[i].lower()](i, feed_shape, prev_channel,
                                                                                 params[i], layer_name)
            self.layers.append(l)

    def inference(self, input, **kwargs):
        model = kwargs.get('model', self.layers)
        self.output = T.reshape(inference(input, model), (-1, )) if self.nkerns[-1] == 1 else inference(input, model)
        return self.output
