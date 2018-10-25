import numpy as np

import neuralnet as nn


class Net:
    def save_params(self, param_file=None):
        param_file = param_file if param_file else self.param_file
        np.savez(param_file, **{p.name: p.get_value() for p in self.params})
        print('Model weights dumped to %s' % param_file)

    def load_params(self, param_file=None):
        param_file = param_file if param_file else self.param_file
        weights = np.load(param_file)
        for p in self.params:
            try:
                p.set_value(weights[p.name])
            except KeyError:
                KeyError('There is no saved weight for %s' % p.name)
        print('Model weights loaded from %s' % param_file)


class ResNet(nn.Sequential, Net):
    def __init__(self, input_shape, block, layers, num_filters, activation='relu', fc=True, pooling=True, num_classes=1000,
                 layer_name='ResNet', **kwargs):
        super(ResNet, self).__init__(input_shape=input_shape, layer_name=layer_name)
        self.activation = activation
        self.custom_block = kwargs.pop('custom_block', None)
        self.kwargs = kwargs
        self.append(nn.ConvNormAct(self.input_shape, num_filters, 7, stride=2, activation=activation, **kwargs))
        if pooling:
            self.append(nn.PoolingLayer(self.output_shape, (3, 3), stride=(2, 2), pad=1))
        self.append(self._make_layer(block, self.output_shape, num_filters, layers[0], name='block1'))
        self.append(self._make_layer(block, self.output_shape, 2 * num_filters, layers[1], stride=2, name='block2'))
        self.append(self._make_layer(block, self.output_shape, 4 * num_filters, layers[2], stride=2, name='block3'))
        self.append(self._make_layer(block, self.output_shape, 8 * num_filters, layers[3], stride=2, name='block4'))

        if fc:
            self.append(nn.GlobalAveragePoolingLayer(self.output_shape, layer_name='glb_avg_pooling'))
            self.append(nn.FullyConnectedLayer(self.output_shape, num_classes, activation='softmax',
                                                       layer_name='output'))

    def _make_layer(self, block, shape, planes, blocks, stride=1, name=''):
        downsample = None
        if stride != 1 or shape[1] != planes * block.upscale_factor:
            downsample = True

        layers = [block(shape, planes, stride, downsample=downsample, activation=self.activation,
                        layer_name=name + '_0', block=self.custom_block, **self.kwargs)]

        for i in range(1, blocks):
            layers.append(block(layers[-1].output_shape, planes, activation=self.activation, layer_name=name + '_%d' % i,
                                block=self.custom_block, **self.kwargs))
        return nn.Sequential(layers, layer_name=name)


class VGG16(nn.Sequential, Net):
    def __init__(self, input_shape, fc=True, num_classes=1000, name='vgg16'):
        super(VGG16, self).__init__(input_shape=input_shape, layer_name=name)
        self.fc = fc
        self.append(nn.Conv2DLayer(self.output_shape, 64, 3, no_bias=False, layer_name=name + '_conv1'))
        self.append(nn.Conv2DLayer(self.output_shape, 64, 3, no_bias=False, layer_name=name + '_conv2'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '_maxpool0'))

        self.append(nn.Conv2DLayer(self.output_shape, 128, 3, no_bias=False, layer_name=name + '_conv3'))
        self.append(nn.Conv2DLayer(self.output_shape, 128, 3, no_bias=False, layer_name=name + '_conv4'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '_maxpool1'))

        self.append(nn.Conv2DLayer(self.output_shape, 256, 3, no_bias=False, layer_name=name + '_conv5'))
        self.append(nn.Conv2DLayer(self.output_shape, 256, 3, no_bias=False, layer_name=name + '_conv6'))
        self.append(nn.Conv2DLayer(self.output_shape, 256, 3, no_bias=False, layer_name=name + '_conv7'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '_maxpool2'))

        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv8'))
        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv9'))
        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv10'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '_maxpool3'))

        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv11'))
        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv12'))
        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv13'))

        if fc:
            self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name+'_maxpool4'))
            self.append(nn.FCLayer(self.output_shape, 4096, layer_name=name+'_fc1'))
            self.append(nn.FCLayer(self.output_shape, 4096, layer_name=name+'_fc2'))
            self.append(nn.SoftmaxLayer(self.output_shape, num_classes, name+'_softmax'))


class VGG19(nn.Sequential, Net):
    def __init__(self, input_shape, fc=True, num_classes=1000, name='vgg19'):
        super(VGG19, self).__init__(input_shape=input_shape, layer_name=name)
        self.fc = fc
        self.append(nn.Conv2DLayer(self.output_shape, 64, 3, no_bias=False, layer_name=name + '_conv1_1'))
        self.append(nn.Conv2DLayer(self.output_shape, 64, 3, no_bias=False, layer_name=name + '_conv1_2'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '_maxpool0'))

        self.append(nn.Conv2DLayer(self.output_shape, 128, 3, no_bias=False, layer_name=name + '_conv2_1'))
        self.append(nn.Conv2DLayer(self.output_shape, 128, 3, no_bias=False, layer_name=name + '_conv2_2'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '_maxpool1'))

        self.append(nn.Conv2DLayer(self.output_shape, 256, 3, no_bias=False, layer_name=name + '_conv3_1'))
        self.append(nn.Conv2DLayer(self.output_shape, 256, 3, no_bias=False, layer_name=name + '_conv3_2'))
        self.append(nn.Conv2DLayer(self.output_shape, 256, 3, no_bias=False, layer_name=name + '_conv3_3'))
        self.append(nn.Conv2DLayer(self.output_shape, 256, 3, no_bias=False, layer_name=name + '_conv3_4'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '_maxpool2'))

        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv4_1'))
        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv4_2'))
        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv4_3'))
        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv4_4'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '_maxpool3'))

        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv5_1'))
        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv5_2'))
        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv5_3'))
        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv5_4'))

        if fc:
            self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name+'_maxpool4'))
            self.append(nn.FCLayer(self.output_shape, 4096, layer_name=name+'_fc1'))
            self.append(nn.FCLayer(self.output_shape, 4096, layer_name=name+'_fc2'))
            self.append(nn.SoftmaxLayer(self.output_shape, num_classes, name+'_softmax'))


class DenseNet(nn.Sequential, Net):
    def __init__(self, input_shape, fc=True, num_classes=1000, first_output=16, growth_rate=12, num_blocks=3, depth=40,
                 dropout=False, name='DenseNet'):
        super(DenseNet, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(nn.ConvolutionalLayer(self.input_shape, first_output, 3, activation='linear',
                                          layer_name=name+'pre_conv'))
        n = (depth - 1) // num_blocks
        for b in range(num_blocks):
            self.append(nn.DenseBlock(self.output_shape, num_conv_layer=n - 1, growth_rate=growth_rate,
                                      dropout=dropout, layer_name=name+'dense_block_%d' % b))
            if b < num_blocks - 1:
                self.append(nn.DenseBlock(self.output_shape, True, None, None, dropout,
                                          layer_name=name+'dense_block_transit_%d' % b))

        self.append(nn.BatchNormLayer(self.output_shape, layer_name=name+'post_bn'))
        if fc:
            self.append(nn.GlobalAveragePoolingLayer(input_shape, name+'_glbavgpooling'))
            self.append(nn.SoftmaxLayer(self.output_shape, num_classes, name+'_softmax'))


def ResNet18(input_shape, num_filters=64, activation='relu', fc=True, pooling=True, num_classes=1000, name='ResNet18', **kwargs):
    return ResNet(input_shape, nn.ResNetBlock, (2, 2, 2, 2), num_filters, activation, fc, pooling, num_classes, name, **kwargs)


def ResNet34(input_shape, num_filters=64, activation='relu', fc=True, pooling=True, num_classes=1000, name='ResNet34', **kwargs):
    return ResNet(input_shape, nn.ResNetBlock, (3, 4, 6, 3), num_filters, activation, fc, pooling, num_classes, name, **kwargs)


def ResNet50(input_shape, num_filters=64, activation='relu', fc=True, pooling=True, num_classes=1000, name='ResNet50', **kwargs):
    return ResNet(input_shape, nn.ResNetBottleneckBlock, (3, 4, 6, 3), num_filters, activation, fc, pooling, num_classes, name, **kwargs)


def ResNet101(input_shape, num_filters=64, activation='relu', fc=True, pooling=True, num_classes=1000, name='ResNet101', **kwargs):
    return ResNet(input_shape, nn.ResNetBottleneckBlock, (3, 4, 23, 3), num_filters, activation, fc, pooling, num_classes, name, **kwargs)


def ResNet152(input_shape, num_filters=64, activation='relu', fc=True, pooling=True, num_classes=1000, name='ResNet152', **kwargs):
    return ResNet(input_shape, nn.ResNetBottleneckBlock, (3, 8, 36, 3), num_filters, activation, fc, pooling, num_classes, name, **kwargs)
