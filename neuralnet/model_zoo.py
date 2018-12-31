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
    def __init__(self, input_shape, block, layers, num_filters, activation='relu', fc=True, pooling=True,
                 num_classes=1000, main_branch=None, res_branch=None, name='ResNet', **kwargs):
        super(ResNet, self).__init__(input_shape=input_shape, layer_name=name)
        self.activation = activation
        self.main_branch = main_branch
        self.res_branch = res_branch
        self.kwargs = kwargs
        self.append(nn.ConvNormAct(self.input_shape, num_filters, 7, stride=2, activation=activation,
                                   layer_name=name + '/first_conv', **kwargs))
        if pooling:
            self.append(nn.PoolingLayer(self.output_shape, (3, 3), stride=(2, 2), pad=1))
        self.append(self._make_layer(block, self.output_shape, num_filters, layers[0], name=name + '/block1'))
        self.append(
            self._make_layer(block, self.output_shape, 2 * num_filters, layers[1], stride=2, name=name + '/block2'))
        self.append(
            self._make_layer(block, self.output_shape, 4 * num_filters, layers[2], stride=2, name=name + '/block3'))
        self.append(
            self._make_layer(block, self.output_shape, 8 * num_filters, layers[3], stride=2, name=name + '/block4'))

        if fc:
            self.append(nn.GlobalAveragePoolingLayer(self.output_shape, layer_name=name + '/glb_avg_pooling'))
            self.append(nn.FullyConnectedLayer(self.output_shape, num_classes, activation='softmax',
                                               layer_name=name + 'output'))

    def _make_layer(self, block, shape, planes, blocks, stride=1, name=''):
        layers = [block(shape, planes, stride, activation=self.activation, layer_name=name + '_0',
                        block=self.main_branch, downsample=self.res_branch, **self.kwargs)]

        for i in range(1, blocks):
            layers.append(
                block(layers[-1].output_shape, planes, activation=self.activation, layer_name=name + '_%d' % i,
                      block=self.main_branch, downsample=self.res_branch, **self.kwargs))
        return nn.Sequential(layers, layer_name=name)


class VGG16(nn.Sequential, Net):
    def __init__(self, input_shape, fc=True, bn=False, dropout=True, border_mode='half', num_classes=1000,
                 name='vgg16'):
        super(VGG16, self).__init__(input_shape=input_shape, layer_name=name)
        self.fc = fc
        self.append(nn.Conv2DLayer(self.output_shape, 64, 3, nn.HeNormal('relu'), activation=None, no_bias=False,
                                   layer_name=name + '/conv1', border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn1') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu1'))
        self.append(nn.Conv2DLayer(self.output_shape, 64, 3, nn.HeNormal('relu'), activation=None, no_bias=False,
                                   layer_name=name + '/conv2', border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn2') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu2'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '_maxpool0'))

        self.append(nn.Conv2DLayer(self.output_shape, 128, 3, nn.HeNormal('relu'), activation=None, no_bias=False,
                                   layer_name=name + '/conv3', border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn3') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu3'))
        self.append(nn.Conv2DLayer(self.output_shape, 128, 3, nn.HeNormal('relu'), activation=None, no_bias=False,
                                   layer_name=name + '/conv4', border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn4') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu4'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '_maxpool1'))

        self.append(nn.Conv2DLayer(self.output_shape, 256, 3, nn.HeNormal('relu'), activation=None, no_bias=False,
                                   layer_name=name + '/conv5', border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn5') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu5'))
        self.append(nn.Conv2DLayer(self.output_shape, 256, 3, nn.HeNormal('relu'), activation=None, no_bias=False,
                                   layer_name=name + '/conv6', border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn6') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu6'))
        self.append(nn.Conv2DLayer(self.output_shape, 256, 3, nn.HeNormal('relu'), activation=None, no_bias=False,
                                   layer_name=name + '/conv7', border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn7') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu7'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '_maxpool2'))

        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, nn.HeNormal('relu'), activation=None, no_bias=False,
                                   layer_name=name + '/conv8', border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn8') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu8'))
        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, nn.HeNormal('relu'), activation=None, no_bias=False,
                                   layer_name=name + '/conv9', border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn9') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu9'))
        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, nn.HeNormal('relu'), activation=None, no_bias=False,
                                   layer_name=name + '/conv10', border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn10') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu10'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '_maxpool3'))

        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, nn.HeNormal('relu'), activation=None, no_bias=False,
                                   layer_name=name + '/conv11', border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn11') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu11'))
        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, nn.HeNormal('relu'), activation=None, no_bias=False,
                                   layer_name=name + '/conv12', border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn11') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu12'))
        self.append(nn.Conv2DLayer(self.output_shape, 512, 3, nn.HeNormal('relu'), activation=None, no_bias=False,
                                   layer_name=name + '/conv13', border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn13') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu3'))

        if fc:
            self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '_maxpool4'))
            self.append(nn.FCLayer(self.output_shape, 4096, layer_name=name + '_fc1'))
            if dropout:
                self.append(nn.DropoutLayer(self.output_shape, drop_prob=.5, layer_name=name + '/dropout1'))
            self.append(nn.FCLayer(self.output_shape, 4096, layer_name=name + '_fc2'))
            if dropout:
                self.append(nn.DropoutLayer(self.output_shape, drop_prob=.5, layer_name=name + '/dropout2'))
            self.append(nn.SoftmaxLayer(self.output_shape, num_classes, name + '_softmax'))


class VGG19(nn.Sequential, Net):
    def __init__(self, input_shape, fc=True, bn=False, dropout=True, border_mode='half', num_classes=1000,
                 name='vgg19'):
        super(VGG19, self).__init__(input_shape=input_shape, layer_name=name)
        self.fc = fc
        self.append(
            nn.Conv2DLayer(self.output_shape, 64, 3, activation=None, no_bias=False, layer_name=name + '/conv1_1',
                           border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn1') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu1'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 64, 3, activation=None, no_bias=False, layer_name=name + '/conv1_2',
                           border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn2') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu2'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '_maxpool0'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 128, 3, activation=None, no_bias=False, layer_name=name + '/conv2_1',
                           border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn3') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu3'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 128, 3, activation=None, no_bias=False, layer_name=name + '/conv2_2',
                           border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn4') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu4'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '_maxpool1'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, activation=None, no_bias=False, layer_name=name + '/conv3_1',
                           border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn5') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu5'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, activation=None, no_bias=False, layer_name=name + '/conv3_2',
                           border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn6') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu6'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, activation=None, no_bias=False, layer_name=name + '/conv3_3',
                           border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn7') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu7'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 256, 3, activation=None, no_bias=False, layer_name=name + '/conv3_4',
                           border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn8') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu8'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '_maxpool2'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 512, 3, activation=None, no_bias=False, layer_name=name + '/conv4_1',
                           border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn9') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu9'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 512, 3, activation=None, no_bias=False, layer_name=name + '/conv4_2',
                           border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn10') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu10'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 512, 3, activation=None, no_bias=False, layer_name=name + '/conv4_3',
                           border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn11') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu11'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 512, 3, activation=None, no_bias=False, layer_name=name + '/conv4_4',
                           border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn12') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu12'))
        self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '_maxpool3'))

        self.append(
            nn.Conv2DLayer(self.output_shape, 512, 3, activation=None, no_bias=False, layer_name=name + '/conv5_1',
                           border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn13') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu13'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 512, 3, activation=None, no_bias=False, layer_name=name + '/conv5_2',
                           border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn14') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu14'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 512, 3, activation=None, no_bias=False, layer_name=name + '/conv5_3',
                           border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn15') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu15'))
        self.append(
            nn.Conv2DLayer(self.output_shape, 512, 3, activation=None, no_bias=False, layer_name=name + '/conv5_4',
                           border_mode=border_mode))
        self.append(nn.BatchNormLayer(self.output_shape, name + '/bn16') if bn
                    else nn.ActivationLayer(self.output_shape, layer_name=name + '/relu16'))

        if fc:
            self.append(nn.MaxPoolingLayer(self.output_shape, (2, 2), layer_name=name + '_maxpool4'))
            self.append(nn.FCLayer(self.output_shape, 4096, layer_name=name + '_fc1'))
            if dropout:
                self.append(nn.DropoutLayer(self.output_shape, drop_prob=.5, layer_name=name + '/dropout1'))
            self.append(nn.FCLayer(self.output_shape, 4096, layer_name=name + '_fc2'))
            if dropout:
                self.append(nn.DropoutLayer(self.output_shape, drop_prob=.5, layer_name=name + '/dropout2'))
            self.append(nn.SoftmaxLayer(self.output_shape, num_classes, name + '_softmax'))


class DenseNet(nn.Sequential, Net):
    def __init__(self, input_shape, fc=True, num_classes=1000, first_output=16, growth_rate=12, num_blocks=3, depth=40,
                 dropout=False, name='DenseNet'):
        super(DenseNet, self).__init__(input_shape=input_shape, layer_name=name)
        self.append(nn.ConvolutionalLayer(self.input_shape, first_output, 3, activation='linear',
                                          layer_name=name + 'pre_conv'))
        n = (depth - 1) // num_blocks
        for b in range(num_blocks):
            self.append(nn.DenseBlock(self.output_shape, num_conv_layer=n - 1, growth_rate=growth_rate,
                                      dropout=dropout, layer_name=name + '/dense_block_%d' % b))
            if b < num_blocks - 1:
                self.append(nn.DenseBlock(self.output_shape, True, None, None, dropout,
                                          layer_name=name + '/dense_block_transit_%d' % b))

        self.append(nn.BatchNormLayer(self.output_shape, layer_name=name + '/post_bn'))
        if fc:
            self.append(nn.GlobalAveragePoolingLayer(input_shape, name + '/glbavgpooling'))
            self.append(nn.SoftmaxLayer(self.output_shape, num_classes, name + '_softmax'))


def ResNet18(input_shape, num_filters=64, activation='relu', fc=True, pooling=True, num_classes=1000, name='ResNet18',
             **kwargs):
    return ResNet(input_shape, nn.ResNetBlock, (2, 2, 2, 2), num_filters, activation, fc, pooling, num_classes, name,
                  **kwargs)


def ResNet34(input_shape, num_filters=64, activation='relu', fc=True, pooling=True, num_classes=1000, name='ResNet34',
             **kwargs):
    return ResNet(input_shape, nn.ResNetBlock, (3, 4, 6, 3), num_filters, activation, fc, pooling, num_classes, name,
                  **kwargs)


def ResNet50(input_shape, num_filters=64, activation='relu', fc=True, pooling=True, num_classes=1000, name='ResNet50',
             **kwargs):
    return ResNet(input_shape, nn.ResNetBottleneckBlock, (3, 4, 6, 3), num_filters, activation, fc, pooling,
                  num_classes, name, **kwargs)


def ResNet101(input_shape, num_filters=64, activation='relu', fc=True, pooling=True, num_classes=1000, name='ResNet101',
              **kwargs):
    return ResNet(input_shape, nn.ResNetBottleneckBlock, (3, 4, 23, 3), num_filters, activation, fc, pooling,
                  num_classes, name, **kwargs)


def ResNet152(input_shape, num_filters=64, activation='relu', fc=True, pooling=True, num_classes=1000, name='ResNet152',
              **kwargs):
    return ResNet(input_shape, nn.ResNetBottleneckBlock, (3, 8, 36, 3), num_filters, activation, fc, pooling,
                  num_classes, name, **kwargs)
