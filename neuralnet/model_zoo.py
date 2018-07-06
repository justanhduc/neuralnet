import numpy as np
import h5py
import theano
from theano import tensor as T

import neuralnet as nn


class ResNet(nn.Layer):
    def __init__(self, input_shape, block, layers, num_filters, activation='relu', fc=True, pooling=True, num_classes=1000,
                 layer_name='ResNet', **kwargs):
        super(ResNet, self).__init__(input_shape, layer_name)
        self.activation = activation
        self.kwargs = kwargs
        self.network = nn.Sequential(input_shape=input_shape, layer_name=layer_name)
        self.network.append(nn.ConvNormAct(self.network.input_shape, num_filters, 7, stride=2, activation=activation, **kwargs))
        if pooling:
            self.network.append(nn.PoolingLayer(self.network.output_shape, (3, 3), stride=(2, 2), pad=1))
        self.shape = self.network.output_shape
        self.network.append(self._make_layer(block, num_filters, layers[0], name='block1'))
        self.network.append(self._make_layer(block, 2 * num_filters, layers[1], stride=2, name='block2'))
        self.network.append(self._make_layer(block, 4 * num_filters, layers[2], stride=2, name='block3'))
        self.network.append(self._make_layer(block, 8 * num_filters, layers[3], stride=2, name='block4'))

        if fc:
            self.network.append(nn.GlobalAveragePoolingLayer(self.network.output_shape, layer_name='glb_avg_pooling'))
            self.network.append(nn.FullyConnectedLayer(self.network.output_shape, num_classes, activation='softmax',
                                                       layer_name='output'))

        self.params = list(self.network.params)
        self.trainable = list(self.network.trainable)
        self.regularizable = list(self.network.regularizable)
        self.descriptions = self.network.descriptions

    def _make_layer(self, block, planes, blocks, stride=1, name=''):
        downsample = None
        if stride != 1 or self.shape[1] != planes * block.upscale_factor:
            downsample = True

        layers = [block(self.shape, planes, stride, downsample=downsample, activation=self.activation,
                        layer_name=name + '_0', **self.kwargs)]
        self.shape = layers[-1].output_shape
        for i in range(1, blocks):
            layers.append(block(self.shape, planes, activation=self.activation, layer_name=name + '_%d' % i, **self.kwargs))
        return nn.Sequential(layers, layer_name=name)

    def get_output(self, input):
        return self.network(input)

    @property
    def output_shape(self):
        return self.network.output_shape


class VGG16(nn.Layer):
    def __init__(self, input_shape, fc=True, num_classes=1000, name='vgg16'):
        super(VGG16, self).__init__(input_shape, name)
        self.fc = fc
        self.model = nn.Sequential(input_shape=input_shape, layer_name=name)
        self.model.append(
            nn.ConvolutionalLayer(self.model.output_shape, 64, 3, no_bias=False, layer_name=name + '_conv1'))
        self.model.append(
            nn.ConvolutionalLayer(self.model.output_shape, 64, 3, no_bias=False, layer_name=name + '_conv2'))
        self.model.append(nn.MaxPoolingLayer(self.model.output_shape, (2, 2), layer_name=name + '_maxpool0'))

        self.model.append(
            nn.ConvolutionalLayer(self.model.output_shape, 128, 3, no_bias=False, layer_name=name + '_conv3'))
        self.model.append(
            nn.ConvolutionalLayer(self.model.output_shape, 128, 3, no_bias=False, layer_name=name + '_conv4'))
        self.model.append(nn.MaxPoolingLayer(self.model.output_shape, (2, 2), layer_name=name + '_maxpool1'))

        self.model.append(
            nn.ConvolutionalLayer(self.model.output_shape, 256, 3, no_bias=False, layer_name=name + '_conv5'))
        self.model.append(
            nn.ConvolutionalLayer(self.model.output_shape, 256, 3, no_bias=False, layer_name=name + '_conv6'))
        self.model.append(
            nn.ConvolutionalLayer(self.model.output_shape, 256, 3, no_bias=False, layer_name=name + '_conv7'))
        self.model.append(nn.MaxPoolingLayer(self.model.output_shape, (2, 2), layer_name=name + '_maxpool2'))

        self.model.append(
            nn.ConvolutionalLayer(self.model.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv8'))
        self.model.append(
            nn.ConvolutionalLayer(self.model.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv9'))
        self.model.append(
            nn.ConvolutionalLayer(self.model.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv10'))
        self.model.append(nn.MaxPoolingLayer(self.model.output_shape, (2, 2), layer_name=name + '_maxpool3'))

        self.model.append(
            nn.ConvolutionalLayer(self.model.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv11'))
        self.model.append(
            nn.ConvolutionalLayer(self.model.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv12'))
        self.model.append(
            nn.ConvolutionalLayer(self.model.output_shape, 512, 3, no_bias=False, layer_name=name + '_conv13'))

        if fc:
            self.model.append(nn.MaxPoolingLayer(self.model.output_shape, (2, 2), layer_name=name+'_maxpool4'))
            self.model.append(nn.FullyConnectedLayer(self.model.output_shape, 4096, layer_name=name+'_fc1'))
            self.model.append(nn.FullyConnectedLayer(self.model.output_shape, 4096, layer_name=name+'_fc2'))
            self.model.append(nn.SoftmaxLayer(self.model.output_shape, num_classes, name+'_softmax'))

        self.params = list(self.model.params)
        self.trainable = list(self.model.trainable)
        self.regularizable = list(self.model.regularizable)
        self.descriptions = self.model.descriptions

    def get_output(self, input):
        return self.model(input)

    @property
    def output_shape(self):
        return self.model.output_shape

    def load_weights(self, filename):
        f = h5py.File(filename, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        filtered_layer_names = []
        for name in layer_names:
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if weight_names:
                filtered_layer_names.append(name)
        layer_names = filtered_layer_names

        filtered_layers = []
        for layer in self.model:
            if 'pool' in layer.layer_name:
                continue
            filtered_layers.append(layer)

        weight_value_tuples = []
        for k, name in enumerate(layer_names):
            if not self.fc:
                if 'fc' in name or 'predictions' in name:
                    continue
            layer = filtered_layers[k]
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            weight_values = [np.transpose(g[weight_name], (3, 2, 0, 1)) if len(g[weight_name].shape) == 4
                             else g[weight_name] for weight_name in weight_names]
            symbolic_weights = tuple(layer.params)
            weight_value_tuples += zip(symbolic_weights, weight_values)
        nn.utils.batch_set_value(weight_value_tuples)
        print('Pretrained weights loaded successfully!')


class DenseNet(nn.Layer):
    def __init__(self, input_shape, fc=True, num_classes=1000, first_output=16, growth_rate=12, num_blocks=3, depth=40,
                 dropout=False, name='DenseNet'):
        super(DenseNet, self).__init__(input_shape, name)

        self.model = nn.Sequential(input_shape=input_shape, layer_name=name)
        self.model.append(nn.ConvolutionalLayer(self.input_shape, first_output, 3, activation='linear',
                                                layer_name=name+'pre_conv'))
        n = (depth - 1) // num_blocks
        for b in range(num_blocks):
            self.model.append(nn.DenseBlock(self.model.output_shape, num_conv_layer=n - 1, growth_rate=growth_rate,
                                            dropout=dropout, layer_name=name+'dense_block_%d' % b))
            if b < num_blocks - 1:
                self.model.append(nn.DenseBlock(self.model.output_shape, True, None, None, dropout,
                                                layer_name=name+'dense_block_transit_%d' % b))

        self.model.append(nn.BatchNormLayer(self.model.output_shape, layer_name=name+'post_bn'))
        if fc:
            self.model.append(
                nn.GlobalAveragePoolingLayer(input_shape, name+'_glbavgpooling'))
            self.model.append(nn.SoftmaxLayer(self.model.output_shape, num_classes, name+'_softmax'))

        self.params = list(self.model.params)
        self.trainable = list(self.model.trainable)
        self.regularizable = list(self.model.regularizable)
        self.descriptions = self.model.descriptions

    def get_output(self, input):
        return self.model(input)

    @property
    def output_shape(self):
        return self.model.output_shape


def resnet18(input_shape, num_filters, activation='relu', fc=True, pooling=True, num_classes=1000, name='ResNet18', **kwargs):
    return ResNet(input_shape, nn.ResNetBlock, (2, 2, 2, 2), num_filters, activation, fc, pooling, num_classes, name, **kwargs)


def resnet34(input_shape, num_filters, activation='relu', fc=True, pooling=True, num_classes=1000, name='ResNet34', **kwargs):
    return ResNet(input_shape, nn.ResNetBlock, (3, 4, 6, 3), num_filters, activation, fc, pooling, num_classes, name, **kwargs)


def resnet50(input_shape, num_filters, activation='relu', fc=True, pooling=True, num_classes=1000, name='ResNet50', **kwargs):
    return ResNet(input_shape, nn.ResNetBottleneckBlock, (3, 4, 6, 3), num_filters, activation, fc, pooling, num_classes, name, **kwargs)


def resnet101(input_shape, num_filters, activation='relu', fc=True, pooling=True, num_classes=1000, name='ResNet101', **kwargs):
    return ResNet(input_shape, nn.ResNetBottleneckBlock, (3, 4, 23, 3), num_filters, activation, fc, pooling, num_classes, name, **kwargs)


def resnet152(input_shape, num_filters, activation='relu', fc=True, pooling=True, num_classes=1000, name='ResNet152', **kwargs):
    return ResNet(input_shape, nn.ResNetBottleneckBlock, (3, 8, 36, 3), num_filters, activation, fc, pooling, num_classes, name, **kwargs)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    root = 'E:/Users/Jongyoo/DeepIQA2/'
    weight_file = root + 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    imagenet_classes_file = root + 'imagenet_classes.txt'
    image_list = [
        'ILSVRC2012_val_00000001.JPEG',
        'ILSVRC2012_val_00000002.JPEG',
        'ILSVRC2012_val_00000003.JPEG',
        'ILSVRC2012_val_00000004.JPEG',
        'ILSVRC2012_val_00000005.JPEG'
    ]

    X = T.tensor4('input')
    net = VGG16((None, 3, 224, 224))
    net.load_weights(weight_file)
    test = theano.function([X], net(X))

    with open(imagenet_classes_file, 'r') as f:
        classes = [s.strip() for s in f.readlines()]
    mean_bgr = np.array([103.939, 116.779, 123.68], dtype='float32')[None, None, :]
    for fname in image_list:
        print(fname)
        rawim, im = nn.utils.prep_image(root+fname, mean_bgr, 'rgb')
        prob = test(im)[0]
        print('Theano:')
        res = sorted(zip(classes, prob), key=lambda t: t[1], reverse=True)[:len(image_list)]
        for c, p in res:
            print('  ', c, p)

        plt.figure()
        plt.imshow(rawim.astype('uint8'))
        plt.axis('off')
        plt.show()
        print('\n')
