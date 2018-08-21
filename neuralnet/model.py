import numpy
import abc

import neuralnet as nn
from neuralnet import layers
from neuralnet.build_training import Training
from neuralnet.build_optimization import Optimization


class Model(Optimization, Training, nn.model_zoo.Net, metaclass=abc.ABCMeta):
    def __init__(self, config_file, **kwargs):
        super(Model, self).__init__(config_file, **kwargs)
        self.input_shape = (None,) + tuple(self.config['model']['input_shape'])
        self.model = layers.Sequential(input_shape=self.input_shape, layer_name=self.config['model']['name'])

    def __iter__(self):
        return self.model.__iter__()

    def __next__(self):
        return self.model.__next__()

    def __len__(self):
        return len(self.model)

    def __call__(self, input, *args, **kwargs):
        return self.inference(input, *args, **kwargs)

    def append(self, layer):
        assert isinstance(layer, (nn.Layer, nn.Sequential)), 'Expect \'layer\' to belong to {} or {}, got {}'.format(
            type(nn.Layer), type(nn.Sequential), type(layer))
        self.model.append(layer)

    def load_pretrained_params(self):
        return

    @abc.abstractmethod
    def inference(self, input, *args, **kwargs):
        pass

    def get_cost(self, input, gt, **kwargs):
        raise NotImplementedError

    def learn(self, input, gt, **kwargs):
        raise NotImplementedError

    @property
    def params(self):
        return self.model.params

    @property
    def trainable(self):
        return self.model.trainable

    @property
    def regularizable(self):
        return self.model.regularizable

    def save_learning(self, param_file=None):
        param_file = param_file if param_file else 'opt.npz'
        numpy.savez(param_file, **{p.name: p.get_value() for p in self.opt.accumulations})
        print('Optimization state dumped to %s' % param_file)

    def load_learning(self, param_file=None):
        param_file = param_file if param_file else 'opt.npz'
        acc = numpy.load(param_file)
        for p in self.opt.accumulations:
            try:
                p.set_value(acc[p.name])
            except KeyError:
                raise KeyError('There is no saved value for %s' % p.name)
        print('Optimization state loaded from %s' % param_file)

    def reset(self):
        self.model.reset()
        if self.opt:
            self.opt.reset()

    def __repr__(self):
        return self.model.descriptions

    @staticmethod
    def set_training_status(training):
        nn.set_training_status(training)
