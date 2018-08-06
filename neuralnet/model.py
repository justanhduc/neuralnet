import numpy
import abc

import neuralnet as nn
from neuralnet import layers
from neuralnet.build_training import Training
from neuralnet.build_optimization import Optimization


class Model(Optimization, Training, metaclass=abc.ABCMeta):
    def __init__(self, config_file, **kwargs):
        super(Model, self).__init__(config_file, **kwargs)
        self.input_shape = (None, self.config['model']['input_shape'][2]) + tuple(
            self.config['model']['input_shape'][:2])
        self.model = layers.Sequential(input_shape=self.input_shape, layer_name=self.config['model']['name'])

    def __iter__(self):
        return self.model.__iter__()

    def __next__(self):
        return self.model.__next__()

    def __len__(self):
        return len(self.model)

    def __call__(self, input, *args, **kwargs):
        return self.inference(input, *args, **kwargs)

    def add(self, layer):
        assert isinstance(layer, layers.Layer), 'Expect \'layer\' to belong to {}, got {}'.format(type(layers.Layer), type(layer))
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

    def save_params(self, param_file=None):
        param_file = param_file if param_file else self.param_file
        numpy.savez(param_file, **{p.name: p.get_value() for p in self.params})
        print('Model weights dumped to %s' % param_file)

    def load_params(self, param_file=None):
        param_file = param_file if param_file else self.param_file
        weights = numpy.load(param_file)
        for p in self.params:
            try:
                p.set_value(weights[p.name])
            except KeyError:
                KeyError('There is no saved weight for %s' % p.name)
        print('Model weights loaded from %s' % param_file)

    def reset(self):
        self.model.reset()
        if self.opt:
            self.opt.reset()

    def __repr__(self):
        return self.model.descriptions

    @staticmethod
    def set_training_status(training):
        nn.set_training_status(training)
