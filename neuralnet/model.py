import numpy
import abc

from neuralnet import layers
from neuralnet import Training
from neuralnet import Optimization


class Model(Optimization, Training, metaclass=abc.ABCMeta):
    def __init__(self, config_file, **kwargs):
        super(Model, self).__init__(config_file, **kwargs)
        self.input_shape = (None, self.config['model']['input_shape'][2]) + tuple(
            self.config['model']['input_shape'][:2])
        self.model = layers.Sequential(input_shape=self.input_shape, layer_name=self.config['model']['name'])
        self.params = []
        self.trainable = []
        self.regularizable = []
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if len(self.model) == 0 or self.idx == len(self.model):
            raise StopIteration
        else:
            self.idx += 1
            return self.model[self.idx - 1]

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

    def get_all_params(self):
        self.params = list(self.model.params)

    def get_trainable(self):
        self.trainable = list(self.model.trainable)

    def get_regularizable(self):
        self.regularizable = list(self.model.regularizable)

    def save_params(self):
        numpy.savez(self.param_file, **{p.name: p.get_value() for p in self.params})
        print('Model weights dumped to %s' % self.param_file)

    def load_params(self):
        weights = numpy.load(self.param_file)
        for p in self.params:
            try:
                p.set_value(weights[p.name])
            except KeyError:
                KeyError('There is no saved weight for %s' % p.name)
        print('Model weights loaded from %s' % self.param_file)

    def reset(self):
        for layer in self.model:
            layer.reset()
        if hasattr(self, 'opt'):
            self.opt.reset()

    def __repr__(self):
        return self.model.descriptions

    @staticmethod
    def set_training_status(training):
        layers.set_training_status(training)
