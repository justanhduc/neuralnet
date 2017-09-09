import numpy

import utils
from build_optimization import Optimization
from build_training import Training
import layers


class BaseModel(Optimization, Training):
    def __init__(self, config_file, **kwargs):
        super(BaseModel, self).__init__(config_file, **kwargs)
        self.model = []
        self.params = []
        self.regularizable = []
        self.index = 0

    def __iter__(self):
        return self

    def next(self):
        if len(self.model) == 0 or self.index == len(self.model):
            self.index = 0
            raise StopIteration
        else:
            self.index += 1
            return self.model[self.index - 1]

    def __len__(self):
        return len(self.model)

    def inference(self, input):
        return utils.inference(input, self.model)

    def save_params(self):
        numpy.savez(self.param_file, **{p.name: p.get_value() for p in self.params})
        print 'Model weights dumped to %s' % self.param_file

    def load_params(self):
        print 'Loading model weights from %s' % self.param_file
        weights = numpy.load(self.param_file)
        for p in self.params:
            try:
                p.set_value(weights[p.name])
            except:
                NameError('There is no saved weight for %s' % p.name)

    @staticmethod
    def set_training_status(training):
        layers.set_training_status(training)
