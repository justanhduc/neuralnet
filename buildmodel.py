import cPickle as pkl

import utils
from buildstructure import Structure
from buildcost import Optimization
import layers


class Model(Optimization, Structure):
    def __init__(self, config_file, **kwargs):
        super(Model, self).__init__(config_file, **kwargs)
        self.config = utils.load_configuration(config_file)
        self.n_epochs = self.config['training']['n_epochs']
        self.continue_training = self.config['training']['continue']
        self.continue_checkpoint = self.config['training']['checkpoint']
        self.multi_gpus = self.config['training']['multi_gpus']
        self.display_cost = self.config['training']['display_cost']
        self.batch_size_testing = self.config['testing']['batch_size']
        self.get_test_output = self.config['testing']['get_output']
        self.summary_dir = self.config['summary_dir']
        self.save_model = self.config['save_load']['save_model']
        self.checkpoint = self.config['save_load']['checkpoint']
        self.checkpoint_dir = self.config['save_load']['checkpoint_dir']
        self.extract_params = self.config['save_load']['extract_params']
        self.param_file_to_save = self.config['save_load']['param_file']
        self.build_model(**kwargs)

    def __iter__(self):
        return self

    def next(self):
        if len(self.layers) == 0 or self.index == len(self.layers):
            self.index = 0
            raise StopIteration
        else:
            self.index += 1
            return self.layers[self.index - 1]

    def __len__(self):
        return len(self.layers)

    def build_model(self, **kwargs):
        if self.continue_training:
            model = pkl.load(open(self.continue_checkpoint, 'rb'))
            self.layers = list(model.layers)
        else:
            super(Model, self).build_model(**kwargs)

    def inference(self, input, **kwargs):
        return super(Model, self).inference(input=input)

    def build_cost(self, y, **kwargs):
        y_pred = kwargs.get('output', self.output)
        kwargs['model'] = self.layers if 'model' not in kwargs else kwargs['model']
        return super(Model, self).build_cost(y_pred, y, **kwargs)

    def build_updates(self, **kwargs):
        cost_to_optimize = self.cost if 'cost' not in kwargs else kwargs['cost']
        kwargs['model'] = self.layers if 'model' not in kwargs else kwargs['model']
        return super(Model, self).build_updates(cost_to_optimize, **kwargs)

    def decrease_learning_rate(self, **kwargs):
        super(Model, self).decrease_learning_rate(**kwargs)

    def reset(self):
        for layer in self.layers:
            layer.reset()

    @staticmethod
    def set_training_status(training):
        layers.DropoutGaussianLayer.turn_dropout_on(training)
        layers.DropoutLayer.turn_dropout_on(training)
        layers.BatchNormLayer.set_training(training)
