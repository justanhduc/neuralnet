import theano
import time

from neuralnet import ConfigParser


class Training(ConfigParser):
    def __init__(self, config_file, **kwargs):
        super(Training, self).__init__(config_file, **kwargs)

        self.n_epochs = self.config['training']['n_epochs']
        self.continue_training = self.config['training']['continue']
        self.batch_size = self.config['training']['batch_size']
        self.validation_frequency = self.config['training']['validation_frequency']
        self.validation_batch_size = self.config['training']['validation_batch_size']
        self.display_cost = self.config['training']['display_cost']
        self.save_path = self.config['training']['save_path']
        self.extract_params = self.config['training']['extract_params']
        self.param_file = self.config['training']['param_file']
        self.testing_batch_size = self.config['testing']['batch_size']

        if self.display_cost:
            import os
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)

    def compile(self, inputs, outputs=None, mode=None, updates=None, givens=None, no_default_updates=False,
                accept_inplace=False, name='theano_function', rebuild_strict=True, allow_input_downcast=False, profile=None,
                on_unused_input='warn'):
        start_time = time.time()
        print('Compiling %s graph...' % name)
        f = theano.function(inputs, outputs=outputs, mode=mode, updates=updates, givens=givens,
                               no_default_updates=no_default_updates, accept_inplace=accept_inplace, name=name,
                               rebuild_strict=rebuild_strict, allow_input_downcast=allow_input_downcast, profile=profile,
                               on_unused_input=on_unused_input)
        print('Compilation took %.2f minutes.' % ((time.time() - start_time) / 60.))
        return f
