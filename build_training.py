import theano
import time

from utils import ConfigParser


class Training(ConfigParser):
    def __init__(self, config_file, **kwargs):
        super(Training, self).__init__(config_file, **kwargs)

        self.n_epochs = self.config['training']['n_epochs']
        self.continue_training = self.config['training']['continue']
        self.continue_checkpoint = self.config['training']['checkpoint']
        self.batch_size = self.config['training']['batch_size']
        self.batch_size_testing = self.config['testing']['batch_size']
        self.save_model = self.config['load_existing']['save_model']
        self.checkpoint = self.config['load_existing']['checkpoint']
        self.extract_params = self.config['load_existing']['extract_params']
        self.param_file = self.config['load_existing']['param_file']

    def compile(self, inputs, outputs=None, mode=None, updates=None, givens=None, no_default_updates=False,
                accept_inplace=False, name='theano_function', rebuild_strict=True, allow_input_downcast=False, profile=None,
                on_unused_input='warn'):
        start_time = time.time()
        print 'Compiling %s graph...' % name
        f = theano.function(inputs, outputs=outputs, mode=mode, updates=updates, givens=givens,
                               no_default_updates=no_default_updates, accept_inplace=accept_inplace, name=name,
                               rebuild_strict=rebuild_strict, allow_input_downcast=allow_input_downcast, profile=profile,
                               on_unused_input=on_unused_input)
        print 'Compilation took %.2f minutes.' % ((time.time() - start_time) / 60.)
        return f
