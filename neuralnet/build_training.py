import theano
import time

from neuralnet import ConfigParser

__all__ = ['compile', 'Training', 'function']


class Training(ConfigParser):
    def __init__(self, config_file, **kwargs):
        super(Training, self).__init__(config_file, **kwargs)
        self.n_epochs = self.config['training'].get('n_epochs', None)
        self.continue_training = self.config['training'].get('continue', False)
        self.batch_size = self.config['training'].get('batch_size', None)
        self.validation_frequency = self.config['training'].get('validation_frequency', None)
        self.validation_batch_size = self.config['training'].get('validation_batch_size', None)
        self.extract_params = self.config['training'].get('extract_params', False)
        self.param_file = self.config['training'].get('param_file', None)
        self.testing_batch_size = self.config['testing'].get('batch_size', self.batch_size)


def compile(inputs, outputs=None, mode=None, updates=None, givens=None, no_default_updates=False,
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


function = compile
