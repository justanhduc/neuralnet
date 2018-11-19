'''
Original version from https://github.com/igul222/improved_wgan_training
Collected and modified by Nguyen Anh Duc
'''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import pickle as pickle
from scipy.misc import imsave
import os
from shutil import copyfile
import visdom
import time
import theano
from theano.compile import function_module as fm
from theano import tensor as T

from neuralnet import utils, model

__all__ = ['track', 'get_tracked_vars', 'eval_tracked_vars', 'Monitor']

_TRACKS = collections.OrderedDict()


def track(name, x):
    assert isinstance(name, str), 'name must be a string, got %s.' % type(name)
    assert isinstance(x, T.TensorVariable), 'x must be a Theano TensorVariable, got %s.' % type(x)
    _TRACKS[name] = x
    return x


def get_tracked_vars(name=None, return_name=False):
    assert isinstance(name, (str, list, tuple)) or name is None, 'name must either be None, a tring, or a list/tuple.'
    if name is None:
        tracked = ([n for n in _TRACKS.keys()], [val for val in _TRACKS.values()]) if return_name \
            else [val for val in _TRACKS.values()]
        return tracked
    elif isinstance(name, (list, tuple)):
        tracked = (name, [_TRACKS[n] for n in name]) if return_name else [_TRACKS[n] for n in name]
        return tracked
    else:
        tracked = (name, _TRACKS[name]) if return_name else _TRACKS[name]
        return tracked


def eval_tracked_vars(feed_dict):
    name, vars = get_tracked_vars(return_name=True)
    dict = collections.OrderedDict()
    for n, v in zip(name, vars):
        try:
            dict[n] = v.eval(feed_dict)
        except fm.UnusedInputError:
            func = theano.function([], v, givens=feed_dict, on_unused_input='ignore')
            dict[n] = func()
    return dict


class Monitor(utils.ConfigParser):
    def __init__(self, config_file=None, model_name='my_model', root='results', current_folder=None, checkpoint=0,
                 use_visdom=False, server='http://localhost', port=8097, disable_visdom_logging=True, valid_freq=None):
        super(Monitor, self).__init__(config_file)
        self.__num_since_beginning = collections.defaultdict(lambda: {})
        self.__num_since_last_flush = collections.defaultdict(lambda: {})
        self.__img_since_last_flush = collections.defaultdict(lambda: {})
        self.__tensor_since_last_flush = collections.defaultdict(lambda: {})
        self.__iter = [checkpoint]
        self.__timer = time.time()

        if self.config:
            self.name = self.config['model'].get('name', model_name)
            self.root = self.config['result'].get('root', root)
            self.valid_freq = self.config['training'].get('validation_frequency', valid_freq)
        else:
            self.root = root
            self.name = model_name
            self.valid_freq = valid_freq

        if current_folder:
            self.current_folder = current_folder
        else:
            self.path = os.path.join(self.root, self.name)
            os.makedirs(self.path, exist_ok=True)
            subfolders = os.listdir(self.path)
            self.current_folder = os.path.join(self.path, 'run%d' % (len(subfolders) + 1))
            idx = 1
            while os.path.exists(self.current_folder):
                self.current_folder = os.path.join(self.path, 'run%d' % (len(subfolders) + 1 + idx))
                idx += 1
            os.makedirs(self.current_folder, exist_ok=True)

        if config_file:
            copyfile(config_file, '%s/network_config.config' % self.current_folder)
        self.dump_files = collections.OrderedDict()

        self.use_visdom = use_visdom
        if use_visdom:
            if disable_visdom_logging:
                import logging
                logging.disable(logging.CRITICAL)
            self.vis = visdom.Visdom(server=server, port=port)
            if not self.vis.check_connection():
                from subprocess import Popen, PIPE
                Popen('visdom', stdout=PIPE, stderr=PIPE)
            self.vis.close()
            print('You can navigate to \'%s:%d\' for visualization' % (server, port))
        print('Result folder: %s' % self.current_folder)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__iter[0] % self.valid_freq == 0:
            self.flush()
        self.tick()

    def dump_model(self, network):
        assert isinstance(network, model.Model), 'network must be an instance of Model, got {}.'.format(type(network))
        with open('%s/network.txt' % self.current_folder, 'w') as outfile:
            outfile.write("\n".join(str(x) for x in network))

    def tick(self):
        self.__iter[0] += 1

    def plot(self, name, value):
        self.__num_since_last_flush[name][self.__iter[0]] = value

    def save_image(self, name, value, callback=lambda x: x):
        self.__img_since_last_flush[name][self.__iter[0]] = callback(value)

    def hist(self, name, value):
        self.__tensor_since_last_flush[name][self.__iter[0]] = value

    def flush(self, use_visdom_for_plots=None, use_visdom_for_image=None):
        use_visdom_for_plots = self.use_visdom if use_visdom_for_plots is None else use_visdom_for_plots
        use_visdom_for_image = self.use_visdom if use_visdom_for_image is None else use_visdom_for_image

        prints = []
        for name, vals in list(self.__num_since_last_flush.items()):
            self.__num_since_beginning[name].update(vals)

            x_vals = np.sort(list(self.__num_since_beginning[name].keys()))
            fig = plt.figure()
            fig.clf()
            plt.xlabel('iteration')
            plt.ylabel(name)
            y_vals = [self.__num_since_beginning[name][x] for x in x_vals]
            if isinstance(y_vals[0], dict):
                keys = list(y_vals[0].keys())
                y_vals = [tuple([y_val[k] for k in keys]) for y_val in y_vals]
                plot = plt.plot(x_vals, y_vals)
                plt.legend(plot, keys)
                prints.append("{}\t{}".format(name,
                                              np.mean(np.array([[val[k] for k in keys] for val in vals.values()]), 0)))
            else:
                plt.plot(x_vals, y_vals)
                prints.append("{}\t{}".format(name, np.mean(np.array(list(vals.values())), 0)))
            fig.savefig(os.path.join(self.current_folder, name.replace(' ', '_') + '.jpg'))
            if use_visdom_for_plots:
                self.vis.matplot(fig, win=name)
            plt.close(fig)
        self.__num_since_last_flush.clear()

        for name, vals in list(self.__img_since_last_flush.items()):
            for val in vals.values():
                if val.dtype != 'uint8':
                    val = (255.99 * val).astype('uint8')
                if len(val.shape) == 4:
                    if use_visdom_for_image:
                        self.vis.images(val, win=name)
                    for num in range(val.shape[0]):
                        img = val[num]
                        if img.shape[0] == 3:
                            img = np.transpose(img, (1, 2, 0))
                            imsave(
                                os.path.join(self.current_folder, name.replace(' ', '_') + '_%d.jpg' % num), img)
                        else:
                            for ch in range(img.shape[0]):
                                img_normed = (img[ch] - np.min(img[ch])) / (np.max(img[ch]) - np.min(img[ch]))
                                imsave(os.path.join(self.current_folder,
                                                    name.replace(' ', '_') + '_%d_%d.jpg' % (num, ch)), img_normed)
                elif len(val.shape) == 3 or len(val.shape) == 2:
                    if use_visdom_for_image:
                        self.vis.image(val if len(val.shape) == 2 else np.transpose(val, (2, 0, 1)), win=name)
                    imsave(os.path.join(self.current_folder, name.replace(' ', '_') + '.jpg'), val)
                else:
                    raise NotImplementedError
        self.__img_since_last_flush.clear()

        for name, vals in list(self.__tensor_since_last_flush.items()):
            k = max(list(self.__tensor_since_last_flush[name].keys()))
            val = vals[k].flatten()
            fig = plt.figure()
            fig.clf()
            plt.hist(val, bins='auto')
            fig.savefig(os.path.join(self.current_folder, name.replace(' ', '_') + '_hist.jpg'))
            plt.close(fig)
        self.__tensor_since_last_flush.clear()

        with open(os.path.join(self.current_folder, 'log.pkl'), 'wb') as f:
            pickle.dump(dict(self.__num_since_beginning), f, pickle.HIGHEST_PROTOCOL)

        print("Elapsed time {:.2f}min \t Iteration {}\t{}".format((time.time() - self.__timer) / 60., self.__iter[0],
                                                                  "\t".join(prints)))

    def dump(self, obj, file, keep=-1):
        assert isinstance(keep, int), 'keep must be an int, got %s' % type(keep)

        file = os.path.join(self.current_folder, file)
        if keep < 2:
            with open(file, 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
                f.close()
            print('Object dumped to %s' % file)
        else:
            name, ext = os.path.splitext(file)
            file_name = name + '-%d' % self.__iter[0] + ext

            if self.dump_files.get(file, None) is None:
                self.dump_files[file] = []
            self.dump_files[file].append(self.__iter[0])

            with open(file_name, 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
                f.close()
            print('Object dumped to %s' % file_name)

            if len(self.dump_files[file]) > keep:
                oldest_key = self.dump_files[file][0]
                file_name = name + '-%d' % oldest_key + ext
                if os.path.exists(file_name):
                    os.remove(file_name)
                else:
                    print("The oldest saved file does not exist")
                self.dump_files[file].remove(oldest_key)

    def load(self, file, version=-1):
        assert isinstance(version, int), 'keep must be an int, got %s' % type(version)

        full_file = os.path.join(self.current_folder, file)
        versions = self.dump_files.get(full_file, [])
        if version <= 0:
            if len(versions) > 0:
                latest = versions[-1]
                name, ext = os.path.splitext(full_file)
                file_name = name + '-%d' % latest + ext
                with open(file_name, 'rb') as f:
                    obj = pickle.load(f)
                    f.close()
            else:
                with open(full_file, 'rb') as f:
                    obj = pickle.load(f)
                    f.close()
        else:
            if len(versions) == 0:
                print('No file named %s found' % file)
                return None
            else:
                name, ext = os.path.splitext(full_file)
                if version in versions:
                    file_name = name + '-%d' % version + ext
                    with open(file_name, 'rb') as f:
                        obj = pickle.load(f)
                        f.close()
                else:
                    print('No file at version %d found' % version)
                    return None
        text = str(version) if version > 0 else 'latest'
        print('Version \'%s\' loaded' % text)
        return obj

    def reset(self):
        self.__num_since_beginning = collections.defaultdict(lambda: {})
        self.__num_since_last_flush = collections.defaultdict(lambda: {})
        self.__img_since_last_flush = collections.defaultdict(lambda: {})
        self.__tensor_since_last_flush = collections.defaultdict(lambda: {})
        self.__iter = [0]
        self.__timer = 0.

    imwrite = save_image
