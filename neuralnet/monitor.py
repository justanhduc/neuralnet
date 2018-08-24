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

from neuralnet import utils, model


class Monitor(utils.ConfigParser):
    def __init__(self, config_file=None, model_name='my_model', root='results', use_visdom=False,
                 disable_visdom_logging=True):
        super(Monitor, self).__init__(config_file)
        self.__num_since_beginning = collections.defaultdict(lambda: {})
        self.__num_since_last_flush = collections.defaultdict(lambda: {})
        self.__img_since_last_flush = collections.defaultdict(lambda: {})
        self.__iter = [0]

        if self.config:
            self.name = self.config['model']['name']
            if self.config['result']['root']:
                self.root = self.config['result']['root']
            else:
                self.root = root
        else:
            self.root = root
            self.name = model_name

        self.path = self.root + '/' + self.name
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        subfolders = os.listdir(self.path)
        self.current_folder = self.path + '/run%d' % (len(subfolders) + 1)
        idx = 1
        while os.path.exists(self.current_folder):
            self.current_folder = self.path + '/run%d' % (len(subfolders) + 1 + idx)
            idx += 1
        os.mkdir(self.current_folder)
        if config_file:
            copyfile(config_file, '%s/network_config.config' % self.current_folder)

        self.use_visdom = use_visdom
        if use_visdom:
            if disable_visdom_logging:
                import logging
                logging.disable(logging.CRITICAL)
            self.vis = visdom.Visdom()
            if not self.vis.check_connection():
                from subprocess import Popen, PIPE
                Popen('visdom', stdout=PIPE, stderr=PIPE)
            self.vis.close()
            print('You can navigate to \'localhost:8097\' for visualization')
        print('Result folder: %s' % self.current_folder)

    def dump_model(self, network):
        assert isinstance(network, model.Model), 'network must be an instance of Model, got {}.'.format(type(network))
        with open('%s/network.txt' % self.current_folder, 'w') as outfile:
            outfile.write("\n".join(str(x) for x in network))

    def tick(self):
        self.__iter[0] += 1

    def plot(self, name, value):
        self.__num_since_last_flush[name][self.__iter[0]] = value

    def save_image(self, name, tensor_img, callback=lambda x: x):
        self.__img_since_last_flush[name][self.__iter[0]] = callback(tensor_img)

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
            fig.savefig(self.current_folder + '/' + name.replace(' ', '_')+'.jpg')
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
                            imsave(self.current_folder + '/' + name + '_%d.jpg' % num, img)
                        else:
                            for ch in range(img.shape[0]):
                                imsave(self.current_folder + '/' + name + '_%d_%d.jpg' % (num, ch), img[ch])
                elif len(val.shape) == 3 or len(val.shape) == 2:
                    if use_visdom_for_image:
                        self.vis.image(val if len(val.shape) == 2 else np.transpose(val, (2, 0, 1)), win=name)
                    imsave(self.current_folder + '/' + name + '.jpg', val)
                else:
                    raise NotImplementedError
        self.__img_since_last_flush.clear()

        with open(self.current_folder + '/log.pkl', 'wb') as f:
            pickle.dump(dict(self.__num_since_beginning), f, pickle.HIGHEST_PROTOCOL)

        print("Iteration {}\t{}".format(self.__iter[0], "\t".join(prints)))

    def reset(self):
        self.__num_since_beginning = collections.defaultdict(lambda: {})
        self.__num_since_last_flush = collections.defaultdict(lambda: {})
        self.__img_since_last_flush = collections.defaultdict(lambda: {})
        self.__iter = [0]


if __name__ == '__main__':
    mon = Monitor(None, use_visdom=True)
    for i in range(10):
        for j in range(5):
            mon.plot('train-valid', {'train': i+j+1, 'valid': i-j})
            mon.plot('x2', (i+j)*2)
            mon.save_image('toy', np.zeros((10, 3, 100, 100), dtype='float32') + i*j/40)
            mon.tick()
        mon.flush()
