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

from neuralnet import utils, model


class Monitor(utils.ConfigParser):
    def __init__(self, config_file=None, model_name='my_model', root='results', current_folder=None, checkpoint=0):
        super(Monitor, self).__init__(config_file)
        self.__num_since_beginning = collections.defaultdict(lambda: {})
        self.__num_since_last_flush = collections.defaultdict(lambda: {})
        self.__img_since_last_flush = collections.defaultdict(lambda: {})
        self.__iter = [checkpoint]

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
        if current_folder:
            self.current_folder = current_folder
        else:
            self.current_folder = self.path + '/run%d' % (len(subfolders) + 1)
            idx = 1
            while os.path.exists(self.current_folder):
                self.current_folder = self.path + '/run%d' % (len(subfolders) + 1 + idx)
                idx += 1
            os.mkdir(self.current_folder)
        if config_file:
            copyfile(config_file, '%s/network_config.config' % self.current_folder)
        self.dump_files = collections.OrderedDict()
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

    def flush(self):
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
            plt.close(fig)
        self.__num_since_last_flush.clear()

        for name, vals in list(self.__img_since_last_flush.items()):
            for val in vals.values():
                if val.dtype != 'uint8':
                    val = (255.99 * val).astype('uint8')
                if len(val.shape) == 4:
                    for num in range(val.shape[0]):
                        img = val[num]
                        if img.shape[0] == 3:
                            img = np.transpose(img, (1, 2, 0))
                            imsave(self.current_folder + '/' + name + '_%d.jpg' % num, img)
                        else:
                            for ch in range(img.shape[0]):
                                imsave(self.current_folder + '/' + name + '_%d_%d.jpg' % (num, ch), img[ch])
                elif len(val.shape) == 3 or len(val.shape) == 2:
                    imsave(self.current_folder + '/' + name + '.jpg', val)
                else:
                    raise NotImplementedError
        self.__img_since_last_flush.clear()

        with open(self.current_folder + '/log.pkl', 'wb') as f:
            pickle.dump(dict(self.__num_since_beginning), f, pickle.HIGHEST_PROTOCOL)

        print("Iteration {}\t{}".format(self.__iter[0], "\t".join(prints)))

    def dump(self, obj, file, keep=-1):
        assert isinstance(keep, int), 'keep must be an int, got %s' % type(keep)

        file = self.current_folder + '/' + file
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

        full_file = self.current_folder + '/' + file
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
        self.__iter = [0]

    imwrite = save_image


if __name__ == '__main__':
    pass
