import json
import sys
import threading
from queue import Queue
import time
import numpy as np
import theano
from theano import tensor as T
from scipy import misc

thread_lock = threading.Lock()


class Thread(threading.Thread):
    def __init__(self, id, name, func):
        threading.Thread.__init__(self)
        self.id = id
        self.name = name
        self.func = func

    def run(self):
        print('Starting ' + self.name)
        thread_lock.acquire()
        self.outputs = self.func()
        thread_lock.release()


class ConfigParser(object):
    def __init__(self, config_file, **kwargs):
        super(ConfigParser, self).__init__()
        self.config_file = config_file
        self.config = self.load_configuration()

    def load_configuration(self):
        try:
            with open(self.config_file) as f:
                data = json.load(f)
            print('Config file loaded successfully')
        except:
            raise NameError('Unable to open config file!!!')
        return data


class DataManager(ConfigParser):
    '''Manage dataset
    '''

    def __init__(self, config_file, placeholders):
        '''

        :param dataset: (features, targets) or features. Features and targets must be numpy arrays
        '''
        super(DataManager, self).__init__(config_file)
        self.path = self.config['data']['path']
        self.training_set = None
        self.testing_set = None
        self.num_train_data = None
        self.num_test_data = None
        self.batch_size = self.config['training']['batch_size']
        self.test_batch_size = self.config['training']['validation_batch_size']
        self.placeholders = placeholders
        self.shuffle = self.config['data']['shuffle']
        self.no_target = self.config['data']['no_target']
        self.augmentation = self.config['data']['augmentation']
        self.num_cached = self.config['data']['num_cached']

    def load_data(self):
        raise NotImplementedError

    def get_batches(self, epoch=None, num_epochs=None, stage='train', show_progress=True, *args):
        batches = self.generator(stage)
        if stage == 'train' and self.augmentation:
            batches = self.augment_minibatches(batches, *args)
        batches = self.generate_in_background(batches)
        if epoch is not None and num_epochs is not None:
            shape = self.num_train_data if stage == 'train' else self.num_test_data
            batch_size = self.batch_size if stage == 'train' else self.test_batch_size
            num_batches = shape // batch_size
            if show_progress:
                batches = self._progress(batches, desc='Epoch %d/%d, Batch ' % (epoch, num_epochs), total=num_batches)
        return batches

    def generate_in_background(self, generator):
        """
        Runs a generator in a background thread, caching up to `num_cached` items.
        """
        queue = Queue(maxsize=self.num_cached)
        sentinel = 'end'

        # define producer (putting items into queue)
        def producer():
            for item in generator:
                queue.put(item)
            queue.put(sentinel)

        # start producer (in a background thread)
        thread = threading.Thread(target=producer)
        thread.daemon = True
        thread.start()

        # run as consumer (read items from queue, in current thread)
        item = queue.get()
        while item is not 'end':
            yield item
            item = queue.get()

    @staticmethod
    def _progress(items, desc='', total=None, min_delay=0.1):
        """
        Returns a generator over `items`, printing the number and percentage of
        items processed and the estimated remaining processing time before yielding
        the next item. `total` gives the total number of items (required if `items`
        has no length), and `min_delay` gives the minimum time in seconds between
        subsequent prints. `desc` gives an optional prefix text (end with a space).
        """
        total = total or len(items)
        t_start = time.time()
        t_last = 0
        for n, item in enumerate(items):
            t_now = time.time()
            if t_now - t_last > min_delay:
                print("\r%s%d/%d (%6.2f%%)" % (
                        desc, n+1, total, n / float(total) * 100), end=" ")
                if n > 0:
                    t_done = t_now - t_start
                    t_total = t_done / n * total
                    print("(ETA: %d:%02d)" % divmod(t_total - t_done, 60), end=" ")
                sys.stdout.flush()
                t_last = t_now
            yield item
        t_total = time.time() - t_start
        print("\r%s%d/%d (100.00%%) (took %d:%02d)" % ((desc, total, total) + divmod(t_total, 60)))

    def augment_minibatches(self, minibatches, *args):
        raise NotImplementedError

    def update_input(self, data, *args):
        no_target = args[0] if args else self.no_target
        if isinstance(self.placeholders, (list, tuple)):
            assert len(self.placeholders) == len(data), 'Data has length %d but placeholders has length %d.' % \
                                                        (len(data), len(self.placeholders))
            for d, p in zip(data, self.placeholders):
                p.set_value(d, borrow=True)
        elif isinstance(self.placeholders, theano.gpuarray.type.GpuArraySharedVariable):
            x = data
            shape_x = self.placeholders.get_value().shape
            # if x.shape != shape_x:
            #     raise ValueError('Input of the shared variable must have the same shape with the shared variable')
            self.placeholders.set_value(x, borrow=True)
        else:
            raise TypeError('Expected theano shared or list/tuple type, got {}'.format(type(self.placeholders)))

    def generator(self, stage='train'):
        dataset = self.training_set if stage == 'train' else self.testing_set
        shape = self.num_train_data if stage == 'train' else self.num_test_data
        shuffle = self.shuffle if stage == 'train' else False
        num_batches = shape // self.batch_size
        if not self.no_target:
            x, y = dataset
            y = np.asarray(y)
        else:
            x = dataset

        if shuffle:
            index = np.arange(0, np.asarray(x).shape[0])
            np.random.shuffle(index)
            x = x[index]
            if not self.no_target:
                y = y[index]
        for i in range(num_batches):
            yield (x[i * self.batch_size:(i + 1) * self.batch_size], y[i * self.batch_size:(i + 1) * self.batch_size]) \
                if not self.no_target else x[i * self.batch_size:(i + 1) * self.batch_size]

    def shared_dataset(self, data_xy):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
        return shared_x, T.cast(shared_y, 'int32')


def crop_center(image, resize=256, crop=(224, 224)):
    h, w = image.shape[:2]
    scale = resize * 1.0 / min(h, w)
    if h < w:
        newh, neww = resize, int(scale * w + 0.5)
    else:
        newh, neww = int(scale * h + 0.5), resize
    image = misc.imresize(image, (newh, neww))

    orig_shape = image.shape
    h0 = int((orig_shape[0] - crop[0]) * 0.5)
    w0 = int((orig_shape[1] - crop[1]) * 0.5)
    image = image[h0:h0 + crop[0], w0:w0 + crop[1]]
    return image.astype('uint8')


def load_weights(weight_file, model):
    weights = np.load(weight_file)
    keys = sorted(weights.keys())
    num_weights = len(keys)
    j = 0
    for i, layer in enumerate(model):
        if j > num_weights - 1:
            break
        try:
            W = weights[keys[j]].transpose(3, 2, 0, 1) if len(weights[keys[j]].shape) == 4 else weights[keys[j]]
            if W.shape == layer.params[0].get_value().shape:
                layer.params[0].set_value(W)
                print('@ Layer %d %s %s' % (i, keys[j], np.shape(weights[keys[j]])))
            else:
                print('No compatible parameters for layer %d %s found. '
                      'Random initialization is used' % (i, keys[j]))
        except:
            W_converted = fully_connected_to_convolution(weights[keys[j]], layer.filter_shape) \
                if hasattr(layer, 'filter_shape') else None

            if W_converted is not None:
                if W_converted.shape == layer.filter_shape:
                    layer.W.set_value(W_converted)
                    print('@ Layer %d %s %s' % (i, keys[j], layer.filter_shape))
                else:
                    print('No compatible parameters for layer %d %s found. '
                          'Random initialization is used' % (i, keys[j]))
            else:
                print('No compatible parameters for layer %d %s found. '
                      'Random initialization is used' % (i, keys[j]))
        try:
            b = weights[keys[j+1]]
            if b.shape == layer.params[1].get_value().shape:
                layer.params[1].set_value(b)
                print('@ Layer %d %s %s' % (i, keys[j+1], np.shape(weights[keys[j+1]])))
            else:
                print('No compatible parameters for layer %d %s found. '
                      'Random initialization is used' % (i, keys[j+1]))
        except:
            print('No compatible parameters for layer %d %s found. '
                  'Random initialization is used' % (i, keys[j+1]))
        j += 2
    print('Loaded successfully!')


def fully_connected_to_convolution(weight, prev_layer_shape):
    weight = np.asarray(weight, 'float32')
    shape = weight.shape
    filter_size_square = shape[0] / prev_layer_shape[1]
    filter_size = np.sqrt(filter_size_square)
    if filter_size == int(filter_size):
        filter_size = int(filter_size)
        return np.reshape(weight, [-1, prev_layer_shape[1], filter_size, filter_size])
    else:
        return None


def maxout(input, **kwargs):
    size = kwargs.get('maxout_size', 4)
    maxout_out = None
    for i in range(size):
        t = input[:, i::size]
        if maxout_out is None:
            maxout_out = t
        else:
            maxout_out = T.maximum(maxout_out, t)
    return maxout_out


def lrelu(x, **kwargs):
    alpha = kwargs.get('alpha', 0.2)
    return T.nnet.relu(x, alpha=alpha)


def linear(x, **kwargs):
    return x


def ramp(x, **kwargs):
    left = T.switch(x < 0, 0, x)
    return T.switch(left > 1, 1, left)


def prelu(x, alpha):
    f1 = 0.5 * (1 + alpha)
    f2 = 0.5 * (1 - alpha)
    return f1 * x + f2 * abs(x)


def swish(x, **kwargs):
    return x * T.nnet.sigmoid(x)


def selu(x, **kwargs):
    lamb = kwargs.get('lambda', 1.0507)
    alpha = kwargs.get('alpha', 1.6733)
    return lamb * T.nnet.elu(x, alpha)


def inference(input, model):
    feed = input
    for layer in model:
        feed = layer(feed)
    return feed


def decrease_learning_rate(learning_rate, iter, epsilon_zero, epsilon_tau, tau):
    eps_zero = epsilon_zero if iter <= tau else 0.
    epsilon = (1 - float(iter)/tau) * eps_zero + float(iter)/tau * epsilon_tau
    learning_rate.set_value(np.cast['float32'](epsilon))


def rgb2gray(img):
    if img.ndim != 4:
        raise ValueError('Input images must have four dimensions, not %d' % img.ndim)
    return (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]).dimshuffle((0, 'x', 1, 2))


def rgb2ycbcr(img):
    if img.ndim != 4:
        raise ValueError('Input images must have four dimensions, not %d' % img.ndim)
    Y = 0. + .299 * img[:, 0] + .587 * img[:, 1] + .114 * img[:, 2]
    Cb = 128. - .169 * img[:, 0] - .331 * img[:, 1] + .5 * img[:, 2]
    Cr = 128. + .5 * img[:, 0] - .419 * img[:, 1] - .081 * img[:, 2]
    return T.concatenate((Y.dimshuffle((0, 'x', 1, 2)), Cb.dimshuffle((0, 'x', 1, 2)), Cr.dimshuffle((0, 'x', 1, 2))), 1)


def ycbcr2rgb(img):
    if img.ndim != 4:
        raise ValueError('Input images must have four dimensions, not %d' % img.ndim)
    R = img[:, 0] + 1.4 * (img[:, 2] - 128.)
    G = img[:, 0] - .343 * (img[:, 1] - 128.) - .711 * (img[:, 2] - 128.)
    B = img[:, 0] + 1.765 * (img[:, 1] - 128.)
    return T.concatenate((R.dimshuffle((0, 'x', 1, 2)), G.dimshuffle((0, 'x', 1, 2)), B.dimshuffle((0, 'x', 1, 2))), 1)


function = {'relu': lambda x, **kwargs: T.nnet.relu(x), 'sigmoid': lambda x, **kwargs: T.nnet.sigmoid(x),
            'tanh': lambda x, **kwargs: T.tanh(x), 'lrelu': lrelu, 'softmax': lambda x, **kwargs: T.nnet.softmax(x),
            'linear': linear, 'elu': lambda x, **kwargs: T.nnet.elu(x), 'ramp': ramp, 'maxout': maxout,
            'sin': lambda x, **kwargs: T.sin(x), 'cos': lambda x, **kwargs: T.cos(x), 'swish': swish, 'selu': selu}


if __name__ == '__main__':
    a = T.tensor4()
    b = ycbcr2rgb(rgb2ycbcr(a))
    f = theano.function([a], b)
    img = misc.imread('E:/Users/Duc/frame_interpolation/utils/v_ApplyEyeMakeup_g04_c05/1.jpg')
    img = np.transpose(img, (2, 0, 1))
    img = img[None]
    from matplotlib import pyplot as plt
    plt.figure()
    plt.imshow(np.uint8(np.transpose(f(img)[0], (1, 2, 0))))
    plt.figure()
    plt.imshow(np.transpose(img[0], (1, 2, 0)))
    plt.show()
