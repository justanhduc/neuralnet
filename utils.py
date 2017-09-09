from __future__ import print_function
import json
import numpy as np
import threading
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time
import sys

import layers
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


def generate_in_background(generator, num_cached=10):
    """
    Runs a generator in a background thread, caching up to `num_cached` items.
    """
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        item = queue.get()


def progress(items, desc='', total=None, min_delay=0.1):
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


def augment_minibatches(minibatches, flip=0.5, trans=4):
    """
    Randomly augments images by horizontal flipping with a probability of
    `flip` and random translation of up to `trans` pixels in both directions.
    """
    for inputs, targets in minibatches:
        batchsize, c, h, w = inputs.shape
        if flip:
            coins = np.random.rand(batchsize) < flip
            inputs = [inp[:, :, ::-1] if coin else inp
                      for inp, coin in zip(inputs, coins)]
            if not trans:
                inputs = np.asarray(inputs)
        outputs = inputs
        if trans:
            outputs = np.empty((batchsize, c, h, w), inputs[0].dtype)
            shifts = np.random.randint(-trans, trans, (batchsize, 2))
            for outp, inp, (x, y) in zip(outputs, inputs, shifts):
                if x > 0:
                    outp[:, :x] = 0
                    outp = outp[:, x:]
                    inp = inp[:, :-x]
                elif x < 0:
                    outp[:, x:] = 0
                    outp = outp[:, :x]
                    inp = inp[:, -x:]
                if y > 0:
                    outp[:, :, :y] = 0
                    outp = outp[:, :, y:]
                    inp = inp[:, :, :-y]
                elif y < 0:
                    outp[:, :, y:] = 0
                    outp = outp[:, :, :y]
                    inp = inp[:, :, -y:]
                outp[:] = inp
        yield outputs, targets


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


def maxout(input, maxout_size=4):
    maxout_out = None
    for i in xrange(maxout_size):
        t = input[:, i::maxout_size]
        if maxout_out is None:
            maxout_out = t
        else:
            maxout_out = T.maximum(maxout_out, t)
    return maxout_out


def lrelu(x, alpha=1e-2):
    return T.nnet.relu(x, alpha=alpha)


def linear(x):
    return x


def ramp(x):
    left = T.switch(x < 0, 0, x)
    return T.switch(left > 1, 1, left)


def prelu(x, alpha):
    f1 = 0.5 * (1 + alpha)
    f2 = 0.5 * (1 - alpha)
    return f1 * x + f2 * abs(x)


def update_input(data, shared_vars, no_response=False, **kwargs):
    if not no_response:
        x, y = data
        shape_y = kwargs.get('shape_y', shared_vars[1].get_value().shape)
        shape_x = kwargs.get('shape_x', shared_vars[0].get_value().shape)
        try:
            x = np.reshape(np.asarray(x, dtype=shared_vars[0].dtype), shape_x)
            y = np.reshape(np.asarray(y, dtype=shared_vars[1].dtype), shape_y)
        except ValueError:
            raise ValueError('Input of the shared variable must have the same shape with the shared variable')
        shared_vars[0].set_value(x, borrow=True)
        shared_vars[1].set_value(y, borrow=True)
    else:
        x = data
        shape_x = shared_vars.get_value().shape if 'shape_x' not in kwargs else kwargs['shape_x']
        try:
            x = np.reshape(np.asarray(x, dtype=shared_vars.dtype), shape_x)
        except ValueError:
            raise ValueError('Input of the shared variable must have the same shape with the shared variable')
        shared_vars.set_value(x, borrow=True)


def generator(data, batch_size, no_target=False):
    if not no_target:
        x, y = data
    else:
        x = data
    num_batch = np.asarray(x).shape[0] / batch_size
    for i in xrange(num_batch):
        yield (x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]) if not no_target \
            else x[i*batch_size:(i+1)*batch_size]


def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, 'int32')


def inference(input, model):
    feed = input
    for idx, layer in enumerate(model):
        feed = layer.get_output(feed.flatten(2)) if isinstance(layer, layers.FullyConnectedLayer) \
            else layer.get_output(feed)
    return feed


def decrease_learning_rate(learning_rate, iter, epsilon_zero, epsilon_tau, tau):
    eps_zero = epsilon_zero if iter <= tau else 0.
    epsilon = (1 - float(iter)/tau) * eps_zero + float(iter)/tau * epsilon_tau
    learning_rate.set_value(np.cast['float32'](epsilon))


def rgb2gray(img):
    if img.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d', img.ndim)
    return (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]).dimshuffle((0, 'x', 1, 2))

function = {'relu': T.nnet.relu, 'sigmoid': T.nnet.sigmoid, 'tanh': T.tanh, 'lrelu': lrelu,
            'softmax': T.nnet.softmax, 'linear': linear, 'elu': T.nnet.elu, 'ramp': ramp, 'maxout': maxout,
            'sin': T.sin, 'cos': T.cos}


