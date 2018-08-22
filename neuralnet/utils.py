import json
import sys
import threading
from queue import Queue
import time
import numpy as np
import theano
from theano import tensor as T
from scipy import misc
from functools import reduce
from itertools import cycle

__all__ = ['ConfigParser', 'DataManager']
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


class ConfigParser:
    def __init__(self, config_file=None, **kwargs):
        super(ConfigParser, self).__init__(**kwargs)
        self.config_file = config_file
        if config_file:
            self.config = self.load_configuration()
        else:
            self.config = None

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

    def __init__(self, config_file, placeholders, **kwargs):
        '''

        :param dataset: (features, targets) or features. Features and targets must be numpy arrays
        '''
        assert config_file or kwargs, 'Either config_file or keyword arguments must be provided.'

        super(DataManager, self).__init__(config_file)
        self.path = kwargs.get('path') if kwargs.get('path') else self.config['data']['path']
        self.batch_size = kwargs.get('batch_size') if kwargs.get('batch_size') else self.config['training'][
            'batch_size']
        self.n_epochs = kwargs.get('n_epochs') if kwargs.get('n_epochs') else self.config['training'][
            'n_epochs']
        self.shuffle = kwargs.get('shuffle') if kwargs.get('shuffle') else self.config['data'][
            'shuffle'] if config_file else False
        self.num_cached = kwargs.get('num_cached') if kwargs.get('num_cached') else self.config['data'][
            'num_cached'] if config_file else 10
        self.augmentation = kwargs.get('augmentation', None)
        self.dataset = None
        self.data_size = None
        self.placeholders = placeholders

    def load_data(self):
        raise NotImplementedError

    def preprocess(self, *args, **kwargs):
        """
        preprocess input tensors and return the processed tensors
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def augment_minibatches(self, minibatches, *args, **kwargs):
        raise NotImplementedError

    def get_batches(self, show_progress=False, *args, **kwargs):
        infinite = kwargs.pop('infinite', False)
        for epoch, _ in enumerate(iter(int, 1)) if infinite else enumerate(range(self.n_epochs)):
            batches = self.generator()
            if self.augmentation:
                batches = self.augment_minibatches(batches, *args, **kwargs)
            batches = self.generate_in_background(batches)
            if epoch:
                num_batches = self.data_size // self.batch_size
                if show_progress:
                    batches = _progress(batches, desc='Epoch %d/%d, Batch ' % (epoch, self.n_epochs), total=num_batches)
            for b in cycle(batches) if infinite else batches:
                self.update_input(b)
                yield

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

    def update_input(self, data):
        if isinstance(self.placeholders, (list, tuple)) and isinstance(data, (list, tuple)):
            assert len(self.placeholders) == len(data), 'Data has length %d but placeholders has length %d.' % \
                                                        (len(data), len(self.placeholders))
            for d, p in zip(data, self.placeholders):
                p.set_value(d, borrow=True)
        elif isinstance(self.placeholders, theano.gpuarray.type.GpuArraySharedVariable) and isinstance(data, np.ndarray):
            x = data
            shape_x = self.placeholders.get_value().shape
            if x.shape != shape_x:
                raise ValueError('Input of the shared variable must have the same shape with the shared variable')
            self.placeholders.set_value(x, borrow=True)
        else:
            raise TypeError(
                'placeholders should be a theano shared or list/tuple type and data should be a list, '
                'tuple or numpy ndarray, got {} and {}'.format(type(self.placeholders), type(data)))

    def generator(self):
        num_batches = self.data_size // self.batch_size
        dataset = self.dataset
        if self.shuffle:
            index = np.arange(0, self.data_size)
            np.random.shuffle(index)
            if isinstance(self.dataset, (list, tuple)):
                dataset = tuple([x[index] for x in self.dataset])
            elif isinstance(self.dataset, np.ndarray):
                dataset = self.dataset[index]
            else:
                raise TypeError('dataset should be a list, tuple or numpy ndarray, got %s.' % type(self.dataset))
        for i in range(num_batches):
            yield [data[i * self.batch_size:(i + 1) * self.batch_size] for data in dataset] \
                if isinstance(self.dataset, (list, tuple)) else dataset[i * self.batch_size:(i + 1) * self.batch_size]


def _progress(items, desc='', total=None, min_delay=0.1):
    """
    Returns a generator over `items`, printing the number and percentage of
    items processed and the estimated remaining processing time before yielding
    the next item. `total` gives the total number of items (required if `items`
    has no length), and `min_delay` gives the minimum time in seconds between
    subsequent prints. `desc` gives an optional prefix text (end with a space).
    """
    total = total or len(list(items))
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


def prep_image(fname, mean_bgr, color='bgr', resize=256):
    im = misc.imread(fname)

    # Resize
    h, w, _ = im.shape
    if h < w:
        new_sh = (resize, int(w * resize / h))
    else:
        new_sh = (int(h * resize / w), resize)
    im = misc.imresize(im, new_sh, interp='bicubic')

    # Crop center 224, 224
    h, w = new_sh
    im = im[h // 2 - 112:h // 2 + 112, w // 2 - 112:w // 2 + 112]

    rawim = np.copy(im).astype('uint8')

    im = im.astype('float32')
    if color == 'bgr':
        im = im[:, :, ::-1] - mean_bgr
    elif color == 'rgb':
        im = im - mean_bgr[:, :, ::-1]
    else:
        raise NotImplementedError
    return rawim, np.transpose(im[None], (0, 3, 1, 2))


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
            W_converted = convert_dense_weights_data_format(weights[keys[j]], layer.filter_shape) \
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


def convert_dense_weights_data_format(weights, previous_feature_map_shape, target_data_format='channels_first'):
    assert target_data_format in {'channels_last', 'channels_first'}
    kernel = np.array(weights, 'float32')
    for i in range(kernel.shape[1]):
        if target_data_format == 'channels_first':
            c, h, w = previous_feature_map_shape
            original_fm_shape = (h, w, c)
            ki = kernel[:, i].reshape(original_fm_shape)
            ki = np.transpose(ki, (2, 0, 1))  # last -> first
        else:
            h, w, c = previous_feature_map_shape
            original_fm_shape = (c, h, w)
            ki = kernel[:, i].reshape(original_fm_shape)
            ki = np.transpose(ki, (1, 2, 0))  # first -> last
        kernel[:, i] = np.reshape(ki, (np.prod(previous_feature_map_shape),))
    return kernel


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


def ramp(x, **kwargs):
    left = T.switch(x < 0, 0, x)
    return T.switch(left > 1, 1, left)


def prelu(x, **kwargs):
    alpha = kwargs.get('alpha', None)
    if alpha is None or not isinstance(alpha, T.sharedvar.ScalarSharedVariable):
        raise ValueError('alpha must be a shared variable, got %s' % type(alpha))
    return T.nnet.relu(x, alpha=alpha)


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


def linspace(start, stop, num):
    # Theano linspace. Behaves similar to np.linspace
    start = T.cast(start, theano.config.floatX)
    stop = T.cast(stop, theano.config.floatX)
    num = T.cast(num, theano.config.floatX)
    step = (stop-start)/(num-1)
    return T.arange(num, dtype=theano.config.floatX)*step+start


def _meshgrid(height, width):
    # This function is the grid generator from eq. (1) in reference [1].
    # It is equivalent to the following numpy code:
    #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
    #                         np.linspace(-1, 1, height))
    #  ones = np.ones(np.prod(x_t.shape))
    #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    # It is implemented in Theano instead to support symbolic grid sizes.
    # Note: If the image size is known at layer construction time, we could
    # compute the meshgrid offline in numpy instead of doing it dynamically
    # in Theano. However, it hardly affected performance when we tried.
    x_t = T.dot(T.ones((height, 1)), linspace(-1.0, 1.0, width).dimshuffle('x', 0))
    y_t = T.dot(linspace(-1.0, 1.0, height).dimshuffle(0, 'x'), T.ones((1, width)))

    x_t_flat = x_t.reshape((1, -1))
    y_t_flat = y_t.reshape((1, -1))
    ones = T.ones_like(x_t_flat)
    grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
    return grid


def interpolate_bilinear(im, x, y, out_shape=None, border_mode='nearest'):
    if im.ndim != 4:
        raise TypeError('im should be a 4D Tensor image, got %dD.' % im.ndim)

    out_shape = out_shape if out_shape else T.shape(im)[2:]
    x, y = x.flatten(), y.flatten()
    n, c, h, w = im.shape
    h_out, w_out = out_shape
    height_f = T.cast(h, theano.config.floatX)
    width_f = T.cast(w, theano.config.floatX)

    # scale coordinates from [-1, 1] to [0, width/height - 1]
    x = (x + 1) / 2 * (width_f - 1)
    y = (y + 1) / 2 * (height_f - 1)

    x0_f = T.floor(x)
    y0_f = T.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1

    if border_mode == 'nearest':
        x0 = T.clip(x0_f, 0, width_f - 1)
        x1 = T.clip(x1_f, 0, width_f - 1)
        y0 = T.clip(y0_f, 0, height_f - 1)
        y1 = T.clip(y1_f, 0, height_f - 1)
    elif border_mode == 'mirror':
        w = 2 * (width_f - 1)
        x0 = T.minimum(x0_f % w, -x0_f % w)
        x1 = T.minimum(x1_f % w, -x1_f % w)
        h = 2 * (height_f - 1)
        y0 = T.minimum(y0_f % h, -y0_f % h)
        y1 = T.minimum(y1_f % h, -y1_f % h)
    elif border_mode == 'wrap':
        x0 = T.mod(x0_f, width_f)
        x1 = T.mod(x1_f, width_f)
        y0 = T.mod(y0_f, height_f)
        y1 = T.mod(y1_f, height_f)
    else:
        raise ValueError("border_mode must be one of "
                         "'nearest', 'mirror', 'wrap'")
    x0, x1, y0, y1 = (T.cast(v, 'int64') for v in (x0, x1, y0, y1))

    base = T.arange(n) * w * h
    base = T.reshape(base, (-1, 1))
    base = T.tile(base, (1, h_out * w_out))
    base = base.flatten()

    base_y0 = base + y0 * w
    base_y1 = base + y1 * w
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    im_flat = T.reshape(im.dimshuffle((0, 2, 3, 1)), (-1, c))
    pixel_a = im_flat[idx_a]
    pixel_b = im_flat[idx_b]
    pixel_c = im_flat[idx_c]
    pixel_d = im_flat[idx_d]

    wa = ((x1_f - x) * (y1_f - y)).dimshuffle((0, 'x'))
    wb = ((x1_f - x) * (1. - (y1_f - y))).dimshuffle((0, 'x'))
    wc = ((1. - (x1_f - x)) * (y1_f - y)).dimshuffle((0, 'x'))
    wd = ((1. - (x1_f - x)) * (1. - (y1_f - y))).dimshuffle((0, 'x'))

    output = T.sum((wa*pixel_a, wb*pixel_b, wc*pixel_c, wd*pixel_d), axis=0)
    output = T.reshape(output, (n, h_out, w_out, c))
    return output.dimshuffle((0, 3, 1, 2))


def transform_affine(theta, input, downsample_factor=(1, 1), border_mode='nearest'):
    n, c, h, w = input.shape
    theta = T.reshape(theta, (-1, 2, 3))

    h_out = T.cast(h // downsample_factor[0], 'int64')
    w_out = T.cast(w // downsample_factor[1], 'int64')
    grid = _meshgrid(h_out, w_out)

    Tg = T.dot(theta, grid)
    xs = Tg[:, 0]
    ys = Tg[:, 1]
    return interpolate_bilinear(input, xs, ys, (h_out, w_out), border_mode)


def floatX(arr):
    """Converts data to a numpy array of dtype ``theano.config.floatX``.
    Parameters
    ----------
    arr : array_like
        The data to be converted.
    Returns
    -------
    numpy ndarray
        The input array in the ``floatX`` dtype configured for Theano.
        If `arr` is an ndarray of correct dtype, it is returned as is.
    """
    return np.asarray(arr, dtype=theano.config.floatX)


def unroll_scan(fn, sequences, outputs_info, non_sequences, n_steps, go_backwards=False):
    """
    Helper function to unroll for loops. Can be used to unroll theano.scan.
    The parameter names are identical to theano.scan, please refer to here
    for more information.
    Note that this function does not support the truncate_gradient
    setting from theano.scan.
    Parameters
    ----------
    fn : function
        Function that defines calculations at each step.
    sequences : TensorVariable or list of TensorVariables
        List of TensorVariable with sequence data. The function iterates
        over the first dimension of each TensorVariable.
    outputs_info : list of TensorVariables
        List of tensors specifying the initial values for each recurrent
        value.
    non_sequences: list of TensorVariables
        List of theano.shared variables that are used in the step function.
    n_steps: int
        Number of steps to unroll.
    go_backwards: bool
        If true the recursion starts at sequences[-1] and iterates
        backwards.
    Returns
    -------
    List of TensorVariables. Each element in the list gives the recurrent
    values at each time step.
    """
    if sequences is None:
        sequences = []

    if not isinstance(sequences, (list, tuple)):
        sequences = [sequences]

    if not isinstance(outputs_info, (list, tuple)):
        outputs_info = [outputs_info]

    # When backwards reverse the recursion direction
    counter = range(n_steps)
    if go_backwards:
        counter = counter[::-1]

    output = []
    prev_vals = list(outputs_info)
    for i in counter:
        step_input = [s[i] for s in sequences] + prev_vals + non_sequences
        out_ = fn(*step_input)
        # The returned values from step can be either a TensorVariable,
        # a list, or a tuple.  Below, we force it to always be a list.
        if isinstance(out_, T.TensorVariable):
            out_ = [out_]
        if isinstance(out_, tuple):
            out_ = list(out_)
        output.append(out_)
        prev_vals = output[-1]

    # iterate over each scan output and convert it to same format as scan:
    # [[output11, output12,...output1n],
    # [output21, output22,...output2n],...]
    output_scan = []
    for i in range(len(output[0])):
        l = map(lambda x: x[i], output)
        output_scan.append(T.stack(*l))
    return output_scan if len(output_scan) > 1 else output_scan[0]


def lagrange_interpolation(x, y, u, order):
    r = range(order+1)
    a = [y[i] / reduce(lambda c, b: c*b, [x[i] - x[j] for j in r if j != i]) for i in r]
    out = T.sum([a[i] * reduce(lambda c, b: c*b, [u - x[j] for j in r if j != i]) for i in r], 0)
    return out


def point_op(image, lut, origin, increment, *args):
    h, w = image.shape
    im = image.flatten()
    lutsize = lut.shape[0] - 2

    pos = (im - origin) / increment
    index = pos.astype('int64')
    index = T.set_subtensor(index[index < 0], 0)
    index = T.set_subtensor(index[index > lutsize], lutsize)
    res = lut[index] + (lut[index+1] - lut[index]) * (pos - index.astype('float32'))
    return T.reshape(res, (h, w)).astype('float32')


def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
    assert kernel_type in ['lanczos', 'gauss', 'box']

    # factor  = float(factor)
    if phase == 0.5 and kernel_type != 'box':
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])

    if kernel_type == 'box':
        assert phase == 0.5, 'Box filter is always half-phased'
        kernel[:] = 1. / (kernel_width * kernel_width)

    elif kernel_type == 'gauss':
        assert sigma, 'sigma is not specified'
        assert phase != 0.5, 'phase 1/2 for gauss not implemented'

        center = (kernel_width + 1.) / 2.
        sigma_sq = sigma * sigma

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center) / 2.
                dj = (j - center) / 2.
                kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj) / (2 * sigma_sq))
                kernel[i - 1][j - 1] = kernel[i - 1][j - 1] / (2. * np.pi * sigma_sq)
    elif kernel_type == 'lanczos':
        assert support, 'support is not specified'
        center = (kernel_width + 1) / 2.

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):

                if phase == 0.5:
                    di = abs(i + 0.5 - center) / factor
                    dj = abs(j + 0.5 - center) / factor
                else:
                    di = abs(i - center) / factor
                    dj = abs(j - center) / factor

                pi_sq = np.pi * np.pi

                val = 1
                if di != 0:
                    val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                    val = val / (pi_sq * di * di)

                if dj != 0:
                    val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                    val = val / (pi_sq * dj * dj)

                kernel[i - 1][j - 1] = val
    else:
        assert False, 'Wrong method name.'

    kernel /= kernel.sum()
    return floatX(kernel)


def replication_pad2d(input, padding):
    """
    Mimicking torch.nn.ReplicationPad2d(padding)

    :param padding: (int, tuple) – the size of the padding. If is int, uses the same padding in all boundaries.
    If a 4-tuple, uses (paddingLeft, paddingRight, paddingTop, paddingBottom)
    :return:
    """

    if not isinstance(padding, (int, list, tuple)):
        raise TypeError('padding must be an int, a list or a tuple. Got %s.' % type(padding))

    if isinstance(padding, int):
        padding = (padding, ) * 4

    output = input
    for axis, i in enumerate(padding):
        for j in range(i):
            if axis == 0:
                output = T.concatenate((output[:, :, :, 0:1], output), 3)
            elif axis == 1:
                output = T.concatenate((output, output[:, :, :, -1:]), 3)
            elif axis == 2:
                output = T.concatenate((output[:, :, 0:1], output), 2)
            elif axis == 3:
                output = T.concatenate((output, output[:, :, -1:]), 2)
            else:
                raise ValueError('padding must have 4 elements. Received %d.' % len(padding))

    return output


def unpool(input, shape):
    assert input.ndim == 4, 'input must be a 4D tensor.'
    assert isinstance(shape, (list, tuple, int)), 'shape must be a list, tuple, or int, got %s' % type(shape)

    shape = tuple(shape) if isinstance(shape, (list, tuple)) else (shape, shape)
    output = T.repeat(input, shape[0], 2)
    output = T.repeat(output, shape[1], 3)
    return output


def batch_set_value(tuples):
    for x, value in tuples:
        if x.get_value().shape != value.shape:
            raise ValueError('Dimension mismatch for %s.' % x.name)
        x.set_value(np.asarray(value, dtype=x.dtype))


def make_tensor_kernel_from_numpy(kern_shape, numpy_kernel, type='each'):
    if type not in ('each', 'all'):
        raise ValueError('type must be \'each\' or \'all\'.')
    if not isinstance(kern_shape, (list, tuple)):
        raise ValueError('kern_shape must be a list or tuple.')
    if len(kern_shape) != 2:
        raise ValueError('kern_shape must be a tuple of (out, in) shape.')
    if numpy_kernel.ndim != 2:
        raise ValueError('numpy_kernel must be a 2D filter.')

    if type == 'each':
        if kern_shape[0] != kern_shape[1]:
            raise ValueError('kern_shape values must be the same for \'each\'.')
        kern = T.zeros(tuple(kern_shape) + numpy_kernel.shape, 'float32')
        for ii in range(kern_shape[0]):
            kern = T.set_subtensor(kern[ii, ii], numpy_kernel)
    else:
        kern = T.tile(numpy_kernel, tuple(kern_shape)+(1, 1), 4)
    return kern


def gaussian2(size, sigma):
    """Returns a normalized circularly symmetric 2D gauss kernel array

    f(x,y) = A.e^{-(x^2/2*sigma^2 + y^2/2*sigma^2)} where

    A = 1/(2*pi*sigma^2)

    as define by Wolfram Mathworld
    http://mathworld.wolfram.com/GaussianFunction.html
    """
    A = 1. / (2. * np.pi * sigma ** 2)
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = A * np.exp(-((x ** 2 / (2. * sigma ** 2)) + (y ** 2 / (2. * sigma ** 2))))
    return np.float32(g)


def laplacian_of_gaussian_kernel(size, sigma):
    """Returns a normalized circularly symmetric 2D gauss kernel array

    f(x,y) = A.e^{-(x^2/2*sigma^2 + y^2/2*sigma^2)} where

    A = 1/(2*pi*sigma^2)

    as define by Wolfram Mathworld
    http://mathworld.wolfram.com/GaussianFunction.html
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = 1 / (2*np.pi*sigma**4) * ((x**2 + y**2 - 2*sigma**2) / sigma**2) * np.exp(-(x**2 + y**2) / (2*sigma**2))
    return np.float32(g)


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = T.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = T.exp(-((T.cast(x, 'float32') ** 2 + T.cast(y, 'float32') ** 2) / (2.0 * sigma ** 2)))
    return g / T.sum(g)


def difference_of_gaussian(x, depth=3, size=21, sigma1=1, sigma2=1.6):
    kern1 = gaussian2(size, sigma1)
    kern2 = gaussian2(size, sigma2)
    kern1 = make_tensor_kernel_from_numpy((depth, depth), kern1)
    kern2 = make_tensor_kernel_from_numpy((depth, depth), kern2)
    x1 = T.nnet.conv2d(x, kern1, border_mode='half')
    x2 = T.nnet.conv2d(x, kern2, border_mode='half')
    return x2 - x1


def pad(img, mul):
    h_new = int(np.ceil(img.shape[0] / mul[0])) * mul[0]
    w_new = int(np.ceil(img.shape[1] / mul[1])) * mul[1]

    new_shape = list(img.shape)
    new_shape[0], new_shape[1] = h_new, w_new
    out = np.zeros(new_shape, img.dtype)
    start_x = (w_new - img.shape[1]) // 2
    start_y = (h_new - img.shape[0]) // 2

    out[start_y:start_y + img.shape[0], start_x:start_x + img.shape[1]] = img
    return out


def unpad(img, old_shape):
    start_x = (img.shape[1] - old_shape[1]) // 2
    start_y = (img.shape[0] - old_shape[0]) // 2
    out = img[start_y:start_y + old_shape[0], start_x:start_x + old_shape[1]]
    return out


def depth_to_space(x, upscale_factor):
    n, c, h, w = x.shape
    oc = c // (upscale_factor ** 2)
    oh = h * upscale_factor
    ow = w * upscale_factor

    z = T.reshape(x, (n, oc, upscale_factor, upscale_factor, h, w))
    z = z.dimshuffle(0, 1, 4, 2, 5, 3)
    return T.reshape(z, (n, oc, oh, ow))


function = {'relu': lambda x, **kwargs: T.nnet.relu(x), 'sigmoid': lambda x, **kwargs: T.nnet.sigmoid(x),
            'tanh': lambda x, **kwargs: T.tanh(x), 'lrelu': lrelu, 'softmax': lambda x, **kwargs: T.nnet.softmax(x),
            'linear': lambda x, **kwargs: x, 'elu': lambda x, **kwargs: T.nnet.elu(x), 'ramp': ramp, 'maxout': maxout,
            'sin': lambda x, **kwargs: T.sin(x), 'cos': lambda x, **kwargs: T.cos(x), 'swish': swish, 'selu': selu,
            'prelu': prelu}


if __name__ == '__main__':
    x = T.tensor4()
    y = unpool(x, (2, 2))
    f = theano.function([x], y)
    print(f(np.random.rand(2, 64, 120, 160).astype('float32')).shape)
