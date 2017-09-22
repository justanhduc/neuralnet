'''
read some common datasets
'''

import cPickle as pkl
import os
import numpy as np
import tqdm
import sys


def read_cifar10(cifar_folder, new_size=(32, 32), validation=0.2, shuffle=False):
    print 'PREPARING CIFAR10 DATA...'
    print 'Data shall be resized to', new_size
    print 'Percentage of validation data: %.2f%%' % (validation * 100)
    print 'Shuffle data:', shuffle
    file_list = os.listdir(cifar_folder)
    train_data, train_label, = [], []
    for file in file_list:
        if 'data_batch' in file:
            with open('%s/%s' % (cifar_folder, file), 'rb') as fo:
                dict = pkl.load(fo)
                train_data.append(dict.values()[0])
                train_label.append(np.array(dict.values()[1], 'int8'))
        elif 'test_batch' in file:
            with open('%s/%s' % (cifar_folder, file), 'rb') as fo:
                dict = pkl.load(fo)
                test_data = dict.values()[0]
                test_label = np.array(dict.values()[1], 'int8')
        else:
            continue
    print 'Processing training data...'
    train_data = reshape_cifar(np.concatenate(train_data, 0), new_size) / 255.
    train_label = np.concatenate(train_label, 0)
    print 'Processing test data...'
    test_data = reshape_cifar(test_data, new_size) / 255.
    if shuffle:
        index = np.array(range(0, train_data.shape[0]))
        np.random.shuffle(index)
        train_data = train_data[index]
        train_label = train_label[index]
    print 'DATA PREPARED!'
    return (train_data[:int((1 - validation) * train_data.shape[0])], train_label[:int((1 - validation) * train_data.shape[0])]), \
           (train_data[int((1 - validation) * train_data.shape[0]):], train_label[int((1 - validation) * train_data.shape[0]):]), \
           (test_data, test_label)


def reshape_cifar(cifar_numpy_array, new_size=(32, 32)):
    from scipy import misc
    data = []
    for i in tqdm.tqdm(xrange(cifar_numpy_array.shape[0]), unit='images'):
        r = misc.imresize(np.reshape(cifar_numpy_array[i, :1024], (32, 32)), new_size)
        g = misc.imresize(np.reshape(cifar_numpy_array[i, 1024: 2048], (32, 32)), new_size)
        b = misc.imresize(np.reshape(cifar_numpy_array[i, 2048:], (32, 32)), new_size)
        data.append(np.dstack((r, g, b)))
    return np.transpose(np.array(data, dtype='float32'), (0, 3, 1, 2))


def download_dataset(path, source='https://www.cs.toronto.edu/~kriz/'
                                  'cifar-10-python.tar.gz'):
    """
    Downloads and extracts the dataset, if needed.
    """
    files = ['data_batch_%d' % (i + 1) for i in range(5)] + ['test_batch']
    for fn in files:
        if not os.path.exists(os.path.join(path, 'cifar-10-batches-py', fn)):
            break  # at least one file is missing
    else:
        return  # dataset is already complete

    print("Downloading and extracting %s into %s..." % (source, path))
    if sys.version_info[0] == 2:
        from urllib import urlopen
    else:
        from urllib.request import urlopen
    import tarfile
    if not os.path.exists(path):
        os.makedirs(path)
    u = urlopen(source)
    with tarfile.open(fileobj=u, mode='r|gz') as f:
        f.extractall(path=path)
    u.close()


def load_dataset(path, new_shape=None):
    download_dataset(path)

    # training data
    data = [np.load(os.path.join(path, 'cifar-10-batches-py',
                                 'data_batch_%d' % (i + 1))) for i in range(5)]
    X_train = np.vstack([d['data'] for d in data])
    y_train = np.hstack([np.asarray(d['labels'], np.int8) for d in data])

    # test data
    data = np.load(os.path.join(path, 'cifar-10-batches-py', 'test_batch'))
    X_test = data['data']
    y_test = np.asarray(data['labels'], np.int8)

    # reshape
    X_train = X_train.reshape(-1, 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)

    if new_shape is not None:
        X_train = reshape_cifar(X_train)
        X_test = reshape_cifar(X_test)

    # normalize
    # try:
    #     mean_std = np.load(os.path.join(path, 'cifar-10-mean_std.npz'))
    #     mean = mean_std['mean']
    #     std = mean_std['std']
    # except IOError:
    #     mean = X_train.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    #     std = X_train.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    #     np.savez(os.path.join(path, 'cifar-10-mean_std.npz'),
    #              mean=mean, std=std)
    # X_train = (X_train - mean) / std
    # X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Generates one epoch of batches of inputs and targets, optionally shuffled.
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

if __name__ == '__main__':
    sets = load_dataset('C:\Users\just.anhduc\Downloads')


    def generator(data, batch_size, no_target=False):
        if not no_target:
            x, y = data
        else:
            x = data
        num_batch = np.asarray(x).shape[0] / batch_size
        for i in xrange(num_batch):
            yield (x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]) if not no_target \
                else x[i * batch_size:(i + 1) * batch_size]


    def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
        """
        Generates one epoch of batches of inputs and targets, optionally shuffled.
        """
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    import time
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
                print"\r%s%d/%d (%6.2f%%) " % (
                    desc, n + 1, total, n / float(total) * 100)
                if n > 0:
                    t_done = t_now - t_start
                    t_total = t_done / n * total
                    print"(ETA: %d:%02d) " % divmod(t_total - t_done, 60)
                    sys.stdout.flush()
                    t_last = t_now
                yield item
            t_total = time.time() - t_start
            print "\r%s%d/%d (100.00%%) (took %d:%02d)" % ((desc, total, total) + divmod(t_total, 60))


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

    batch = generator(sets[0], 100)
    batch = augment_minibatches(batch)
    batch = generate_in_background(batch)
    batch = progress(
        batch, desc='Epoch %d/%d, Batch ' % (1 + 1, 1),
        total=5000)
    for b in batch:
        print b[0].shape

    # read_cifar10('C:\Users\just.anhduc\Downloads\cifar-10-batches-py', (100, 100), 0.2, True)
