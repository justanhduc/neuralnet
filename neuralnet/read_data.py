'''
read some common datasets
'''

import pickle as pkl
import os
import numpy as np
import tqdm
import sys


def read_cifar10(cifar_folder, new_size=(32, 32), validation=0.2, shuffle=False):
    print('PREPARING CIFAR10 DATA...')
    print('Data shall be resized to', new_size)
    print('Percentage of validation data: %.2f%%' % (validation * 100))
    print('Shuffle data:', shuffle)
    file_list = os.listdir(cifar_folder)
    train_data, train_label, = [], []
    for file in file_list:
        if 'data_batch' in file:
            with open('%s/%s' % (cifar_folder, file), 'rb') as fo:
                dict = pkl.load(fo)
                train_data.append(list(dict.values())[0])
                train_label.append(np.array(list(dict.values())[1], 'int8'))
        elif 'test_batch' in file:
            with open('%s/%s' % (cifar_folder, file), 'rb') as fo:
                dict = pkl.load(fo)
                test_data = list(dict.values())[0]
                test_label = np.array(list(dict.values())[1], 'int8')
        else:
            continue
    print('Processing training data...')
    train_data = reshape_cifar(np.concatenate(train_data, 0), new_size) / 255.
    train_label = np.concatenate(train_label, 0)
    print('Processing test data...')
    test_data = reshape_cifar(test_data, new_size) / 255.
    if shuffle:
        index = np.array(list(range(0, train_data.shape[0])))
        np.random.shuffle(index)
        train_data = train_data[index]
        train_label = train_label[index]
    print('DATA PREPARED!')
    return (train_data[:int((1 - validation) * train_data.shape[0])], train_label[:int((1 - validation) * train_data.shape[0])]), \
           (train_data[int((1 - validation) * train_data.shape[0]):], train_label[int((1 - validation) * train_data.shape[0]):]), \
           (test_data, test_label)


def reshape_cifar(cifar_numpy_array, new_size=(32, 32)):
    from scipy import misc
    data = []
    for i in tqdm.tqdm(range(cifar_numpy_array.shape[0]), unit='images'):
        r = misc.imresize(np.reshape(cifar_numpy_array[i, 0], (32, 32)), new_size)
        g = misc.imresize(np.reshape(cifar_numpy_array[i, 1], (32, 32)), new_size)
        b = misc.imresize(np.reshape(cifar_numpy_array[i, 2], (32, 32)), new_size)
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

    print(("Downloading and extracting %s into %s..." % (source, path)))
    if sys.version_info[0] == 2:
        from urllib.request import urlopen
    else:
        from urllib.request import urlopen
    import tarfile
    if not os.path.exists(path):
        os.makedirs(path)
    u = urlopen(source)
    with tarfile.open(fileobj=u, mode='r|gz') as f:
        f.extractall(path=path)
    u.close()


def load_dataset(path, normalize=False, new_shape=None):
    download_dataset(path)

    # training data
    data = [pkl.load(open(os.path.join(path, 'cifar-10-batches-py',
                             'data_batch_%d' % (i + 1)), 'rb'), encoding='latin-1') for i in range(5)]
    X_train = np.vstack([d['data'] for d in data])
    y_train = np.hstack([np.asarray(d['labels'], np.int8) for d in data])

    # test data
    data = pkl.load(open(os.path.join(path, 'cifar-10-batches-py', 'test_batch'), 'rb'), encoding='latin-1')
    X_test = data['data']
    y_test = np.asarray(data['labels'], np.int8)

    # reshape
    X_train = X_train.reshape(-1, 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)

    if new_shape:
        X_train = reshape_cifar(X_train)
        X_test = reshape_cifar(X_test)

    if normalize:
        # normalize
        try:
            mean_std = np.load(os.path.join(path, 'cifar-10-mean_std.npz'))
            mean = mean_std['mean']
            std = mean_std['std']
        except IOError:
            mean = X_train.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
            std = X_train.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
            np.savez(os.path.join(path, 'cifar-10-mean_std.npz'),
                     mean=mean, std=std)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

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


def load_mnist(gzip_file):
    import gzip
    f = gzip.open(gzip_file, 'rb')
    train_set, valid_set, test_set = pkl.load(f)
    f.close()
    return train_set, valid_set, test_set


def load_data(path):
    print('Loading data...')
    data = np.load(path)
    X_train, y_train = data['x_train'], np.argmax(data['y_train'], axis=-1)
    X_valid, y_valid = data['x_valid'], np.argmax(data['y_valid'], axis=-1)
    X_test, y_test = data['x_test'], np.argmax(data['y_test'], axis=-1)

    # reshape for convolutions
    X_train = X_train.reshape((X_train.shape[0], 1, 60, 60))
    X_valid = X_valid.reshape((X_valid.shape[0], 1, 60, 60))
    X_test = X_test.reshape((X_test.shape[0], 1, 60, 60))

    return dict(
        X_train=np.float32(X_train),
        y_train=y_train.astype('int32'),
        X_valid=np.float32(X_valid),
        y_valid=y_valid.astype('int32'),
        X_test=np.float32(X_test),
        y_test=y_test.astype('int32'),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_height=X_train.shape[2],
        input_width=X_train.shape[3],
        output_dim=10)


if __name__ == '__main__':
    dataset = load_mnist('C:\\Users\just.anhduc\PycharmProjects\ml_assignment/mnist.pkl.gz')
    pass
