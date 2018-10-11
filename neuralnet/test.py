import theano
from theano import tensor as T
import numpy as np

import neuralnet as nn


def assert_allclose(x, y):
    assert np.all(np.isclose(x, y))


def test_spearman():
    size = (64,)

    pred = T.vector('pred')
    gt = T.vector('gt')
    corr = nn.spearman(pred, gt)
    func = theano.function([pred, gt], corr)

    gt = np.random.normal(size=size).astype(theano.config.floatX)
    pred = np.random.uniform(size=size).astype(theano.config.floatX)
    c = func(pred, gt)

    from scipy import stats
    c_ref, _ = stats.spearmanr(pred.flatten(), gt.flatten())
    assert_allclose(c_ref, c)


def test_pearsonr():
    size = (64, 1000)

    pred = T.matrix('pred')
    gt = T.matrix('gt')
    corr = nn.pearson_correlation(pred, gt)
    func = theano.function([pred, gt], corr)

    gt = np.random.normal(size=size).astype(theano.config.floatX)
    pred = np.random.uniform(size=size).astype(theano.config.floatX)
    c = func(pred, gt)

    from scipy import stats
    c_ref, _ = stats.pearsonr(pred.flatten(), gt.flatten())
    assert_allclose(c_ref, c)


def test_monitor_hist():
    valid_freq = 2
    size = (64, 32, 3, 3)
    n_iters = 20

    import os
    import shutil
    if os.path.exists('results'):
        shutil.rmtree('results')
    mon = nn.Monitor(valid_freq=valid_freq)
    filter = np.random.uniform(-1, 1, size)
    for i in range(n_iters):
        with mon:
            mon.hist('filter%d' % i, filter + i / 10. * np.random.normal(.5, size=size))
    mon.flush()


def test_model_zoo_vgg16():
    top = 1
    root = 'test_files/'
    weight_file = root + 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    imagenet_classes_file = root + 'imagenet_classes.txt'
    image_list = [
        'ILSVRC2012_val_00000001.JPEG',
        'ILSVRC2012_val_00000002.JPEG',
        'ILSVRC2012_val_00000003.JPEG',
        'ILSVRC2012_val_00000004.JPEG',
        'ILSVRC2012_val_00000005.JPEG'
    ]

    net = nn.model_zoo.VGG16((None, 3, 224, 224), True)
    net.load_params(weight_file)

    X = T.tensor4('input')
    test = theano.function([X], net(X))

    with open(imagenet_classes_file, 'r') as f:
        classes = [s.strip() for s in f.readlines()]
    mean_bgr = np.array([103.939, 116.779, 123.68], dtype='float32')[None, None, :]

    results = []
    for fname in image_list:
        print(fname)
        rawim, im = nn.utils.prep_image(root+fname, mean_bgr, 'rgb')
        prob = test(im)[0]
        res = sorted(zip(classes, prob), key=lambda t: t[1], reverse=True)[:top]
        for c, p in res:
            print('  ', c, p)
            results.append(c)
        print('\n')

    sample_res = ['water snake', 'ski', 'Shetland sheepdog, Shetland sheep dog, Shetland', 'soup bowl', 'bassinet']
    assert results == sample_res


def test_model_zoo_vgg19():
    top = 1
    root = 'test_files/'
    weight_file = root + 'vgg19_weights.h5'
    imagenet_classes_file = root + 'imagenet_classes.txt'
    image_list = [
        'ILSVRC2012_val_00000001.JPEG',
        'ILSVRC2012_val_00000002.JPEG',
        'ILSVRC2012_val_00000003.JPEG',
        'ILSVRC2012_val_00000004.JPEG',
        'ILSVRC2012_val_00000005.JPEG'
    ]

    net = nn.model_zoo.VGG19((None, 3, 224, 224), True)
    net.load_params(weight_file)

    X = T.tensor4('input')
    test = theano.function([X], net(X))

    with open(imagenet_classes_file, 'r') as f:
        classes = [s.strip() for s in f.readlines()]
    mean_bgr = np.array([103.939, 116.779, 123.68], dtype='float32')[None, None, :]

    results = []
    for fname in image_list:
        print(fname)
        rawim, im = nn.utils.prep_image(root+fname, mean_bgr, 'rgb')
        prob = test(im)[0]
        res = sorted(zip(classes, prob), key=lambda t: t[1], reverse=True)[:top]
        for c, p in res:
            print('  ', c, p)
            results.append(c)
        print('\n')

    sample_res = ['sea snake', 'ski', 'Shetland sheepdog, Shetland sheep dog, Shetland', 'soup bowl', 'bassinet']
    assert results == sample_res


def test_replication_padding():
    padding = 50

    input = T.tensor4('input')
    out = nn.utils.replication_pad(input, padding)
    func = theano.function([input], out)

    from scipy import misc
    img = misc.imread('test_files/lena_small.png').astype(theano.config.floatX) / 255.
    img = np.transpose(img, (2, 0, 1))[None]
    out = func(img)
    out = np.transpose(out[0], (1, 2, 0))
    misc.imsave('test_files/lena_small_rep_padded.jpg', out)


def test_reflection_padding():
    padding = 50

    input = T.tensor4('input')
    out = nn.utils.reflect_pad(input, padding, 2)
    func = theano.function([input], out)

    from scipy import misc
    img = misc.imread('test_files/lena_small.png').astype(theano.config.floatX) / 255.
    img = np.transpose(img, (2, 0, 1))[None]
    out = func(img)
    out = np.transpose(out[0], (1, 2, 0))
    misc.imsave('test_files/lena_small_ref_padded.jpg', out)


def test_spatial_transformer():
    input_shape = (None, 3, 220, 220)
    down_factor = 1
    dnn = False

    input = T.tensor4('input')
    theta = T.matrix('theta')
    sptf = nn.SpatialTransformerLayer(input_shape, down_factor, dnn=dnn)
    out = sptf((input, theta))
    func = theano.function([input, theta], out)

    from scipy import misc
    img = misc.imread('test_files/lena_small.png').astype(theano.config.floatX) / 255.
    img = np.transpose(img, (2, 0, 1))[None]
    theta = np.array([[np.cos(np.deg2rad(45)), -np.sin(np.deg2rad(45)), 0],
                      [np.sin(np.deg2rad(45)), np.cos(np.deg2rad(45)), 0]], theano.config.floatX)
    out = func(img, theta)
    out = np.transpose(out[0], (1, 2, 0))
    misc.imsave('test_files/lena_small_rotated.jpg', out)


def test_lstm_dnn():
    input_dim = 32
    hidden_dim = 16
    batch_size = 2
    depth = 3
    timesteps = 5

    X = T.tensor3('X')
    lstm = nn.RNNBlockDNN((None, None, input_dim), hidden_dim, depth, 'lstm')
    out = lstm(X)
    func = theano.function([X], out)

    x_val = np.random.random((timesteps, batch_size, input_dim)).astype(theano.config.floatX)
    out = func(x_val)
    print(out.shape)
    print(out)
