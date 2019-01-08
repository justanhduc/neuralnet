import numpy as np
import theano
from theano import tensor as T

import neuralnet as nn


def assert_allclose(x, y):
    assert np.all(np.isclose(x, y))


def test_chamfer_distance():
    m = 10
    n1 = 6000
    n2 = 5000
    d = 3

    a = T.tensor3('a')
    b = T.tensor3('b')
    dist = nn.chamfer_distance(a, b)

    a_ = np.ones((m, n1, d), dtype=theano.config.floatX)
    b_ = np.zeros((m, n2, d), dtype=theano.config.floatX)
    dist.eval({a: a_, b: b_})

    import time
    start = time.time()
    res = dist.eval({a: a_, b: b_})
    print('Took %fs.' % (time.time() - start))
    assert_allclose(res, 330000.)


def test_frac_bilinear_upsampling():
    frac_ratio = ((5, 2), (11, 5))

    X = T.tensor4('input')
    Y = nn.utils.frac_bilinear_upsampling(X, frac_ratio)
    f = nn.function([X], Y)

    from imageio import imread, imwrite
    x = imread('test_files/lena_small.png').astype('float32') / 255.
    x = np.transpose(x, (2, 0, 1))[None]
    y = f(x)
    out = np.array(np.transpose(y[0], (1, 2, 0)))
    imwrite('test_files/lena_small_frac_bi_up.jpg', out)


def test_meshgrid():
    height = 10.
    width = 100.

    h = T.scalar()
    w = T.scalar()
    x, y = nn.utils.linspace(0, w, w), nn.utils.linspace(0, h, h)
    X, Y = nn.utils.meshgrid(x, y)
    f = nn.function([w, h], [X, Y])

    X_th, Y_th = f(width, height)

    x, y = np.linspace(0, width, width), np.linspace(0, height, height)
    X_np, Y_np = np.meshgrid(x, y)

    assert_allclose(X_th, X_np)
    assert_allclose(Y_th, Y_np)


def test_rfft2():
    X = T.tensor4('input')
    Y = nn.utils.rfft2(X)
    Y_i = nn.utils.irfft2(Y)
    rfft2 = nn.function([X], [Y, Y_i])

    from imageio import imread
    x = imread('test_files/lena_small.png').astype('float32') / 255.
    x = np.transpose(x, (2, 0, 1))
    x_fft_np = np.fft.rfft2(x)

    x = x[None]
    x_fft_th, x_ifft_th = rfft2(x)
    x_fft_th = x_fft_th[0]

    assert_allclose(np.real(x_fft_np), x_fft_th[..., 0])
    assert_allclose(np.imag(x_fft_np), x_fft_th[..., 1])
    np.allclose(x_ifft_th, x, 1e-4)


def test_monitor_dump():
    shape = (128, 256, 3, 3)
    n_iters = 10
    print_freq = 2

    a = np.random.rand(*shape).astype('float32')

    import os
    import shutil
    if os.path.exists('results'):
        shutil.rmtree('results')
    mon = nn.Monitor(print_freq=print_freq)
    res = []
    for i in range(n_iters):
        a += 1.
        res.append(np.copy(a))
        with mon:
            mon.dump(a, 'foo.npy', 3)
    mon.flush()

    mon = nn.Monitor(current_folder='results/my_model/run1')
    loaded_a = [mon.load('foo.npy', version) for version in (6, 8, 10)]
    assert_allclose(np.array(res)[[6, 8, 9]], loaded_a)


def test_downsample():
    shape = (1, 3, 220, 220)
    kernel_type = 'gauss1sq2'

    x = T.tensor4('input')
    downsample = nn.DownsamplingLayer(shape, 4, kernel_type=kernel_type)
    y = downsample(x)
    f = nn.function([x], y)

    from imageio import imwrite, imread
    x = imread('test_files/lena_small.png').astype('float32') / 255.
    x = np.transpose(x[None], (0, 3, 1, 2))
    out = f(x)
    out = np.transpose(out[0], (1, 2, 0))
    imwrite('test_files/lena_small_downsampled.jpg', out)


def test_partial_conv_based_padding():
    shape = (1, 3, 5, 5)
    num_filters = 2
    filter_size = 3

    x = T.ones(shape)
    conv = nn.Conv2DLayer(shape, num_filters, filter_size, border_mode='partial')
    y = conv(x)
    f = nn.function([], y)
    print(f())
    print(conv.update_mask.eval())
    print(conv.mask_ratio)


def test_monitor_plot():
    valid_freq = 2
    n_iters = 20

    import os
    import shutil
    if os.path.exists('results'):
        shutil.rmtree('results')
    mon = nn.Monitor(print_freq=valid_freq)
    for i in range(n_iters):
        with mon:
            mon.plot('dummy plot', np.exp(-i))
    mon.flush()


def test_tracking():
    trivial_loops = 5
    shape = (3, 3)

    a = T.matrix('a')
    b = T.scalar('b')
    b_ = nn.placeholder(value=1.)

    def foo(x, y):
        for i in range(trivial_loops):
            nn.track('loop %d' % (i + 1), x)
            x += y
        return x

    c = foo(a, b)
    func = nn.function([a], c, givens={b: b_})

    a_ = np.zeros(shape, 'float32')
    res_numpy = [a_ + i for i in range(trivial_loops + 1)]
    assert np.allclose(func(a_), res_numpy[-1])

    for i in range(trivial_loops):
        trackeds = nn.eval_tracked_vars({a: a_, b: b_.get_value()})
        assert all(np.allclose(x, y) for x, y in zip(list(trackeds.values()), res_numpy[:-1]))


def test_yiq():
    from scipy import misc
    im = misc.imread('test_files/lena.jpg').astype(theano.config.floatX) / 255.
    im = np.transpose(im[None], (0, 3, 1, 2))

    a = T.tensor4('input')
    b = nn.utils.rgb2yiq(a)
    c = nn.utils.yiq2rgb(b)
    func = nn.function([a], [b[:, 0:1], c])

    im_out_y, im_out = func(im)
    misc.imsave('test_files/lena_from_yiq.jpg', np.transpose(im_out[0], (1, 2, 0)))
    misc.imsave('test_files/lena_y.jpg', im_out_y[0, 0])


def test_diff_of_gaussians():
    size1 = 21
    size2 = 51

    x = T.tensor4('image')
    x1 = nn.utils.difference_of_gaussian(x, size1)
    x2 = nn.utils.difference_of_gaussian(x, size2)
    func = nn.function([x], [x1, x2])

    from scipy import misc
    im = misc.imread('test_files/lena_small.png').astype(theano.config.floatX) / 255.
    im = np.transpose(im[None], (0, 3, 1, 2))
    x1_, x2_ = func(im)
    misc.imsave('test_files/lena_small_dog1.jpg', np.transpose(x1_[0], (1, 2, 0)))
    misc.imsave('test_files/lena_small_dog2.jpg', np.transpose(x2_[0], (1, 2, 0)))


def test_lr_annealing():
    base_lr = 1.
    n_iters = 50
    decay = 1e-2
    step = 10

    def anneal_learning_rate(lr, t, method='half-life', **kwargs):
        lr_ = lr
        if method == 'half-life':
            num_iters = kwargs.pop('num_iters', None)
            decay = kwargs.pop('decay', .1)
            if num_iters is None:
                raise ValueError('num_iters must be provided.')

            if (t == num_iters // 2) or (t == 3 * num_iters // 4):
                return lr_ * decay
            else:
                return lr_
        elif method == 'step':
            step = kwargs.pop('step', None)
            decay = kwargs.pop('decay', .5)
            if step is None:
                raise ValueError('step must be provided.')

            if t % step == 0:
                return lr_ * decay
            else:
                return lr_
        elif method == 'exponential':
            decay = kwargs.pop('decay', 1e-4)
            return lr_ * np.exp(-decay * t)
        elif method == 'linear':
            num_iters = kwargs.pop('num_iters', None)
            if num_iters is None:
                raise ValueError('num_iters must be provided.')
            return lr_ * (1. - t / np.cast[theano.config.floatX](num_iters))
        elif method == 'inverse':
            decay = kwargs.pop('decay', .01)
            return lr_ / (1. + decay * t)
        else:
            raise ValueError('Unknown annealing method.')

    idx = T.scalar('it', 'int32')
    for method in ('linear', 'step', 'exponential', 'inverse', 'half-life'):
        print('Testing method %s' % method)
        lr_ = nn.placeholder(dtype=theano.config.floatX, value=base_lr, name='lr')
        y = 0. + lr_
        nn.anneal_learning_rate(lr_, idx, method, num_iters=n_iters, decay=decay, step=step)
        func = nn.function([idx], y)
        vals_th, vals_np = [], []
        lr = base_lr
        for it in range(n_iters):
            func(it + 1)
            vals_th.append(lr_.get_value())
            lr = anneal_learning_rate(lr if method in ('step', 'half-life') else base_lr, it + 1, method,
                                      num_iters=n_iters, decay=decay, step=step)
            vals_np.append(lr)
        assert_allclose(vals_th, vals_np)


def test_data_manager():
    from scipy import misc
    path = 'test_files'
    image_list = [
        'ILSVRC2012_val_00000001.JPEG',
        'ILSVRC2012_val_00000002.JPEG',
        'ILSVRC2012_val_00000003.JPEG',
        'ILSVRC2012_val_00000004.JPEG'
    ]
    label_list = list(range(5))
    bs = 2
    n_epochs = 10
    crop = 224
    resize = 256
    mean = [112, 112, 112]
    std = [255, 255, 255]

    class DataManager(nn.DataManager):
        def __init__(self, path, placeholders, bs, n_epochs, *args, **kwargs):
            super(DataManager, self).__init__(path=path, placeholders=placeholders, batch_size=bs, n_epochs=n_epochs,
                                              *args, **kwargs)
            self.load_data()

        def load_data(self):
            images = np.array([misc.imread(self.path + '/' + img) for img in image_list])
            images = np.transpose(images, (0, 3, 1, 2))
            self.dataset = (images, np.array(label_list))
            self.data_size = len(image_list)

    X = nn.placeholder((bs, 3, crop, crop), name='input')
    y = nn.placeholder((bs,), 'int32', name='label')

    transforms = [
        nn.transforms.Normalize(mean=mean, std=std),
        nn.transforms.RandomCrop(crop, resize=resize)
    ]
    dm = DataManager(path, (X, y), bs, n_epochs, augmentation=transforms, shuffle=True)
    num_iters = n_epochs * len(dm) // bs
    while True:
        try:
            i = dm.__next__()
        except StopIteration:
            break

        print(i)
        images = X.get_value()
        for j in range(images.shape[0]):
            misc.imsave('%s/image %d at iteration %d.jpg' % (path, j, i), np.transpose(images[j], (1, 2, 0)) + .5)

    assert i + 1 == num_iters


def test_model_zoo_resnet18():
    top = 1
    root = 'test_files/'
    weight_file = root + 'resnet18_from_pytorch.npz'
    imagenet_classes_file = root + 'imagenet_classes.txt'
    image_list = [
        'ILSVRC2012_val_00000001.JPEG',
        'ILSVRC2012_val_00000002.JPEG',
        'ILSVRC2012_val_00000003.JPEG',
        'ILSVRC2012_val_00000004.JPEG',
        'ILSVRC2012_val_00000005.JPEG'
    ]

    net = nn.model_zoo.ResNet18((None, 3, 224, 224))
    net.load_params(weight_file)

    X = T.tensor4('input')
    test = nn.function([X], net(X))

    with open(imagenet_classes_file, 'r') as f:
        classes = [s.strip() for s in f.readlines()]
    mean_rgb = np.array([0.485, 0.456, 0.406], dtype=theano.config.floatX)[None, None, :]
    std_rgb = np.array([0.229, 0.224, 0.225], dtype=theano.config.floatX)[None, None, :]

    results = []
    for fname in image_list:
        print(fname)
        rawim, im = nn.utils.prep_image2(root + fname, mean_rgb, std_rgb)
        prob = test(im)[0]
        res = sorted(zip(classes, prob), key=lambda t: t[1], reverse=True)[:top]
        for c, p in res:
            print('  ', c, p)
            results.append(c)
        print('\n')

    sample_res = ['sea snake', 'ski', 'Shetland sheepdog, Shetland sheep dog, Shetland', 'soup bowl', 'crib, cot']
    assert results == sample_res


def test_model_zoo_resnet34():
    top = 1
    root = 'test_files/'
    weight_file = root + 'resnet34_from_pytorch.npz'
    imagenet_classes_file = root + 'imagenet_classes.txt'
    image_list = [
        'ILSVRC2012_val_00000001.JPEG',
        'ILSVRC2012_val_00000002.JPEG',
        'ILSVRC2012_val_00000003.JPEG',
        'ILSVRC2012_val_00000004.JPEG',
        'ILSVRC2012_val_00000005.JPEG'
    ]

    net = nn.model_zoo.ResNet34((None, 3, 224, 224))
    net.load_params(weight_file)

    X = T.tensor4('input')
    test = nn.function([X], net(X))

    with open(imagenet_classes_file, 'r') as f:
        classes = [s.strip() for s in f.readlines()]
    mean_rgb = np.array([0.485, 0.456, 0.406], dtype=theano.config.floatX)[None, None, :]
    std_rgb = np.array([0.229, 0.224, 0.225], dtype=theano.config.floatX)[None, None, :]

    results = []
    for fname in image_list:
        print(fname)
        rawim, im = nn.utils.prep_image2(root + fname, mean_rgb, std_rgb)
        prob = test(im)[0]
        res = sorted(zip(classes, prob), key=lambda t: t[1], reverse=True)[:top]
        for c, p in res:
            print('  ', c, p)
            results.append(c)
        print('\n')

    sample_res = ['sea snake', 'ski', 'Shetland sheepdog, Shetland sheep dog, Shetland', 'soup bowl', 'cradle']
    assert results == sample_res


def test_model_zoo_resnet50():
    top = 1
    root = 'test_files/'
    weight_file = root + 'resnet50_from_pytorch.npz'
    imagenet_classes_file = root + 'imagenet_classes.txt'
    image_list = [
        'ILSVRC2012_val_00000001.JPEG',
        'ILSVRC2012_val_00000002.JPEG',
        'ILSVRC2012_val_00000003.JPEG',
        'ILSVRC2012_val_00000004.JPEG',
        'ILSVRC2012_val_00000005.JPEG'
    ]

    net = nn.model_zoo.ResNet50((None, 3, 224, 224))
    net.load_params(weight_file)

    X = T.tensor4('input')
    test = nn.function([X], net(X))

    with open(imagenet_classes_file, 'r') as f:
        classes = [s.strip() for s in f.readlines()]
    mean_rgb = np.array([0.485, 0.456, 0.406], dtype=theano.config.floatX)[None, None, :]
    std_rgb = np.array([0.229, 0.224, 0.225], dtype=theano.config.floatX)[None, None, :]

    results = []
    for fname in image_list:
        print(fname)
        rawim, im = nn.utils.prep_image2(root + fname, mean_rgb, std_rgb)
        prob = test(im)[0]
        res = sorted(zip(classes, prob), key=lambda t: t[1], reverse=True)[:top]
        for c, p in res:
            print('  ', c, p)
            results.append(c)
        print('\n')

    sample_res = ['sea snake', 'ski', 'Shetland sheepdog, Shetland sheep dog, Shetland', 'soup bowl', 'crib, cot']
    assert results == sample_res


def test_model_zoo_resnet101():
    top = 1
    root = 'test_files/'
    weight_file = root + 'resnet101_from_pytorch.npz'
    imagenet_classes_file = root + 'imagenet_classes.txt'
    image_list = [
        'ILSVRC2012_val_00000001.JPEG',
        'ILSVRC2012_val_00000002.JPEG',
        'ILSVRC2012_val_00000003.JPEG',
        'ILSVRC2012_val_00000004.JPEG',
        'ILSVRC2012_val_00000005.JPEG'
    ]

    net = nn.model_zoo.ResNet101((None, 3, 224, 224))
    net.load_params(weight_file)

    X = T.tensor4('input')
    test = nn.function([X], net(X))

    with open(imagenet_classes_file, 'r') as f:
        classes = [s.strip() for s in f.readlines()]
    mean_rgb = np.array([0.485, 0.456, 0.406], dtype=theano.config.floatX)[None, None, :]
    std_rgb = np.array([0.229, 0.224, 0.225], dtype=theano.config.floatX)[None, None, :]

    results = []
    for fname in image_list:
        print(fname)
        rawim, im = nn.utils.prep_image2(root + fname, mean_rgb, std_rgb)
        prob = test(im)[0]
        res = sorted(zip(classes, prob), key=lambda t: t[1], reverse=True)[:top]
        for c, p in res:
            print('  ', c, p)
            results.append(c)
        print('\n')

    sample_res = ['sea snake', 'ski', 'Shetland sheepdog, Shetland sheep dog, Shetland', 'soup bowl', 'crib, cot']
    assert results == sample_res


def test_model_zoo_resnet152():
    top = 1
    root = 'test_files/'
    weight_file = root + 'resnet152_from_pytorch.npz'
    imagenet_classes_file = root + 'imagenet_classes.txt'
    image_list = [
        'ILSVRC2012_val_00000001.JPEG',
        'ILSVRC2012_val_00000002.JPEG',
        'ILSVRC2012_val_00000003.JPEG',
        'ILSVRC2012_val_00000004.JPEG',
        'ILSVRC2012_val_00000005.JPEG'
    ]

    net = nn.model_zoo.ResNet152((None, 3, 224, 224))
    net.load_params(weight_file)

    X = T.tensor4('input')
    test = nn.function([X], net(X))

    with open(imagenet_classes_file, 'r') as f:
        classes = [s.strip() for s in f.readlines()]
    mean_rgb = np.array([0.485, 0.456, 0.406], dtype=theano.config.floatX)[None, None, :]
    std_rgb = np.array([0.229, 0.224, 0.225], dtype=theano.config.floatX)[None, None, :]

    results = []
    for fname in image_list:
        print(fname)
        rawim, im = nn.utils.prep_image2(root + fname, mean_rgb, std_rgb)
        prob = test(im)[0]
        res = sorted(zip(classes, prob), key=lambda t: t[1], reverse=True)[:top]
        for c, p in res:
            print('  ', c, p)
            results.append(c)
        print('\n')

    sample_res = ['sea snake', 'ski', 'Shetland sheepdog, Shetland sheep dog, Shetland', 'soup bowl', 'crib, cot']
    assert results == sample_res


def test_upsampling_layer():
    input_shape = (1, 3, 220, 220)
    X = T.tensor4('input')
    bilinear_up = nn.UpsamplingLayer(input_shape, 4, method='bilinear')
    nearest_up = nn.UpsamplingLayer(input_shape, 4, method='nearest')

    x_bilinear = bilinear_up(X)
    x_nearest = nearest_up(X)
    func = nn.function([X], [x_bilinear, x_nearest])

    from scipy import misc
    im = np.transpose(misc.imread('test_files/lena_small.png').astype(theano.config.floatX), (2, 0, 1))[None] / 255.
    x_bi_, x_ne_ = func(im)
    x_bi_ = np.transpose(x_bi_[0], (1, 2, 0))
    x_ne_ = np.transpose(x_ne_[0], (1, 2, 0))
    misc.imsave('test_files/lena_small_bilinear.jpg', x_bi_)
    misc.imsave('test_files/lena_small_nearest.jpg', x_ne_)


def test_vertical_flipping():
    from scipy import misc
    im = misc.imread('test_files/lena.jpg').astype(theano.config.floatX) / 255.
    im = np.transpose(np.stack((im, im)), (0, 3, 1, 2))
    trans = nn.transforms.RandomVerticalFlip()
    im = trans(im)
    im = np.transpose(im, (0, 2, 3, 1))
    misc.imsave('test_files/lena_rand_ver_flip_1.jpg', im[0])
    misc.imsave('test_files/lena_rand_ver_flip_2.jpg', im[1])


def test_spearman():
    size = (64,)
    mag = 10.

    pred = T.vector('pred')
    gt = T.vector('gt')
    corr = nn.spearmanr(pred, gt)
    f = theano.function([pred, gt], corr)

    gt = np.random.normal(size=size).astype(theano.config.floatX) * mag
    pred = np.random.uniform(size=size).astype(theano.config.floatX) * mag
    c = f(pred, gt)

    from scipy import stats
    c_ref, _ = stats.spearmanr(pred.flatten(), gt.flatten())
    assert_allclose(c_ref, c)


def test_pearsonr():
    size = (64, 1000)

    pred = T.matrix('pred')
    gt = T.matrix('gt')
    corr = nn.pearsonr(pred, gt)
    f = nn.function([pred, gt], corr)

    gt = np.random.normal(size=size).astype(theano.config.floatX)
    pred = np.random.uniform(size=size).astype(theano.config.floatX)
    c = f(pred, gt)

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
    mon = nn.Monitor(print_freq=valid_freq, hist_last=True)
    filter = np.random.normal(scale=.25, size=size)
    for i in range(n_iters):
        with mon:
            mon.plot('foo', i ** 2 / 4.)
            mon.hist('filter_last', filter * (i + 1) * .25 + i / 10., last_only=True, n_bins=10)
            mon.hist('filter', filter * (i + 1) * .25 + i / 15.)
    mon.flush()


def test_model_zoo_vgg16():
    top = 1
    root = 'test_files/'
    weight_file = root + 'vgg16_from_pytorch.npz'
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
    test = nn.function([X], net(X))

    with open(imagenet_classes_file, 'r') as f:
        classes = [s.strip() for s in f.readlines()]
    mean_rgb = np.array([0.485, 0.456, 0.406], dtype=theano.config.floatX)[None, None, :]
    std_rgb = np.array([0.229, 0.224, 0.225], dtype=theano.config.floatX)[None, None, :]

    results = []
    for fname in image_list:
        print(fname)
        rawim, im = nn.utils.prep_image2(root + fname, mean_rgb, std_rgb)
        prob = test(im)[0]
        res = sorted(zip(classes, prob), key=lambda t: t[1], reverse=True)[:top]
        for c, p in res:
            print('  ', c, p)
            results.append(c)
        print('\n')

    sample_res = ['sea snake', 'ski', 'Shetland sheepdog, Shetland sheep dog, Shetland', 'soup bowl', 'bassinet']
    assert results == sample_res


def test_model_zoo_vgg19():
    top = 1
    root = 'test_files/'
    weight_file = root + 'vgg19_from_pytorch.npz'
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
    test = nn.function([X], net(X))

    with open(imagenet_classes_file, 'r') as f:
        classes = [s.strip() for s in f.readlines()]
    mean_rgb = np.array([0.485, 0.456, 0.406], dtype=theano.config.floatX)[None, None, :]
    std_rgb = np.array([0.229, 0.224, 0.225], dtype=theano.config.floatX)[None, None, :]

    results = []
    for fname in image_list:
        print(fname)
        rawim, im = nn.utils.prep_image2(root + fname, mean_rgb, std_rgb)
        prob = test(im)[0]
        res = sorted(zip(classes, prob), key=lambda t: t[1], reverse=True)[:top]
        for c, p in res:
            print('  ', c, p)
            results.append(c)
        print('\n')

    sample_res = ['rock python, rock snake, Python sebae', 'ski', 'Shetland sheepdog, Shetland sheep dog, Shetland',
                  'soup bowl', 'bassinet']
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
    out = nn.utils.reflection_pad(input, padding, 2)
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
