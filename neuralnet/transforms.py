"""
Written by Duc Nguyen. Inspired from PyTorch.
"""

import numpy as np
import numbers
import collections
import theano

from neuralnet import utils


class Normalize:
    """Normalize a tensor variable
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean, theano.config.floatX)[None, :, None, None]
        self.std = np.array(std, theano.config.floatX)[None, :, None, None]

    def __call__(self, batch):
        return (batch - self.mean) / self.std


class RandomCrop:
    """Crop the given batch of numpy images"""

    def __init__(self, size, padding=0, resize=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = tuple(size)
        self.padding = padding
        self.resize = resize
        if padding:
            self.pad = Pad(padding)

    def __call__(self, batch):
        if self.padding:
            batch = self.pad(batch)
        res = np.array(
            [np.transpose(utils.crop_random(np.transpose(img, (1, 2, 0)), self.size, self.resize), (2, 0, 1)) for
             img in batch], theano.config.floatX)
        return res


class Pad:
    """Pad a batch of images
    """

    def __init__(self, padding, fill=0):
        """

        :param padding: Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        :param fill: Pixel fill value. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
        """
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, tuple))
        if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill

    def __call__(self, batch):
        n, c, h, w = batch.shape
        if isinstance(self.fill, (list, tuple)):
            assert c == len(self.fill), 'The length of fill must be the same as the number of channels, ' \
                                        'got %d and %d.' % (len(self.fill), c)
            fill = np.array(self.fill)[None, :, None, None]
        else:
            fill = self.fill

        if isinstance(self.padding, numbers.Number):
            padding = int(self.padding)
            new = np.ones((n, c, h+2*padding, w+2*padding)) * fill
            new[:, :, padding:padding+h, padding:padding+w] = batch
        else:
            if len(self.padding) == 2:
                lr, tb = self.padding
                new = np.ones((n, c, h+2*tb, w+2*lr)) * fill
                new[:, :, tb:tb+h, lr:lr+w] = batch
            else:
                le, to, ri, bo = self.padding
                new = np.ones((n, c, h+to+bo, w+le+ri)) * fill
                new[:, :, to:to+h, le:le+w] = batch

        return new.astype(theano.config.floatX)


class RandomHorizontalFlip:
    def __call__(self, batch):
        return np.array([img[..., ::-1] if np.random.random() < .5 else img for img in batch], theano.config.floatX)


class RandomVerticalFlip:
    def __call__(self, batch):
        return np.array([img[:, ::-1] if np.random.random() < .5 else img for img in batch], theano.config.floatX)
