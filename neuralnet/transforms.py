"""
Written by Duc Nguyen. Inspired from PyTorch.
"""

import numpy as np
import numbers
import collections


class Normalize:
    """Normalize a tensor variable
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean, 'float32')[None, :, None, None]
        self.std = np.array(std, 'float32')[None, :, None, None]

    def __call__(self, batch):
        return (batch - self.mean) / self.std


class RandomCrop:
    """Crop the given batch of numpy images"""

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = tuple(size)
        self.padding = padding
        if padding:
            self.pad = Pad(padding)

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (numpy array of shape (c, h, w)): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        _, h, w = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = np.random.randint(0, h - th)
        j = np.random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, batch):
        def apply(img):
            i, j, h, w = self.get_params(img, self.size)
            img = img[:, i:i + h, j:j + w]
            return img

        if self.padding:
            batch = self.pad(batch)
        res = np.array([apply(img) for img in batch], 'float32')
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

        return new.astype('float32')


class RandomHorizontalFlip:
    def __call__(self, batch):
        return np.array([img[..., ::-1] if np.random.random() < .5 else img for img in batch], 'float32')


class RandomVerticalFlip:
    def __call__(self, batch):
        return np.array([img[:, ::-1] if np.random.random() < .5 else img for img in batch], 'float32')


if __name__ == '__main__':
    from scipy import misc
    from matplotlib import pyplot as plt
    im = misc.imread('E:/Users/Duc/frame_interpolation/utils/Camila Cabello - Havana ft. Young Thug/100.jpg').astype(
        'float32') / 255.
    im = np.transpose(np.stack((im, im)), (0, 3, 1, 2))
    trans = RandomVerticalFlip()
    im = trans(im)
    im = np.transpose(im, (0, 2, 3, 1))
    plt.figure()
    plt.imshow(im[0])
    plt.figure()
    plt.imshow(im[1])
    plt.show()
