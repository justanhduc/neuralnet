import numpy as np
import theano
from theano import gof
import theano.tensor as T

__all__ = ['meshgrid']


class MeshgridOp(gof.Op):
    __props__ = ()

    def output_type(self, inps):
        return T.TensorType(inps[0].dtype, broadcastable=[False] * len(inps))

    def make_node(self, *xi):
        xi = [T.squeeze(T.as_tensor_variable(x)) for x in xi]
        if any([x.ndim != 1 for x in xi]):
            raise TypeError('%s: input must be 1D' % self.__class__.__name__)

        return gof.Apply(self, xi, [self.output_type(xi)(), self.output_type(xi)()])

    def perform(self, node, inputs, output_storage, params=None):
        X, Y = np.meshgrid(*inputs, indexing='ij')
        output_storage[0][0] = X
        output_storage[1][0] = Y


meshgrid_op = MeshgridOp()


def meshgrid(*xi, **kwargs):
    indexing = kwargs.get('indexing', 'xy')
    if indexing not in ['xy', 'ij']:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")

    ndim = len(xi)
    s0 = tuple(range(ndim))
    outputs = meshgrid_op(*xi)

    if indexing == 'xy' and ndim > 1:
        outputs = [output.dimshuffle((1, 0) + s0[2:]).astype(theano.config.floatX) for output in outputs]
    return outputs


def linspace(start, stop, num, dtype=None):
    """
    Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the
    interval [`start`, `stop`].

    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : scalar
        The starting value of the sequence.
    stop : scalar
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int
        Number of samples to generate. Default is 50. Must be non-negative.
    dtype : dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.

        .. versionadded:: 1.9.0

    Returns
    -------
    samples : ndarray
        There are `num` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``
        (depending on whether `endpoint` is True or False).
    step : float, optional
        Only returned if `retstep` is True

        Size of spacing between samples.

    """
    dtype = dtype if dtype else theano.config.floatX
    div = num - 1

    # Convert float/complex array scalars to float, gh-3504
    # and make sure one can use variables that have an __array_interface__, gh-6634
    start = T.cast(start, theano.config.floatX)
    stop = T.cast(stop, theano.config.floatX)

    y = T.arange(0, num, dtype=dtype)

    delta = stop - start
    # In-place multiplication y *= delta/div is faster, but prevents the multiplicant
    # from overriding what class is produced, and thus prevents, e.g. use of Quantities,
    # see gh-7142. Hence, we multiply in place only for standard scalar types.
    if num > 1:
        step = delta / div
        if step == 0:
            # Special handling for denormal numbers, gh-5437
            y /= div
            y *= delta
        else:
            y *= step
    else:
        # 0 and 1 item long sequences have an undefined step
        step = None
        # Multiply with delta to allow possible override of output class.
        y = y * delta

    y += start

    if num > 1:
        y[-1] = stop

    return y.astype(dtype)
