import numpy as np
import theano
from theano import gof
import theano.tensor as T

__all__ = ['fft', 'ifft', 'meshgrid']


class RFFTOp(gof.Op):
    __props__ = ()

    def output_type(self, inp):
        # add extra dim for real/imag
        return T.TensorType(inp.dtype, broadcastable=[False] * (inp.type.ndim + 1))

    def make_node(self, a, s=None):
        a = T.as_tensor_variable(a)
        if a.ndim < 2:
            raise TypeError('%s: input must have dimension > 2, with first dimension batches' %
                            self.__class__.__name__)

        if s is None:
            s = a.shape[1:]
            s = T.as_tensor_variable(s)
        else:
            s = T.as_tensor_variable(s)
            if s.dtype not in T.integer_dtypes:
                raise TypeError('%s: length of the transformed axis must be'
                                ' of type integer' % self.__class__.__name__)
        return gof.Apply(self, [a, s], [self.output_type(a)()])

    def perform(self, node, inputs, output_storage):
        a = inputs[0]
        s = inputs[1]

        A = np.fft.fftn(a, s=tuple(s))
        # Format output with two extra dimensions for real and imaginary
        # parts.
        out = np.zeros(A.shape + (2,), dtype=a.dtype)
        out[..., 0], out[..., 1] = np.real(A), np.imag(A)
        output_storage[0][0] = out


fft_op = RFFTOp()


class IRFFTOp(gof.Op):

    __props__ = ()

    def output_type(self, inp):
        # remove extra dim for real/imag
        return T.TensorType(inp.dtype,
                            broadcastable=[False] * (inp.type.ndim - 1))

    def make_node(self, a, s=None):
        a = T.as_tensor_variable(a)
        if a.ndim < 3:
            raise TypeError('%s: input must have dimension >= 3,  with ' %
                            self.__class__.__name__ +
                            'first dimension batches and last real/imag parts')

        if s is None:
            s = a.shape[1:-1]
            s = T.as_tensor_variable(s)
        else:
            s = T.as_tensor_variable(s)
            if s.dtype not in T.integer_dtypes:
                raise TypeError('%s: length of the transformed axis must be'
                                ' of type integer' % self.__class__.__name__)
        return gof.Apply(self, [a, s], [self.output_type(a)()])

    def perform(self, node, inputs, output_storage):
        a = inputs[0]
        s = inputs[1]

        # Reconstruct complex array from two float dimensions
        inp = a[..., 0] + 1j * a[..., 1]
        out = np.real(np.fft.ifftn(inp, s=tuple(s)))
        # Remove numpy's default normalization
        # Cast to input type (numpy outputs float64 by default)
        output_storage[0][0] = (out * s.prod()).astype(a.dtype)


ifft_op = IRFFTOp()


def fft(inp, norm=None):
    s = inp.shape[1:]
    cond_norm = _unitary(norm)
    scaling = 1
    if cond_norm == "ortho":
        scaling = T.sqrt(s.prod().astype(inp.dtype))

    return fft_op(inp, s) / scaling


def ifft(inp, norm=None, is_odd=False):
    if is_odd not in (True, False):
        raise ValueError("Invalid value %s for id_odd, must be True or False" % is_odd)

    s = inp.shape[1:-1]
    cond_norm = _unitary(norm)
    scaling = 1
    # Numpy's default normalization is 1/N on the inverse transform.
    if cond_norm is None:
        scaling = s.prod().astype(inp.dtype)
    elif cond_norm == "ortho":
        scaling = T.sqrt(s.prod().astype(inp.dtype))
    return ifft_op(inp, s) / scaling


def _unitary(norm):
    if norm not in (None, "ortho", "no_norm"):
        raise ValueError("Invalid value %s for norm, must be None, 'ortho' or "
                         "'no norm'" % norm)
    return norm


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
