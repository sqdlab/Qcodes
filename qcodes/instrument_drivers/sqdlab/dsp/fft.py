from collections import namedtuple
import numpy as np

import reikna
from reikna.core import Type, Annotation, Parameter
from reikna.cluda import dtypes
from reikna.algorithms import PureParallel
from reikna.transformations import Transformation, copy
from reikna.fft import FFT

class Broadcast(Transformation):
    code = """
        ${output.store_same}(${input.load_idx}(${', '.join(indices(idxs))}));
    """
    def __init__(self, shape, in_type):
        '''
        Input transformation that broadcasts arrays.

        Arguments
        ---------
        shape: `tuple`
            Shape of the output array.
        in_type: `reikna.core.Type`
            Shape and dtype of the input vector.
        '''
        # check if broadcastable and make ndim the same for input and output
        for so, si in zip(shape[::-1], in_type.shape[::-1]):
            if (so != si) and (si != 1) and (so != 1):
                raise ValueError('Unable to broadcast {} to {}'
                                 .format(in_type.shape, shape))
        in_shape = in_type.shape
        # index calculation function
        def indices(vars):
            result = []
            for idx in range(len(in_shape)):
                var = vars[-1-idx] if len(list(vars)) > idx else None
                singleton = (idx >= len(in_shape)) or (in_shape[-1-idx] == 1)
                result.append('0' if (var is None) or singleton else var)
            return tuple(result[::-1])
        # generate parameters
        out_type = Type(in_type.dtype, shape)
        out_param = Parameter('output', Annotation(out_type, 'o'))
        in_param = Parameter('input', Annotation(in_type, 'i'))
        super(Broadcast, self).__init__(
            [out_param, in_param], self.code, dict(indices=indices)
        )


class Padded(Transformation):
    code = """
        ${ctype} value = ${default};
        if (true
            %for idx, size in zip(idxs, input.shape):
            && (${idx} < ${size})
            %endfor
        ) {
            value = ${input.load_idx}(${', '.join(idxs)});
        }
        ${output.store_same}(value);
    """
    def __init__(self, out_type, in_type, default):
        '''
        Input transformation that pads an array parameter with a default value.

        Arguments
        ---------
        out_type: `reikna.core.Type`
            Data type and shape of the output array.
        in_type: `reikna.core.Type`
            Data type and shape of the input array. Data type must be implicitly
            castable to the output type.
        default
            Default value for padding. Must be a valid literal for the output
            dtype.
        '''
        out_param = Parameter('output', Annotation(out_type, 'o'))
        in_param = Parameter('input', Annotation(in_type, 'i'))
        super(Padded, self).__init__(
            [out_param, in_param], self.code, 
            dict(ctype=out_type.ctype, default=default)
        )


class Cast(Transformation):
    code = """
    ${input.ctype} value = ${input.load_same};
    ${output.store_same}(convert_${ctype}(${input.load_same}));
    """
    def __init__(self, out_type, in_type):
        '''
        Input transformation that implements an explicit type cast.

        Arguments
        ---------
        out_type: `reikna.core.Type`
            Output dtype and shape.
        in_type: `reikna.core.Type`
            Input dtype and shape.

        Notes
        -----
        * `in_type` and `out_type` shapes must be equal.
        * Does not support real-to-complex and complex-to-real conversions.
        '''
        if(in_type.shape != out_type.shape):
            raise ValueError('shapes of out_type and in_type must be equal.')
        if(issubclass(in_type.dtype.type, np.complexfloating) != 
           issubclass(out_type.dtype.type, np.complexfloating)
           #np.iscomplexobj(in_type) != np.iscomplexobj(out_type)
           ):
            raise ValueError('Unable to cast real to complex and vice versa.')
        out_param = Parameter('output', Annotation(out_type, 'o'))
        in_param = Parameter('input', Annotation(in_type, 'i'))
        ctype = out_type.ctype.replace('unsigned ', 'u')
        super(Cast, self).__init__([out_param, in_param], self.code,
                                   dict(ctype=ctype))


class Complex(Transformation):
    code = """
    ${output.store_same}((${output.ctype})(${input.load_same}, 0.));
    """
    def __init__(self, in_type):
        '''
        Input transformation that converts real numbers to complex numbers with
        the same dtype of the components.

        Arguments
        ---------
        in_type: `reikna.core.Type`
            Input data type and shape.
        '''
        try:
            ttype = reikna.cluda.dtypes.complex_for(in_type.dtype)
        except KeyError:
            raise ValueError('unsupported type {}.'.format(in_type.dtype))
        out_param = Parameter('output', Annotation(Type(ttype, in_type.shape), 'o'))
        in_param = Parameter('input', Annotation(in_type, 'i'))
        super(Complex, self).__init__([out_param, in_param], self.code)


def Multiply(type):
    return PureParallel(
        [Parameter('output', Annotation(type, 'o')),
         Parameter('in1', Annotation(type, 'i')),
         Parameter('in2', Annotation(type, 'i'))],
        """
        ${ctype} f1 = ${in1.load_same}, f2 = ${in2.load_same};
        #if ${complex}
        ${output.store_same}((${ctype})(f1.x*f2.x - f1.y*f2.y, f1.x*f2.y + f1.y*f2.x));
        #else
        ${output.store_same}(f1*f2);
        #endif
        """,
        render_kwds=dict(ctype=type.ctype, complex=int(dtypes.is_complex(type)))
    )


class FFT(FFT):
    def __init__(self, arr_t, padding=False, axes=None, **kwargs):
        '''
        Wrapper around `reikna.fft.FFT` with automatic real-to-complex casting
        and optional padding for higher performance.

        Input
        -----
        padding: bool, default=True
            If True, the input array is padded to the next power of two on the 
            transformed axes.
        axes: tuple
            Axes over which to perform the transform. Defaults to all axes.

        Note
        ----
        Because reikna does not allow nodes of the transformation tree with the
        identical names, the input array is called `input_`.
        '''
        if axes is None:
            axes = range(len(arr_t.shape))# if axes is None else tuple(axes)
        else:
            axes = tuple(v+len(arr_t.shape) if v<0 else v for v in axes)
        for v in axes:
            if v not in range(0, len(arr_t.shape)):
                raise IndexError('axis is out of range')
        dtype = (arr_t.dtype if dtypes.is_complex(arr_t.dtype) else 
                 dtypes.complex_for(arr_t.dtype))
        if padding:
            shape = tuple(1<<int(np.ceil(np.log2(v))) if ax in axes else v
                          for ax, v in enumerate(arr_t.shape))
        else:
            shape = arr_t.shape
        super(FFT, self).__init__(Type(dtype, shape), axes=axes, **kwargs)
        input = self.parameter.input
        if dtype != arr_t.dtype:
            complex_tr = Complex(Type(arr_t.dtype, input.shape))
            input.connect(complex_tr, complex_tr.output, in_real=complex_tr.input)
            input = self.parameter.in_real
        if shape != arr_t.shape:
            pad_tr = Padded(input, arr_t, default='0.')
            input.connect(pad_tr, pad_tr.output, in_padded=pad_tr.input)
            input = self.parameter.in_padded
        copy_tr = copy(input)
        input.connect(copy_tr, copy_tr.output, input_=copy_tr.input)


class FFTConvolve(object):
    def __init__(self, in1_type, in2_type, axis=-1):
        '''
        Fast convolution with FFT

        Uses transforms of length N1+N2 padded to a power of two, because 
        overlap-add is not significantly faster for the indended shape ranges.

        Input
        -----
        in1_type, in2_type: `reikna.core.Type`
            Shape and dtype of the arrays to be convolved.
        axis: `int`
            Array axis over which the convolution is evaluated.

        Notes
        -----
        * The output is always an array of complex numbers.
        * The arrays are matched using numpy's broadcasting rules.
        '''
        self._thread = None
        # normalize axis
        ndim = max(len(in1_type.shape), len(in2_type.shape))
        if axis < 0:
            axis += ndim
        if axis not in range(ndim):
            raise ValueError('axis is out of range.')
        # check if in1 and in2 are broadcastable
        for ax, s1, s2 in zip(range(ndim-1, 0, -1), in1_type.shape[::-1], in2_type.shape[::-1]):
            if (ax != axis) and (s1 != s2) and (s1 != 1) and (s2 != 1):
                raise ValueError('in1 and in2 have incompatible shapes')
        # calculate shapes
        in1_shape = (1,)*(ndim-len(in1_type.shape)) + in1_type.shape
        in2_shape = (1,)*(ndim-len(in2_type.shape)) + in2_type.shape
        in1_padded = in1_shape[:axis] + (in1_shape[axis]+in2_shape[axis]-1,) + in1_shape[axis+1:]
        in2_padded = in2_shape[:axis] + (in1_shape[axis]+in2_shape[axis]-1,) + in2_shape[axis+1:]
        out_shape = tuple(max(s1, s2) for s1, s2 in zip(in1_padded, in2_padded))
        out_dtype = (in1_type.dtype if dtypes.is_complex(in1_type.dtype) else 
                     dtypes.complex_for(in1_type.dtype))

        fft1 = FFT(Type(in1_type.dtype, in1_padded), axes=(axis,))
        pad_in1 = Padded(fft1.parameter.input_, Type(in1_type.dtype, in1_shape), default='0.')
        fft1.parameter.input_.connect(pad_in1, pad_in1.output, input_p=pad_in1.input)
        fft2 = FFT(Type(in2_type.dtype, in2_padded), axes=(axis,))
        pad_in2 = Padded(fft2.parameter.input_, Type(in2_type.dtype, in2_shape), default='0.')
        fft2.parameter.input_.connect(pad_in2, pad_in2.output, input_p=pad_in2.input)
        mul = Multiply(Type(out_dtype, out_shape))
        bcast_in1 = Broadcast(out_shape, fft1.parameter.output)
        mul.parameter.in1.connect(bcast_in1, bcast_in1.output, in1_p=bcast_in1.input)
        bcast_in2 = Broadcast(out_shape, fft2.parameter.output)
        mul.parameter.in2.connect(bcast_in2, bcast_in2.output, in2_p=bcast_in2.input)
        ifft = FFT(Type(out_dtype, out_shape), axes=(axis,))
        self._comps = [fft1, fft2, mul, ifft]
        # emulate reikna parameter attribute        
        parameters = namedtuple('DummyParameters', ['output', 'in1', 'in2'])
        self.parameter = parameters(ifft.parameter.output, in1_type, in2_type)

    def compile(self, thread):
        '''Compile all component computations.'''
        self._thread = thread
        ccomps = [comp.compile(thread) for comp in self._comps]
        self.fft1, self.fft2, self.mul, self.ifft = ccomps
        return self

    def __call__(self, output, in1, in2):
        if self._thread is None:
            raise RuntimeError("Computations must be compile()'d first")
        out1 = self._thread.empty_like(self.fft1.parameter.output)
        self.fft1(out1, in1)
        out2 = self._thread.empty_like(self.fft2.parameter.output)
        self.fft2(out2, in2)
        prod = self._thread.empty_like(self.mul.parameter.output)
        self.mul(prod, out1, out2)
        return self.ifft(output, prod, True)