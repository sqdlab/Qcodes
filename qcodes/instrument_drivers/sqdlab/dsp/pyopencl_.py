import numpy as np
import pyopencl as cl
import pyopencl.array
from mako.template import Template
import reikna

def ctype(dtype):
    '''Find opencl type for a numpy dtype.'''
    type_map = [(np.float32, 'float'), 
                (np.float64, 'double'), 
                (np.complex64, 'float2'),
                (np.complex128, 'double2')]
    for _dtype, ctype in type_map:
        if dtype == _dtype:
            return ctype
    raise TypeError('unknown type {}'.format(dtype))


class ShortToFloat(object):
    code = """
        #define itype short
        //#define rtype float
        #define btype char

        __kernel void to_${rtype}(
            __global ${rtype} *output,
            __global const itype *input,
            const unsigned int blocklen,
            const unsigned int bits
        ) {
            const size_t in_idx = get_global_id(1)*blocklen + get_global_id(0), 
                         out_idx = get_global_id(1)*get_global_size(0) + get_global_id(0);
            itype sample = (get_global_id(0) < blocklen) ? input[in_idx] : 0;
            // sign extension
            if(bits != 16) {
                const itype sign_bit = 1 << (bits - 1);
                sample = (sample & (sign_bit - 1)) - (sample & sign_bit);
            }
            output[out_idx] = (get_global_id(0) < blocklen) ? convert_${rtype}(sample) : 0.;
        }

        __kernel void to_${rtype}_with_markers(
            __global ${rtype} *output,
            __global btype *marker1,
            __global btype *marker2, 
            __global const itype *input, 
            const unsigned int blocklen,
            const unsigned int bits
        ) {
            const size_t in_idx = get_global_id(1)*blocklen + get_global_id(0), 
                         out_idx = get_global_id(1)*get_global_size(0) + get_global_id(0);
            itype sample = (get_global_id(0) < blocklen) ? input[in_idx] : 0;
            marker1[out_idx] = sample & 0x8000 ? 1 : 0;
            marker2[out_idx] = sample & 0x4000 ? 1 : 0;
            //sample = (sample << 2) >> 2; // sign extension (sadly optimized away by the compiler)
            //sample = (sample & 0x3fff) | ((sample << 1) & (1<<14)) | ((sample << 2) & (1<<15)); // sign extension
            // sign extension
            const itype sign_bit = 1 << (bits - 1);
            sample = (sample & (sign_bit - 1)) - (sample & sign_bit);
            output[out_idx] = convert_${rtype}(sample);
        }
    """

    def __init__(self, context):
        code = ''.join(Template(self.code).render(rtype=rtype) 
                       for rtype in ('float', 'double'))
        self.prg = cl.Program(context, code).build()

    def __call__(self, queue, in1, markers=False, bits=16, padding=0, out=None, 
                 rtype=None):
        '''
        Convert input samples to floating point numbers and separate the marker bits.
        
        Input
        -----
        queue
            OpenCL command queue the operation is performed in. 
        in1: `pyopencl.array.Array` or `np.ndarray`
            Input samples. Must be a 2d numpy ndarray or pyopencl Array of 16bit 
            integers.
        markers: `bool`
            If True, the two most significant bits are considered markers.
        padding: `int`
            Number of bytes of zero padding to add to each block of samples.
        out: `pyopencl.array.Array`, optional
            Output array if markers is False, tuple of output and marker arrays
            if markers is True. Use pre-allocated arrays to speed up processing.
        rtype: `np.dtype`, 
            Data type of the analog samples. Default out.dtype if out is set and
            np.float32 otherwise.
            
        Return
        ------
        out: pyopencl.array.Array, dtype=rtype, shape=(in1.shape[0], blocklen)
            Sample values as float
        marker1, marker2: cl.Array, dtype=uint8, same shape as out
            uint8(1) for each set and uint8(0) for each unset marker bit
        '''
        #rtype = np.float32
        btype = np.uint8
        # check input and transfer to device
        if (in1.dtype != np.int16) and (in1.dtype != np.uint16):
            raise TypeError('in1 must be a vector of 16bit integers.')
        if in1.ndim < 1:
            raise ValueError('in1 must be a vector.')
        if isinstance(in1, np.ndarray):
            in1 = cl.array.to_device(queue, in1, async=True)
        if rtype is None:
            if out is None:
                rtype = np.float32
            else:
                rtype = out.dtype if not markers else out[0].dtype
        if (rtype != np.float32) and (rtype != np.float64):
            raise ValueError('rtype must be float32 or float64')
        ctype = 'float' if rtype == np.float32 else 'double'
        # check output(s)
        out_shape = in1.shape[:-1] + (in1.shape[-1]+padding,)
        if out is not None:
            for arr in out if markers else (out,):
                if arr.shape != out_shape:
                    raise ValueError('shape of outputs must be {}'.format(out_shape))
        global_size = (out_shape[-1], np.prod((1,)+out_shape[:-1]))
        if markers:
            # allocate result arrays
            if out is None:
                out = cl.array.Array(queue, out_shape, rtype)
                marker1 = cl.array.Array(queue, out.shape, btype)
                marker2 = cl.array.Array(queue, out.shape, btype)
            else:
                out, marker1, marker2 = out
            # run computation
            function = getattr(self.prg, 'to_{}_with_markers'.format(ctype))
            function(
                queue, global_size, None, 
                out.data, marker1.data, marker2.data, in1.data, np.uint32(in1.shape[-1], np.uint32(bits))
            )
            return out, marker1, marker2
        else:
            # allocate result arrays
            if out is None:
                out = cl.array.Array(queue, out_shape, rtype)
            # run computation
            function = getattr(self.prg, 'to_{}'.format(ctype))
            function(queue, global_size, None, out.data, in1.data, 
                     np.uint32(in1.shape[-1]), np.uint32(bits))
            return out


class BroadcastMultiply(object):
    code = """
        __kernel void bcast_mul_${suffix}(
            __global ${out_t} *output, 
            __global const ${in1_t} *arg1, 
            __global const ${in2_t} *arg2
        ) {
            const size_t start = get_global_id(1)*get_global_size(0), 
                         offset = get_global_id(0);
            ${in1_t} f1 = arg1[start+offset];
            ${in2_t} f2 = arg2[offset];
            ${out_t} product;
            #if ${complex}
            product = (${out_t})(f1.x*f2.x - f1.y*f2.y, f1.x*f2.y + f1.y*f2.x);
            #else
            product = f1*f2;
            #endif
            output[start+offset] = product;
        }
    """
    @staticmethod
    def variant(in1_dtype, in2_dtype):
        in1_ctype = ctype(in1_dtype)
        in2_ctype = ctype(in2_dtype)
        out_dtype = ( # different precisions must not be mixed
            np.float32 if (in1_dtype == np.float32) and (in2_dtype == np.float32) else
            np.float64 if (in1_dtype == np.float64) and (in2_dtype == np.float64) else
            np.complex64 if (in1_dtype == np.complex64) or (in2_dtype == np.complex64) else
            np.complex128
        )
        is_complex = ((in1_dtype == np.complex64) and (in2_dtype == np.complex64) or
                      (in1_dtype == np.complex128) and (in2_dtype == np.complex128))
        return dict(
            suffix='{}_{}'.format(in1_ctype, in2_ctype), 
            complex=1 if is_complex else 0, 
            out_t=ctype(out_dtype), 
            in1_t=in1_ctype, 
            in2_t=in2_ctype,
            out_dtype=out_dtype
        )
    
    variants = [
        (np.float32, np.float32), (np.float32, np.complex64), 
        (np.complex64, np.float32), (np.complex64, np.complex64),
        (np.float64, np.float64), (np.float64, np.complex128),
        (np.complex128, np.float64), (np.complex128, np.complex128)
    ]

    def __init__(self, context):
        # compile real/complex combinations from the same code
        variants = [
            self.variant(in1_dtype, in2_dtype)
            for in1_dtype, in2_dtype in self.variants
        ]
        code = ''.join(Template(self.code).render(**kws) for kws in variants)
        self.prg = cl.Program(context, code).build()
    
    def __call__(self, queue, in1, in2, out=None):
        '''
        Perform broadcast multiplication in1 with in2.
        
        Input
        -----
        queue
            OpenCL command queue the operation is performed in. 
        in1: cl.Array or np.ndarray, dtype=float32 or complex64, ndim>=1
            First factor.
        in2: cl.Array or np.ndarray, dtype=float32 or complex64, ndim=1
            Second factor.
        out: `cl.Array`, optional
            Output array. If unset, a new `cl.Array` is allocated.
        
        Returns
        -------
        output: cl.Array, dtype=float32 or complex64
            Elementwise product of in1 and in2.
        '''
        # check inputs & transfer to device
        if ((in1.dtype, in2.dtype) not in self.variants):
            raise TypeError('in1 {}, in2 {} type combination not supported.'
                            .format(in1.dtype, in2.dtype))
        if in1.shape[-in2.ndim:] != in2.shape[-in2.ndim:]:
            raise ValueError('Trailing dimension(s) of in1 and in2 must agree.')
        if not isinstance(in1, cl.array.Array):
            in1 = cl.array.to_device(queue, in1, async=True)
        if not isinstance(in2, cl.array.Array):
            in2 = cl.array.to_device(queue, in2, async=True)
        # select kernel and output type
        variant = self.variant(in1.dtype, in2.dtype)
        function = getattr(self.prg, 'bcast_mul_{suffix}'.format(**variant))
        otype = variant['out_dtype']
        # allocate or check output array
        if out is None:
            out = cl.array.Array(queue, in1.shape, otype)
        else:
            if out.shape != in1.shape:
                raise ValueError('out must have the same shape as in1.')
            if out.dtype != otype:
                raise ValueError('out must have type {}'.format(otype))
        # run computation
        global_size = (np.prod(in1.shape[-in2.ndim:]), np.prod((1,)+in1.shape[:-in2.ndim]))
        function(queue, global_size, None, out.data, in1.data, in2.data)
        return out


class DDC(BroadcastMultiply):
    def __init__(self, ctx, if_freq=None, scale=1.):
        super(DDC, self).__init__(ctx)
        self.if_freq = if_freq
        self.scale = scale
        self.ddc_wf = None
        self.ddc_wf_args = None

    def __call__(self, queue, in1, if_freq=None, scale=None, **kwargs):
        '''
        Perform digital down-conversion of the input.
        
        Input
        -----
        queue
            OpenCL command queue the operation is performed in. 
        in1: cl.Array or np.ndarray, dtype=float32 or complex64, ndim>=1
            Input samples.
        if_freq:
            Intermediate frequency. 1.0 is pi radians/sample.
        
        Returns
        -------
        output: cl.Array, dtype=complex64
        '''
        if if_freq is None:
           if_freq = self.if_freq
        if if_freq is None:
            raise ValueError('if_freq is undefined.')
        if scale is None:
            scale = self.scale
        
        ddc_wf_args = (scale, if_freq, in1.shape, in1.dtype)
        if (self.ddc_wf is None) or (ddc_wf_args != self.ddc_wf_args):
            ddc_wf = scale*np.exp(-1j*np.pi*if_freq*np.arange(in1.shape[-2]))[:,None]
            ddc_wf = np.tile(ddc_wf, (1,in1.shape[-1]))
            if (in1.dtype == np.float32) or (in1.dtype == np.complex64):
                ddc_wf = ddc_wf.astype(np.complex64)
            self.ddc_wf = cl.array.to_device(queue, ddc_wf, async=True)
            self.if_freq = if_freq
            self.ddc_wf_args = (scale, if_freq, in1.shape, in1.dtype)
        return super(DDC, self).__call__(queue, in1, self.ddc_wf, **kwargs)


class Convolve(object):
    code = """
        /**
         * 1d convolution arg1*arg2 with broadcasting of arg2
         * returns only samples that do not depend on padding
         *
         * global_size (channels, arg1_samples-arg2_samples+decimation)//decimation, blocks)
         * arg1_samples, arg2_samples and channels can be defined at build-time
         * if channels is set to 1, 2 or 4 at build time, the code is vectorized and the first 
         * element of global size must be 1
         */
        #if ${arg1_samples}
            #define ARG1_SAMPLES ${arg1_samples}
        #endif
        #if ${arg2_samples}
            #define ARG2_SAMPLES ${arg2_samples}
        #endif
        #if ${channels} == 2
            #define rtype ${rtype}2
            #define ctype ${ctype}4
        #elif ${channels} == 4
            #define rtype ${rtype}4
            #define ctype ${ctype}8
        #else
            #define rtype ${rtype}
            #define ctype ${ctype}2
        #endif

        #if (${channels} == 1) || (${channels} == 2) || (${channels} == 4)
            __kernel void convolve_valid_cr(
                __global ctype *output, 
                __global const ctype *arg1, 
                __constant const rtype *arg2,
                #if !${arg1_samples}
                const int ARG1_SAMPLES,
                #endif
                #if !${arg2_samples}
                const int ARG2_SAMPLES, 
                #endif
                const int decimation
            ) {
                const size_t block=get_global_id(2), out_sample=get_global_id(1);
                const size_t out_samples=get_global_size(1);

                output += block*out_samples + out_sample;
                arg1 += block*ARG1_SAMPLES + out_sample*decimation;

                ctype sample = 0.;
                size_t arg1_idx = 0, arg2_idx = ARG2_SAMPLES;
                do {
                    arg2_idx--;
                    #if ${channels} == 1
                        sample += arg1[arg1_idx] * arg2[arg2_idx];
                    #elif ${channels} == 2
                        sample += arg1[arg1_idx] * arg2[arg2_idx].xxyy;
                    #elif ${channels} == 4
                        sample += arg1[arg1_idx] * arg2[arg2_idx].xxyyzzww;
                    #else
                    #error Unsupported number of channels.
                    #endif
                    arg1_idx++;
                } while (arg2_idx != 0);
                *output = sample;
            }
        #else
            __kernel void convolve_valid_cr(
                __global ctype *output, 
                __global const ctype *arg1, 
                __constant const rtype *arg2,
                #if !${arg1_samples}
                const int ARG1_SAMPLES,
                #endif
                #if !${arg2_samples}
                const int ARG2_SAMPLES, 
                #endif
                const int decimation
            ) {
                #if ${channels}
                #define CHANNELS ${channels}
                #else
                const size_t CHANNELS = get_global_size(0);
                #endif
                const size_t block=get_global_id(2), out_sample=get_global_id(1), channel=get_global_id(0);
                const size_t out_samples=get_global_size(1);

                output += (block*out_samples + out_sample)*CHANNELS + channel;
                arg1 += (block*ARG1_SAMPLES + out_sample*decimation)*CHANNELS + channel;

                ctype sample = 0.;
                size_t arg1_idx = 0, arg2_idx = ARG2_SAMPLES*CHANNELS + channel;
                do {
                    arg2_idx -= CHANNELS;
                    sample += arg1[arg1_idx] * arg2[arg2_idx];
                    arg1_idx += CHANNELS;
                } while (arg2_idx != channel);
                *output = sample;
            }
        #endif
    """

    def __init__(self, context):
        self.context = context

    def compile(self, **kwargs):
        cached = all(getattr(self, k, None) == v for k, v in kwargs.items())
        if cached:
            return
        code = Template(self.code).render(**kwargs)
        self.prg = cl.Program(self.context, code).build()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, queue, arg1, arg2, decimation, out=None, **kwargs):
        '''
        Calculate the convolution of arg1 with arg2 over the second-to-last 
        dimension. Returns only elements that do not depend on zero-padding
        (mode='valid' in numpy)

        Input
        -----
        queue
            OpenCL command queue the operation is performed in. 
        arg1: cl.Array or np.ndarray, dtype=complex64, ndim>=3
            First factor.
        arg2: cl.Array or np.ndarray, dtype=float32, ndim=2
            Second factor.
        decimation: `int`
            Return every decimation'th sample.
        out: `cl.Array`, optional
            Output array. If unset, a new `cl.Array` is allocated.
        local_size: `tuple` of `int`
            Local work size for the convolution kernel. (3d)
        
        Returns
        -------
        output: cl.Array, dtype=complex64
            Convolution of in1 and in2.

        Performance
        -----------
        * arg2.shape[0] == 2**n is faster than 2**n+1
        * decimation == 1 is faster than 2 <= decimation < 16
        * use FFT convolution instead if arg2.shape[0] >> log2(arg1.shape[-2])
        '''
        if arg1.dtype not in (np.complex64, np.complex128):
            raise TypeError('arg1 must have dtype complex64 or complex128.')
        ctype = arg1.dtype
        if arg2.dtype not in (np.float32, np.float64):
            raise TypeError('arg2 must have dtype float32 or float64.')
        if ((arg1.dtype == np.complex64) and (arg2.dtype == np.float64) or 
            (arg1.dtype == np.complex128) and (arg2.dtype == np.float32)):
            raise TypeError('arg1 and arg2 must have the same precision.')
        rtype = arg2.dtype

        if arg1.ndim < 2:
            raise TypeError('arg1 must be a three-dimensional array')
        if (arg2.ndim != 2):
            raise TypeError('arg2 must be a two-dimensional array')
        if arg1.shape[-1] != arg2.shape[-1]:
            raise ValueError('shapes {} and {} are incompatible.'.format(arg1.shape, arg2.shape))
        if not isinstance(arg1, cl.array.Array):
            arg1 = cl.array.to_device(queue, arg1, async=True)
        if not isinstance(arg2, cl.array.Array):
            arg2 = cl.array.to_device(queue, arg2, async=True)
        
        arg1_samples = arg1.shape[-2]
        arg2_samples = arg2.shape[0]
        channels = arg1.shape[-1]
        typemap = {np.float32: 'float', np.complex64: 'float',
                   np.float64: 'double', np.complex128: 'double'}
        for _rtype, rctype in typemap.items():
            if _rtype == rtype:
                break
        for _ctype, cctype in typemap.items():
            if _ctype == ctype:
                break
        self.compile(arg1_samples=arg1_samples, arg2_samples=arg2_samples, 
                     channels=channels, rtype=rctype, ctype=cctype)

        out_shape = arg1.shape[:-2] + ((arg1_samples-arg2_samples+decimation)//decimation, channels)
        blocks = np.prod(out_shape[:-2]) if arg1.ndim > 2 else 1
        if out is None:
            out = cl.array.Array(queue, out_shape, dtype=ctype)
        else:
            if (out.dtype != ctype) or (out.shape != out_shape):
                raise ValueError('out must have dtype={} and shape={}'
                                 .format(ctype, out_shape))
        if arg1.shape[-1] in [1, 2, 4]:
            global_size = (1, out_shape[-2], blocks)
        else:
            global_size = (out_shape[-1], out_shape[-2], blocks)
        kernel = self.prg.convolve_valid_cr
        if 'local_size' in kwargs:
            local_size = kwargs['local_size']
        else:
            max_wgs = kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, queue.device)
            max_dims = queue.device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)
            local_size = (min(max_dims[0], global_size[0]),)
            local_size += (min(max_dims[1], global_size[1], max_wgs//local_size[0]),)
            local_size += (min(max_dims[2], global_size[2], max_wgs//np.prod(local_size)),)
        local_size = None
        kernel(queue, global_size, local_size, out.data, arg1.data, arg2.data, 
               np.int32(decimation))
        return out


class Mean(object):
    code = """
    __kernel void sum_${otype}_${itype}(
        global ${otype} *output, 
        global ${itype} *input, 
        const unsigned int count,
        const unsigned int mean
    ) {
        // sum over first axis of a two-dimensional array
        size_t offset=get_global_id(0), stride=get_global_size(0);

        ${otype} sum = 0.;
        for(size_t idx = offset; idx < stride*count; idx += stride) {
            sum += convert_${otype}(input[idx]);
        }
        if(mean) {
            output[offset] = sum / convert_float(count);
        } else {
            output[offset] = sum;
        }
    }    
    """

    # No float3 because of alignment issues
    types = ()
    ftypes = {1:'float', 2:'float2', 4:'float4', 8:'float8', 16:'float16'}
    dtypes = {1:'double', 2:'double2', 4:'double4', 8:'double8', 16:'double16'}

    def __init__(self, context):
        code = ''
        for otype, itype in [('float', 'float'), ('double', 'float'), 
                             ('double', 'double')]:
            for width in ['', '2']:
                render_kws = dict(otype=otype+width, itype=itype+width)
                code += Template(self.code).render(**render_kws)
        self.prg = cl.Program(context, code).build()

    def __call__(self, cq, in1, ndim, mean=True, out=None):
        '''
        Calculate the mean over the the leading dimensions of `in1` such that
        `ndim` dimensions remain.

        Input
        -----
        cq: `cl.CommandQueue`
            OpenCL command queue the operation is performed in. 
        in1: cl.Array or np.ndarray, dtype=complex64
            Data array.
        ndim: `int`
            Reduce the input array to `ndim` number of dimensions.
        out: `cl.Array`, optional
            Output array. If unset, a new `cl.Array` is allocated.
        
        Returns
        -------
        out: cl.Array, dtype=complex64
            in1.mean(axis=tuple(range(in1.ndim-ndim)))
        '''
        # check input(s)
        typemap = {np.float32: 'float', np.complex64: 'float2',
                   np.float64: 'double', np.complex128: 'double2'}
        for dtype, ctype in typemap.items():
            if in1.dtype == dtype:
                break
        else:
            raise TypeError('dtype of in1 is not supported.')
        if isinstance(in1, np.ndarray):
            in1 = cl.array.to_device(cq, in1, async=True)
        # check output(s)
        out_shape = in1.shape[::-1][:ndim][::-1]
        if out is None:
            out = cl.array.Array(cq, out_shape, in1.dtype)
        else:
            if (out.dtype != in1.dtype) or (out.shape != out_shape):
                raise ValueError('out must have dtype={} and shape={}'
                                 .format(in1.dtype, out_shape))
        # run computation
        stride = np.prod((1,) + out_shape)#
        count = np.prod((1,) + in1.shape[::-1][ndim:])
        #print (out_shape, stride, count)
        kernel = getattr(self.prg, 'sum_{}_{}'.format(ctype, ctype))
        kernel(cq, (stride,), None, out.data, in1.data, np.uint32(count), 
               np.uint32(mean))
        return out


class Sum(object):
    code = """
    #include <pyopencl-complex.h>
    
    __kernel void sum_${otype}_${itype}(
        global ${otype} *output, 
        global ${itype} *input, 
        unsigned int count
    ) {
        // sum over first axis of a two-dimensional array
        size_t offset=get_global_id(0), stride=get_global_size(0);

        ${otype} b = ${neutral};
        for(size_t idx = offset; idx < stride*count; idx += stride) {
            ${otype} a = convert_${otype}(input[idx]);
            ${statement};
        }
        output[offset] = b;
    }    
    """

    def __init__(self, context, out_dtype, in_dtype, statement=None):
        '''
        Calculate the sum over the the leading dimensions of an array.
        
        Input
        -----
        context: `cl.Context`
            OpenCL context to to compile for
        out_dtype: `np.dtype`
            Data type of the output array. Array elements are cast to this 
            type prior to adding.
        in_dtype: `np.dtype`
            Data type of the input array.
        statement: `str`, optional
            OpenCL statement that updates the current value `b` of the result
            with the new input a. Defaults to 'b += a' for floating point 
            types and 'b = add_set(b, a)' for integer types.
        '''
        self._out_dtype = out_dtype
        self._out_ctype = ((cl.tools.dtype_to_ctype(out_dtype().real.dtype)+'2')
                           if np.issubdtype(out_dtype, np.complex) 
                           else cl.tools.dtype_to_ctype(out_dtype))
        self._in_dtype = in_dtype
        self._in_ctype = ((cl.tools.dtype_to_ctype(in_dtype().real.dtype)+'2')
                          if np.issubdtype(in_dtype, np.complex) 
                          else cl.tools.dtype_to_ctype(in_dtype))
        #self._in_ctype = cl.tools.dtype_to_ctype(in_dtype)
        if np.issubdtype(out_dtype, np.integer):
            statement = 'b = add_sat(b, a)'
            neutral = '0'
        elif np.issubdtype(out_dtype, np.floating):
            statement = 'b += a'
            neutral = '0.'
        elif np.issubdtype(out_dtype, np.complex):
            statement = 'b += a'
            #neutral = '{}_new(0., 0.)'.format(self._out_ctype[:-2])
            neutral = '0.'
        else:
            raise TypeError('out_dtype {} not supported.'.format(out_dtype))
    
        render_kws = dict(otype=self._out_ctype, itype=self._in_ctype,
                          neutral=neutral, statement=statement)
        code = Template(self.code).render(**render_kws)
        self.prg = cl.Program(context, code).build()

    def __call__(self, cq, in1, ndim, out=None):
        '''
        Calculate the mean over the the leading dimensions of `in1` such that
        `ndim` dimensions remain.

        Input
        -----
        cq: `cl.CommandQueue`
            OpenCL command queue the operation is performed in. 
        in1: cl.Array or np.ndarray
            Data array.
        ndim: `int`
            Reduce the input array to `ndim` number of dimensions.
        out: `cl.Array`, optional
            Output array. If unset, a new `cl.Array` is allocated.
        
        Returns
        -------
        out: cl.Array
            in1.sum(axis=tuple(range(in1.ndim-ndim)))
        '''
        # check input(s)
        if (in1.dtype != self._in_dtype):
            raise TypeError('dtype of in1 must be {}.'.format(self._in_dtype))
        if isinstance(in1, np.ndarray):
            in1 = cl.array.to_device(cq, in1, async=True)
        # check output(s)
        out_shape = in1.shape[::-1][:ndim][::-1]
        if out is None:
            out = cl.array.Array(cq, out_shape, in1.dtype)
        else:
            if (out.dtype != self._out_dtype) or (out.shape != out_shape):
                raise ValueError('out must have dtype={} and shape={}.'
                                 .format(self._out_dtype, out_shape))
        # run computation
        stride = int(np.prod(out_shape))
        count = int(np.prod(in1.shape[::-1][ndim:]))
        #print (out_shape, stride, count)
        kernel = getattr(self.prg, 'sum_{}_{}'.format(self._out_ctype, self._in_ctype))
        kernel(cq, (stride,), None, out.data, in1.data, np.uint32(count))
        return out


class TimeIntegrate(object):
    code = """
    // kernel for one channel
    __kernel void time_integrate_${otype}_${itype}_1(
        global ${otype} *output, 
        global ${itype} *input,
        unsigned int start,
        unsigned int stop
    ) {
        size_t idx = get_global_id(0);
        size_t sizex = get_global_size(0);

        size_t samples = get_global_size(1);

        ${otype} channel1_data = 0;
        for(int i = start; i < stop; i++)
        {
            channel1_data = channel1_data + convert_${otype}(input[idx*samples + i]);
        }
        output[idx] = channel1_data;
    }
        // kernel for two channel
        __kernel void time_integrate_${otype}_${itype}_2(
        global ${otype} *output, 
        global ${itype} *input,
        unsigned int start,
        unsigned int stop
    ) {
        size_t idx = get_global_id(0);
        size_t sizex = get_global_size(0);

        size_t samples = get_global_size(1);

        ${otype} channel1_data = 0;
        ${otype} channel2_data = 0;
        
        for(int i = start; i < stop*2; i = i+2)
        {
            channel1_data = channel1_data + convert_${otype}(input[idx*samples*2 + i]);
            channel2_data = channel2_data + convert_${otype}(input[idx*samples*2 + i + 1]);
        }   
        output[2*idx] = channel1_data;
        output[2*idx+1] = channel2_data;    
    }    
    """

    # No float3 because of alignment issues
    types = ()
    ftypes = {1:'float', 2:'float2', 4:'float4', 8:'float8', 16:'float16'}
    dtypes = {1:'double', 2:'double2', 4:'double4', 8:'double8', 16:'double16'}

    def __init__(self, context):
        code = ''
        for otype, itype in [('float', 'float'), ('double', 'float'), 
                            ('double', 'double')]:
            for width in ['', '2']:
                render_kws = dict(otype=otype+width, itype=itype+width)
                code += Template(self.code).render(**render_kws)
        self.prg = cl.Program(context, code).build()

    def __call__(self, cq, in1, out=None, start=0, stop=-1):
        '''
        Calculate the mean over the the third dimension of `in1`. If

        Input
        -----
        cq: `cl.CommandQueue`
            OpenCL command queue the operation is performed in. 
        in1: cl.Array or np.ndarray, dtype=complex64
            Data array.
        out: `cl.Array`, optional
            Output array. If unset, a new `cl.Array` is allocated.
        
        Returns
        -------
        out: cl.Array, dtype=complex64
            in1.mean(axis=2)
        '''
        # check input(s)
        typemap = {np.float32: 'float', np.complex64: 'float2',
                   np.float64: 'double', np.complex128: 'double2'}
        for dtype, ctype in typemap.items():
            if in1.dtype == dtype:
                break
        original_shape = in1.shape
        out_shape = (np.prod(in1.shape[:-2]), in1.shape[-1])
        samples = in1.shape[-2]
        channels = in1.shape[-1]
        # Reshapes the input array to 2 dimensions. Everything but iterations is flattenned.
        in1 = in1.reshape(np.prod(in1.shape[:-2]),in1.shape[-2]*in1.shape[-1])
        if isinstance(in1, np.ndarray):
            in1 = cl.array.to_device(cq, in1, async=True)
        # else:
        #     raise TypeError('dtype of in1 is not supported.')
        # check output(s)
        # removes the second last
        if out is None:
            out = cl.array.Array(cq, out_shape, in1.dtype)
        else:
            if (out.dtype != in1.dtype) or (out.shape != out_shape):
                raise ValueError('out must have dtype={} and shape={}'
                                 .format(in1.dtype, out_shape))
        # run computation
        #print (out_shape, stride, count)
        kernel = getattr(self.prg, 'time_integrate_{}_{}_{}'.format(ctype, ctype, channels))
        start = start%samples
        stop = stop%samples
        assert stop > start, "Must integrate a natural number of points."
        kernel(cq, (out_shape[0], samples), None, out.data, in1.data, np.uint32(), np.uint32(stop%samples + 1))
        return out.get().reshape(np.delete(original_shape, -2))

class MarkerExtractionForML(object):
    ''' Copy of Short to Float class without converting to float.'''

    code = """
        #define itype short
        #define btype char

        __kernel void extract(
            __global itype *output,
            __global const itype *input,
            const unsigned int blocklen,
            const unsigned int bits
        ) {
            const size_t in_idx = get_global_id(1)*blocklen + get_global_id(0), 
                         out_idx = get_global_id(1)*get_global_size(0) + get_global_id(0);
            itype sample = (get_global_id(0) < blocklen) ? input[in_idx] : 0;
            // sign extension
            if(bits != 16) {
                const itype sign_bit = 1 << (bits - 1);
                sample = (sample & (sign_bit - 1)) - (sample & sign_bit);
            }
            output[out_idx] = (get_global_id(0) < blocklen) ? sample : 0.;
        }

        __kernel void extract_with_markers(
            __global itype *output,
            __global btype *marker1,
            __global btype *marker2, 
            __global const itype *input, 
            const unsigned int blocklen,
            const unsigned int bits
        ) {
            const size_t in_idx = get_global_id(1)*blocklen + get_global_id(0), 
                         out_idx = get_global_id(1)*get_global_size(0) + get_global_id(0);
            itype sample = (get_global_id(0) < blocklen) ? input[in_idx] : 0;
            marker1[out_idx] = sample & 0x8000 ? 1 : 0;
            marker2[out_idx] = sample & 0x4000 ? 1 : 0;
            //sample = (sample << 2) >> 2; // sign extension (sadly optimized away by the compiler)
            //sample = (sample & 0x3fff) | ((sample << 1) & (1<<14)) | ((sample << 2) & (1<<15)); // sign extension
            // sign extension
            const itype sign_bit = 1 << (bits - 1);
            sample = (sample & (sign_bit - 1)) - (sample & sign_bit);
            output[out_idx] = sample;
        }
    """

    def __init__(self, context):
        code = ''.join(Template(self.code).render())
        self.prg = cl.Program(context, code).build()

    def __call__(self, queue, in1, markers=False, bits=16, padding=0, out=None):
        '''
        Convert input samples to floating point numbers and separate the marker bits.
        
        Input
        -----
        queue
            OpenCL command queue the operation is performed in. 
        in1: `pyopencl.array.Array` or `np.ndarray`
            Input samples. Must be a 2d numpy ndarray or pyopencl Array of 16bit 
            integers.
        markers: `bool`
            If True, the two most significant bits are considered markers.
        padding: `int`
            Number of bytes of zero padding to add to each block of samples.
        out: `pyopencl.array.Array`, optional
            Output array if markers is False, tuple of output and marker arrays
            if markers is True. Use pre-allocated arrays to speed up processing.
            
        Return
        ------
        out: pyopencl.array.Array, shape=(in1.shape[0], blocklen)
            Sample values as float
        marker1, marker2: cl.Array, dtype=uint8, same shape as out
            uint8(1) for each set and uint8(0) for each unset marker bit
        '''
        btype = np.uint8
        # check input and transfer to device
        if (in1.dtype != np.int16) and (in1.dtype != np.uint16):
            raise TypeError('in1 must be a vector of 16bit integers.')
        if in1.ndim < 1:
            raise ValueError('in1 must be a vector.')
        if isinstance(in1, np.ndarray):
            in1 = cl.array.to_device(queue, in1, async=True)
        # check output(s)
        out_shape = in1.shape[:-1] + (in1.shape[-1]+padding,)
        if out is not None:
            for arr in out if markers else (out,):
                if arr.shape != out_shape:
                    raise ValueError('shape of outputs must be {}'.format(out_shape))
        global_size = (out_shape[-1], np.prod((1,)+out_shape[:-1]))
        if markers:
            # allocate result arrays
            if out is None:
                out = cl.array.Array(queue, out_shape, in1.dtype)
                marker1 = cl.array.Array(queue, out.shape, btype)
                marker2 = cl.array.Array(queue, out.shape, btype)
            else:
                out, marker1, marker2 = out
            # run computation
            function = getattr(self.prg, 'extract_with_markers')
            function(
                queue, global_size, None, 
                out.data, marker1.data, marker2.data, in1.data, np.uint32(in1.shape[-1], np.uint32(bits))
            )
            return out, marker1, marker2
        else:
            # allocate result arrays
            if out is None:
                out = cl.array.Array(queue, out_shape, in1.dtype)
            # run computation
            function = getattr(self.prg, 'extract')
            function(queue, global_size, None, out.data, in1.data, 
                     np.uint32(in1.shape[-1]), np.uint32(bits))
            return out
