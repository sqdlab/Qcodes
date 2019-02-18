import pytest
from pytest import fixture, raises, mark, skip, xfail

import timeit
import numpy as np
import scipy.signal
import pyopencl as cl
import pyopencl.array

from dsp import ShortToFloat, BroadcastMultiply, Convolve, DDC, Mean, Sum

@fixture(scope='module')
def ctx():
    for platform in cl.get_platforms():
        try:
            return cl.Context(
                dev_type=cl.device_type.GPU, 
                properties=[(cl.context_properties.PLATFORM, platform)]
            )
        except cl.cffi_cl.RuntimeError:
            pass
    raise RuntimeError('Unable to create an opencl context for a GPU device.')

@fixture(scope='module')
def cq(ctx):
    return cl.CommandQueue(ctx)

@fixture(scope='module')
def cq2(ctx):
    return cl.CommandQueue(ctx)

# generate data for ShortToFloat and the integration tests
@fixture
def adc(shape, markers=False):
    '''2bit markers, 14bit analog data'''
    #arr = np.arange(-1<<15, 1<<15, 1<<8, dtype=np.int16)
    arr = np.random.randint(-1<<15, (1<<15)-1, size=shape, dtype=np.int16)
    if markers:
        # convert numbers to 14bit and make lsbs markers instead of msbs
        # (does not make much sense for random numbers!)
        arr = ((arr >> 2) & 0x3fff) | (arr << 14)
    return arr.reshape(shape)

def random(shape, dtype):
    if np.issubdtype(dtype, np.floating):
        return np.random.rand(*shape).astype(dtype)
    elif np.issubdtype(dtype, np.complex):
        return (np.random.rand(*shape) + 1j*np.random.rand(*shape)).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        iinfo = np.iinfo(dtype)
        min_ = max(-2<<31, iinfo.min)
        max_ = min((2<<31)-1, iinfo.max)
        return np.random.random_integers(min_/4, max_/4, shape).astype(dtype)
    else:
        raise ValueError('unable to generate random numbers for type {}'
                         .format(dtype))
    return arr

@fixture(params=[False, True])
def pass_out(request):
    return request.param


class TestShortToFloat(object):
    @fixture
    def stf(self, ctx):
        return ShortToFloat(ctx)

    @fixture(params=[0, 4, 16, 64, 256])
    def padding(self, request):
        return request.param

    @fixture(params=[False, True])
    def markers(self, request):
        return request.param

    @fixture(params=[(np.int16, (256,)),
                     (np.int16, (16,16)),
                     (np.int16, (4,4,16)), 
                     (np.int16, (2,2,4,16)), 
                     (np.uint16, (16,16))],
             ids=['int16-256', 'int16-16x16', 'int16-4x4x16', 'int16-2x2x4x16', 'uint16-16x16'])
    def cpu_arr(self, request, markers):
        '''Input data as a host array, format as provided by the digitizer.'''
        dtype, shape = request.param
        arr = np.arange(-1<<15, 1<<15, 1<<8, dtype=dtype)
        if markers:
            # convert numbers to 14bit and make lsbs markers instead of msbs
            arr = ((arr >> 2) & 0x3fff) | (arr << 14)
        return arr.reshape(shape)

    @fixture
    def gpu_arr(self, cq, cpu_arr):
        '''Input data as a device array.'''
        gpu_arr = cl.array.Array(cq, cpu_arr.shape, cpu_arr.dtype)
        gpu_arr.set(cpu_arr)
        return gpu_arr
    
    @fixture(params=['cpu', 'gpu'])
    def arr(self, request, cpu_arr, gpu_arr):
        '''Input data either as a host or device array.'''
        if request.param == 'cpu':
            return cpu_arr
        else:
            return gpu_arr

    @fixture
    def reference(self, cpu_arr, padding, markers, dtype):
        '''Perform the computation on the CPU.'''
        pad_arr = np.zeros(cpu_arr.shape[:-1] + (padding,), dtype=np.int16)
        arr = np.concatenate((cpu_arr, pad_arr), axis=-1)
        if markers:
            return (((arr & 0x3fff) | np.where(arr & 0x2000, 0xc000, 0x0000)).astype(np.int16).astype(dtype),
                    ((arr >> 15) & 0x1).astype(np.uint8), 
                    ((arr >> 14) & 0x1).astype(np.uint8))
        else:
            return arr.astype(np.int16).astype(dtype)

    @mark.parametrize('dtype', [np.float32, np.float64])
    def test_results(self, stf, cq, reference, arr, markers, padding, pass_out, dtype):
        '''Check output against same computation in numpy.'''
        # run
        if pass_out:
            if markers:
                output = cl.array.Array(cq, reference[0].shape, dtype)
                marker1 = cl.array.Array(cq, reference[0].shape, np.int8)
                marker2  = cl.array.Array(cq, reference[0].shape, np.int8)
                out = (output, marker1, marker2)
            else:
                out = cl.array.Array(cq, reference.shape, dtype)
            result = stf(cq, arr, markers=markers, padding=padding, out=out)
        else:
            result = stf(cq, arr, markers=markers, padding=padding)
        # analyze
        if markers:
            output, marker1, marker2 = result
            ref_output, ref_marker1, ref_marker2 = reference
            assert np.allclose(output.get(), ref_output)
            assert np.all(marker1.get() == ref_marker1)
            assert np.all(marker2.get() == ref_marker2)
        else:
            assert np.allclose(result.get(), reference)

    @mark.parametrize('dtype,fail', [(np.uint8, True), (np.int8, True), (np.uint16, False), (np.int16, False), (np.uint32, True), (np.int32, True), (np.float16, True), (np.float32, True), (np.float64, True), (np.complex64, True), (np.complex128, True)])
    def test_dtype(self, stf, cq, dtype, fail):
        '''Should fail for any type other than int16 and uint16.'''
        arr = np.zeros((16,16), dtype)
        if fail:
            with raises(TypeError):
                stf(cq, arr)
        else:
            stf(cq, arr)


class TestBroadcastMultiply(object):
    @fixture
    def tmul(self, ctx):
        return BroadcastMultiply(ctx)

    @fixture(params=[(16,), (4,16), (4,4,16), (4,4,4,16)])
    def shape(self, request):
        return request.param

    def arr(self, shape, dtype):
        if dtype == np.float32:
            return np.random.rand(*shape).astype(dtype)
        else:
            return (np.random.rand(*shape) + 1j*np.random.rand(*shape)).astype(dtype)
        return arr

    @fixture(params=[np.float32, np.complex64])
    def arg1(self, request, shape):
        return self.arr(shape, request.param)

    @fixture(params=[(1,np.float32), (1,np.complex64), (2,np.float32), (2,np.complex64)])
    def arg2(self, request, shape):
        ndim, dtype = request.param
        return self.arr(shape[-ndim:], dtype)

    @mark.parametrize('device_arg1', [False, True], ids=['host', 'device'])
    @mark.parametrize('device_arg2', [False, True], ids=['host', 'device'])
    def test_result(self, cq, tmul, arg1, arg2, device_arg1, device_arg2, pass_out):
        reference = arg1 * arg2
        if device_arg1:
            arg1 = cl.array.to_device(cq, arg1)
        if device_arg2:
            arg2 = cl.array.to_device(cq, arg2)
        if pass_out:
            out = cl.array.Array(cq, reference.shape, reference.dtype) 
        else:
            out = None
        result = tmul(cq, arg1, arg2, out=out)
        assert result.dtype == reference.dtype
        assert result.shape == reference.shape
        assert np.allclose(result.get(), reference)

    @mark.parametrize('dtype1', [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128])
    @mark.parametrize('dtype2', [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128])
    def test_type_mismatch(self, cq, tmul, dtype1, dtype2):
        arg1 = cl.array.to_device(cq, np.linspace(0, 1, dtype=dtype1))
        arg2 = np.linspace(0, 1, dtype=dtype2)
        if (dtype1 in [np.float32, np.complex64]) and (dtype2 in [np.float32, np.complex64]):
            tmul(cq, arg1, arg2)
        else:
            with raises(TypeError):
                tmul(cq, arg1, arg2)

    def test_shape_mismatch(self, cq, tmul, arg1):
        arg1 = cl.array.to_device(cq, arg1)
        arg2 = np.random.random(arg1.shape[-1]-1).astype(np.float32)
        with raises(ValueError):
            tmul(cq, arg1, arg2)


class TestDDC(object):
    @fixture
    def ddc(self, ctx):
        return DDC(ctx)

    @fixture
    def if_freq(self):
        return 0.1

    @fixture
    def arg1(self, if_freq):
        phase = np.linspace(0, np.pi, 8)[:,None]
        times = np.arange(120)[None,:]
        omega = if_freq*np.pi
        return np.cos(omega*times+phase).astype(np.float32)

    @fixture
    def reference(self, arg1, if_freq):
        ts = np.arange(arg1.shape[-2])
        ddc_wf = np.exp(-1j*np.pi*if_freq*ts).astype(np.complex64)[:,None]
        return arg1 * ddc_wf

    @mark.parametrize('reps', (1, 3))
    def test(self, cq, ddc, arg1, if_freq, reference, reps):
        for _ in range(reps):
            result = ddc(cq, arg1, if_freq=if_freq)
            assert np.allclose(result.get(), reference, rtol=1e-4, atol=1e-6)


class TestConvolve(object):
    @fixture
    #def convolve(self, ctx, arg1_shape, arg2_shape):
    #    print(dict(arg1_shape=arg1_shape, arg2_shape=arg2_shape, channels=arg1_shape[2]))
    #    return Convolve(ctx, arg1_shape=arg1_shape, arg2_shape=arg2_shape, channels=arg1_shape[2])
    def convolve(self, ctx):
        return Convolve(ctx)

    @fixture(params=[(1024, 1), (1024, 2), (1024, 3), (1024, 4), 
                     (64, 1024, 1), (64, 1024, 2), (64, 1024, 3), (64, 1024, 4), 
                     (4, 32, 1024, 1), (4, 32, 1024, 2), (4, 32, 1024, 3), (4, 32, 1024, 4),
                     #(512, 4096, 1), (512, 4096, 2), (512, 4096, 3), (512, 4096, 4), 
                     ],
             ids=['1024,1', '1024,2', '1024,3', '1024,4',
                  '64,1024,1', '64,1024,2', '64,1024,3', '64,1024,4',
                  '4,32,1024,1', '4,32,1024,2', '4,32,1024,3', '4,32,1024,4',
                  #'512,4096,1', '512,4096,2', '512,4096,3', '512,4096,4',
                  ])
    def arg1_shape(self, request):
        return request.param

    @fixture(params=[128, 129])#[32, 33, 64, 65, 128, 129])
    def arg2_shape(self, request, arg1_shape):
        return (request.param, arg1_shape[-1])

    @fixture(params=[1, 4, 16, 64])
    def decimation(self, request):
        return request.param

    @fixture
    def arg1(self, arg1_shape):
        return (np.random.rand(*arg1_shape) + 1j*np.random.rand(*arg1_shape)).astype(np.complex64)
        #return (np.ones(arg1_shape) + 1j*np.ones(arg1_shape)).astype(np.complex64)

    @fixture
    def arg2(self, arg2_shape):
        return np.random.rand(*arg2_shape).astype(np.float32)
        #return np.linspace(0, 1, arg2_shape[0])[:,None].astype(np.float32)

    @fixture
    def reference(self, arg1, arg2, decimation):
        result = np.zeros(arg1.shape[:-2] + ((arg1.shape[-2]-arg2.shape[0]+decimation)//decimation, arg1.shape[-1]), dtype=np.complex64)
        for block in np.ndindex(*arg1.shape[:-2]):
            for channel in range(arg1.shape[-1]):
                result[block][:,channel] = scipy.signal.fftconvolve(arg1[block][:,channel], arg2[:,channel], mode='valid')[::decimation]
        return result

    @mark.parametrize('local_size', [True, False])
    def test(self, cq, convolve, arg1, arg2, reference, decimation, pass_out, local_size):
        arg1 = cl.array.to_device(cq, arg1)
        arg2 = cl.array.to_device(cq, arg2)
        out = cl.array.Array(cq, reference.shape, reference.dtype) if pass_out else None
        kwargs = dict(local_size=None) if local_size else dict()
        result = convolve(cq, arg1, arg2, decimation, out=out, **kwargs)
        assert result.shape == reference.shape
        print(result.get())
        print(reference)
        print (result.get() - reference)
        assert np.allclose(result.get(), reference)

    @mark.skip
    def test_speed(self, cq, convolve, arg1_shape, arg2_shape):
        arg1 = (np.random.rand(*arg1_shape) + 1j*np.random.rand(*arg1_shape)).astype(np.complex64)
        arg2 = np.random.rand(*arg2_shape).astype(np.float32)
        arg1 = cl.array.to_device(cq, arg1)
        arg2 = cl.array.to_device(cq, arg2)
        cq.finish()
        duration = timeit.repeat('convolve(cq, arg1, arg2, decimation=1); cq.finish()', 
                                 number=1, repeat=3, globals=locals())
        raise Exception('average runtime was {:.1f}ms'.format(1e3*np.mean(duration)))
        

class TestMean(object):
    @fixture
    def mean(self, ctx):
        return Mean(ctx)

    @fixture(params=[np.float32, np.complex64, np.float64, np.complex128])
    def dtype(self, request):
        return request.param

    def run_average(self, cq, mean, shape, ndim, pass_out, dtype, **kwargs):
        in1 = random(shape, dtype)
        ref = in1.mean(axis=tuple(range(in1.ndim-ndim)))
        if pass_out:
            #out_type = (np.float64 if (dtype == np.float32) or (dtype == np.float64) else
            #            np.complex128)
            out = cl.array.Array(cq, ref.shape, dtype)
            mean(cq, in1, ndim, out=out, **kwargs)
        else:
            out = mean(cq, in1, ndim, **kwargs)
        #print(in1)
        #print(out)
        #print(ref)
        #print(out.get()-ref)
        assert(np.allclose(out.get(), ref, atol=1e-6))

    @mark.skip('obsolete')
    @mark.parametrize('max_vector_width', [1, 2, 3, 4, 8, 16])
    @mark.parametrize('shape', [(16,1), (16,2), (16,3), (16,4), (16,5), (16,8), (16,12), (16,16),
                                (1<<12,1<<12), (1<<12,(1<<12)-128), (1<<12,(1<<12)+128), (1<<12,3*(1<<12))])
    @mark.parametrize('dtype', [np.float32, np.complex64])
    def test_vectorization(self, cq, mean, shape, dtype, max_vector_width):
        self.run_average(cq, mean, shape, ndim=1, pass_out=False, dtype=dtype, 
                         max_vector_width=max_vector_width)

    @mark.parametrize('shape,ndim', [[(16,), 0], [(16,), 1], 
                                     [(8,16), 0], [(8,16), 1], [(8,16), 2], 
                                     [(4,8,16), 0], [(4,8,16), 1], [(4,8,16), 2], [(4,8,16), 3]])
    @mark.parametrize('dtype', [np.float32, np.complex64, np.float64, np.complex128])
    def test_shapes_and_ndim(self, cq, mean, shape, ndim, pass_out, dtype):
        self.run_average(cq, mean, shape, ndim, pass_out, dtype=dtype)

    #@mark.parametrize('local_size', [(256,), (128,), (64,), (32,), (16,), (8,), (4,), (2,), (1,)])
    #@mark.parametrize('shape', [(1<<12,1<<12)])
    #def test_local_size(self, cq, mean, shape, dtype, local_size):
    #    self.run_average(cq, mean, shape, ndim=1, pass_out=False, dtype=dtype, local_size=local_size)

    
class TestSum(object):
    @fixture
    def sum(self, ctx, dtype, in_dtype):
        return Sum(ctx, dtype, in_dtype)

    @fixture(params=[np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, 
                 np.int64, np.uint64, np.float32, np.float64, np.complex64, 
                 np.complex128])
    def dtype(self, request):
        return request.param
        
    @fixture(params=[np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, 
                 np.int64, np.uint64, np.float32, np.float64, np.complex64, 
                 np.complex128])
    def in_dtype(self, request):
        return request.param

    @fixture
    def shape(self):
        return (64, 64)
        
    @fixture
    def ndim(self):
        return 1
        
    @fixture
    def input(self, shape, in_dtype):
        return random(shape, in_dtype)
        
    @fixture
    def reference(self, input, shape, dtype, in_dtype, ndim):
        if np.issubdtype(dtype, np.integer):
            # gpu saturates in each operation, we need to manually take the sum
            iinfo = np.iinfo(dtype)
            input = input.reshape((int(np.prod(shape[::-1][ndim:][::-1]))), 
                                   int(np.prod(shape[::-1][:ndim][::-1]))).T
            results = np.empty((input.shape[0],), dtype=dtype)
            for idx, chunk in enumerate(input):
                result = 0
                for item in chunk:
                    result = max(iinfo.min, min(iinfo.max, result + dtype(item)))
                results[idx] = result
            return results.reshape(shape[::-1][:ndim][::-1])
        else:
            return input.astype(dtype).sum(axis=tuple(range(input.ndim-ndim)))
            
    def test_result(self, cq, sum, ndim, input, reference):
        output = sum(cq, input, ndim)
        assert np.all(output.get() == reference)
    
class TestIntegration(object):
    @fixture
    def shape(self):
        #return (1<<14, 1, 4096-128, 1) # mul 45ms, convolve 100ms, mean 8ms
        #return (1<<14, 1, 4096, 1) # mul 9ms, convolve 1250ms, mean 9ms
        #return (1<<14, 1, 4096+128, 1) # mul 9ms, convolve 107ms, mean 9ms
        #return (1<<13, 1, 4096+128, 2) # mul 9ms, convolve 85ms, mean 6ms
        #return (1<<12, 1, 4096+128, 4) # mul 9ms, convolve 130ms, mean 5ms
        #return (1<<11, 1, 4096+128, 8) # mul 9ms, convolve 202ms, mean 6ms
        #return (1<<8, 1, 61, 4096+128, 2) # mul 17ms, convolve 167ms, mean 11ms
        #return (1<<16, 1, 4096+128, 1) # mul 36ms, convolve 446ms, mean 268ms
        return (1<<12, 1, 4096+128, 2) # runs in a reasonable time

    @staticmethod
    def adc(shape, if_freq, signal, noise, markers=False):
        '''
        Generate adc input that simulates a noisy readout signal generated by
        a Rabi type experiment.

        if_freq: `float`
         Carrier frequency of the input signal, 1=Nyquist frequency
        signal: `float`
            Signal amplitude
        noise: `float`
            Noise amplitude
        '''
        averages, segments, samples, channels = shape
        # for each waveform, determine the state of the qubit (True or False)
        excited_exp = (1-np.cos(3*np.pi*(np.arange(segments))/segments))/2
        excited_shot = excited_exp > np.random.rand(averages, segments, channels)
        # map the state to iq points
        ground = -(1+1j)/np.sqrt(2)
        excited = (1+1j)/np.sqrt(2)
        iq_shot = np.where(excited_shot, signal*excited, signal*ground)
        # modulate carrier with iq points
        carrier = np.exp(1j*np.pi*if_freq*np.arange(samples))
        waveforms = np.real(iq_shot[...,None]*carrier)
        # add noise
        if noise:
            waveforms += noise*(-1+2*np.random.rand(averages, segments, samples, channels))
        # convert to short
        scale = (2**13) if markers else (2**15)
        data = np.clip(scale*waveforms, -scale, scale-1).astype(np.int16)
        if markers:
            # marker 1 is the sequence start marker
            # marker 2 is at a fixed position
            m1_start = min(10, samples)
            m1_stop = min(m1_start+10, samples)
            data &= 0x3ffff
            data[:, 0, m1_start:m1_stop] |= 0x8000
            data[:, :, m1_start:m1_stop] |= 0x4000
        data = np.roll(data, np.random.randint(0, segments), 1)
        return data

    @fixture
    def fir_coeffs(self, shape, bandwidth=0.1):
        b = scipy.signal.firwin(129, [bandwidth]).astype(np.float32)
        return np.tile(b[:,None], [1,shape[-1]])

    @fixture
    def reps(self):
        return 3

    def test_tvmode_simple(self, ctx, cq, adc, fir_coeffs, reps, if_freq=0.1, markers=False, padding=0, decimation=1):
        '''No preallocated buffers.'''
        stf = ShortToFloat(ctx)
        ddc = DDC(ctx)
        convolve = Convolve(ctx)
        mean = Mean(ctx)

        for _ in range(reps):
            float_data = stf(cq, adc, markers=markers, padding=padding)
            ddc_data = ddc(cq, float_data, if_freq=if_freq)
            del float_data
            fir_data = convolve(cq, ddc_data, fir_coeffs, decimation)
            del ddc_data
            out_data = mean(cq, fir_data, ndim=adc.ndim-1)
            del fir_data
            out_data.get()
            del out_data            

    def test_tvmode_single_buffered(self, ctx, cq, adc, fir_coeffs, reps, if_freq=0.1, markers=False, padding=0, decimation=1):
        '''One thread, single buffering'''
        stf = ShortToFloat(ctx)
        ddc = DDC(ctx)
        convolve = Convolve(ctx)
        mean = Mean(ctx)

        adc_data = cl.array.Array(cq, adc.shape, adc.dtype)
        float_data = cl.array.Array(cq, adc_data.shape, np.float32) # with no padding
        ddc_data = cl.array.Array(cq, float_data.shape, np.complex64)
        fir_coeffs = cl.array.to_device(cq, fir_coeffs, async=True)
        fir_data_shape = adc.shape[:-2] + (adc.shape[-2]-fir_coeffs.shape[0]+1, adc.shape[-1])
        fir_data = cl.array.Array(cq, fir_data_shape, np.complex64)
        out_data = cl.array.Array(cq, fir_data_shape[1:], np.complex64)

        for _ in range(reps):
            adc_data.set(adc, async=True)
            stf(cq, adc_data, markers=markers, padding=padding, out=float_data)
            ddc(cq, float_data, if_freq=if_freq, out=ddc_data)
            convolve(cq, ddc_data, fir_coeffs, decimation, out=fir_data)
            mean(cq, fir_data, adc.ndim-1, out=out_data)
            out_data.get()

    def test_tvmode_double_buffered(self, ctx, cq, cq2, adc, fir_coeffs, reps, if_freq=0.1, markers=False, padding=0, decimation=1):
        '''Two threads, double buffering'''
        stf = ShortToFloat(ctx)
        ddc = DDC(ctx)
        convolve = Convolve(ctx)
        mean = Mean(ctx)

        cq1 = cq
        fir_data_shape = adc.shape[:-2] + (adc.shape[-2]-fir_coeffs.shape[0]+1, adc.shape[-1])
        adc_data1 = cl.array.Array(cq, adc.shape, adc.dtype)
        float_data1 = cl.array.Array(cq, adc.shape, np.float32) # with no padding
        ddc_data1 = cl.array.Array(cq, adc.shape, np.complex64)
        fir_coeffs1 = cl.array.to_device(cq, fir_coeffs, async=True)
        fir_data1 = cl.array.Array(cq, fir_data_shape, np.complex64)
        out_data1 = cl.array.Array(cq, fir_data_shape[1:], np.complex64)
        adc_data2 = cl.array.Array(cq2, adc.shape, adc.dtype)
        float_data2 = cl.array.Array(cq2, adc.shape, np.float32) # with no padding
        ddc_data2 = cl.array.Array(cq2, adc.shape, np.complex64)
        fir_coeffs2 = cl.array.to_device(cq2, fir_coeffs, async=True)
        fir_data2 = cl.array.Array(cq2, fir_data_shape, np.complex64)
        out_data2 = cl.array.Array(cq2, fir_data_shape[1:], np.complex64)

        for rep in range(reps):
            if rep % 2 == 0:
                cq, adc_data, float_data, ddc_data, fir_data, out_data = cq1, adc_data1, float_data1, ddc_data1, fir_data1, out_data1
            else:
                cq, adc_data, float_data, ddc_data, fir_data, out_data = cq2, adc_data2, float_data2, ddc_data2, fir_data2, out_data2
            if rep != 0:
                out_data.get()
            adc_data.set(adc, async=True)
            stf(cq, adc_data, markers=markers, padding=padding, out=float_data)
            ddc(cq, float_data, if_freq=if_freq, out=ddc_data)
            convolve(cq, ddc_data, fir_coeffs, decimation, out=fir_data)
            mean(cq, fir_data, adc.ndim-1, out=out_data)
        out_data.get()
