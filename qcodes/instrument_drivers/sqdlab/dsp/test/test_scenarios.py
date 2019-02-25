import pytest
from pytest import fixture, raises, mark, skip, xfail

import numpy as np
import scipy.signal
import pyopencl as cl
import pyopencl.array

from ... import dsp
from ...dsp import ShortToFloat, BroadcastMultiply, Convolve, DDC, Mean
from ...dsp.fft import FFT, FFTConvolve

import reikna
import reikna.algorithms
from reikna.core import Type, Annotation, Parameter

#
# GENERAL SETUP
#
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

@fixture(scope='module')
def api():
    return reikna.cluda.ocl_api()

@fixture(scope='module')
def thread(api, cq):
    return api.Thread(cq)

#
# INPUT DATA
#
@fixture
def real_t():
    return np.float32
    #return np.float64

@fixture
def complex_t():
    return np.complex64
    #return np.complex128

@fixture
def channels():
    return 1

@fixture
def blocksize():
    return 1<<25 # 32MS

@fixture
def dsamples():
    return 0
    return 128
    return -128

@fixture
def samples(dsamples):
    # if the computation involves convolution, the number of samples must 
    # be adjusted according to the convolution mode for best performance
    # the number of output samples should be a power of two
    return 4096 + dsamples

@fixture
def shape(blocksize, samples, channels):
    return (blocksize//samples//channels, samples, channels)

@fixture
def markers():
    return False

@fixture
def padding():
    return 0

@fixture
def decimation():
    return 1

@fixture
def adc_analog(shape):
    # 14bit analog data
    return np.random.randint(-1<<13, 1<<13, size=shape, dtype=np.int16)

@fixture
def adc_markers(shape):
    # two digital marker channels as uint8 arrays
    return (np.random.randint(0, 1, size=shape, dtype=np.uint8),
            np.random.randint(0, 2, size=shape, dtype=np.uint8))

@fixture
def adc_short(adc_analog, adc_markers, markers):
    # samples as returned by the digitizer
    # [15:14] markers, [13:0] analog data
    if not markers:
        return adc_analog
    else:
        return ((adc_markers[0].astype(np.uint16) << 15) | 
                (adc_markers[1].astype(np.uint16) << 14) |
                (adc_analog & 0x3fff))

@fixture
def adc_float(adc_analog, real_t):
    return adc_analog.astype(real_t)

@fixture
def if_freq():
    return 0.1

@fixture
def ddc_output(adc_float, if_freq, complex_t):
    ts = np.arange(adc_float.shape[-2])
    ddc_wf = np.exp(-1j*np.pi*if_freq*ts).astype(complex_t)[:,None]
    return adc_float * ddc_wf

@fixture
def fir_coeffs(shape, real_t):
    return np.random.rand(129,shape[-1]).astype(real_t)

def convolve(in1, in2, dtype, axis, mode='full'):
    # helper functions
    pad_tuple = lambda shape, ndim: (1,)*(ndim-len(shape)) + shape
    pad_shape = lambda arr, ndim: arr.reshape(pad_tuple(arr.shape, ndim))
    # equalise number of dimensions, make 'axis' the last
    ndim = max(in1.ndim, in2.ndim)
    in1 = np.rollaxis(pad_shape(in1, ndim), axis, ndim)
    in2 = np.rollaxis(pad_shape(in2, ndim), axis, ndim)
    # determine result shape
    shape = tuple(max(s1, s2) for s1, s2 in zip(in1.shape, in2.shape))[:-1]
    if mode == 'valid':
        shape += (in1.shape[-1]-in2.shape[-1]+1,)
    elif mode == 'same':
        shape += (max(in1.shape[-1], in2.shape[-1]),)
    elif mode == 'full':
        shape += (in1.shape[-1]+in2.shape[-1]-1,)
    else:
        raise ValueError('unsupported mode {}'.format(mode))
    # broadcast arrays except for last dimension
    in1 = np.broadcast_to(in1, shape[:-1] + (in1.shape[-1],))
    in2 = np.broadcast_to(in2, shape[:-1] + (in2.shape[-1],))
    # evaluate 1d convolutions
    result = np.empty(shape, dtype)
    for idx in np.ndindex(*shape[:-1]):
        result[idx] = np.convolve(in1[idx], in2[idx], mode=mode)
    # put 'axis' back in place
    return np.rollaxis(result, -1, axis)

@fixture
def decimation():
    return 1

@fixture
def fir_output(ddc_output, fir_coeffs, decimation):
    dtype = ddc_output.dtype if np.iscomplexobj(ddc_output) else fir_coeffs.dtype
    return convolve(ddc_output, fir_coeffs, dtype, axis=-2, mode='valid')

def mean(in1, ndim):
    return in1.mean(axis=tuple(range(in1.ndim-ndim)))

@fixture
def tvmode_output(fir_output):
    return mean(fir_output, ndim=fir_output.ndim-1)

@mark.parametrize('dsamples', [128])
def test_tvmode_steps(ctx, cq, markers, padding, if_freq, fir_coeffs, decimation, 
                      adc_short, adc_float, ddc_output, fir_output, tvmode_output):
    '''Check the outputs of all intermediate steps'''
    stf = ShortToFloat(ctx)
    float_data = stf(cq, adc_short, markers=markers, padding=padding)
    assert np.allclose(float_data.get(), adc_float, atol=1e-6)

    ddc = DDC(ctx)
    ddc_data = ddc(cq, float_data, if_freq=if_freq)
    assert np.allclose(ddc_data.get(), ddc_output, atol=1e-6)

    convolve = Convolve(ctx)
    fir_data = convolve(cq, ddc_data, fir_coeffs, decimation)
    assert np.allclose(fir_data.get(), fir_output, rtol=2e-3, atol=1e-3)

    mean = Mean(ctx)
    tvmode_data = mean(cq, fir_data, ndim=adc_short.ndim-1)
    assert tvmode_data.shape == tvmode_output.shape
    assert np.allclose(tvmode_data.get(), tvmode_output, rtol=1e-3, atol=1e-3)


@fixture
def reps():
    '''number of repetitions for speed tests'''
    return 3

@mark.parametrize('dsamples', [128]) # 128
def test_tvmode_simple(ctx, cq, adc_short, markers, padding, if_freq, 
                       fir_coeffs, decimation, reps, tvmode_output):
    '''Test speed without preallocated buffers.'''
    stf = ShortToFloat(ctx)
    ddc = DDC(ctx)
    convolve = Convolve(ctx)
    mean = Mean(ctx)

    results = []
    for _ in range(reps):
        float_data = stf(cq, adc_short, markers=markers, padding=padding)
        ddc_data = ddc(cq, float_data, if_freq=if_freq)
        fir_data = convolve(cq, ddc_data, fir_coeffs, decimation)
        tvmode_data = mean(cq, fir_data, ndim=adc_short.ndim-1)
        results.append(tvmode_data.get())
    good = [np.allclose(result, tvmode_output, rtol=1e-4, atol=1e-4) for result in results]
    assert np.all(good)

@fixture
def fft_output(adc_float):
    return np.fft.fft(adc_float, axis=-2)

@fixture
def spec_output(fft_output):
    return mean(abs(fft_output), ndim=fft_output.ndim-1)

@mark.parametrize('dsamples', [0])
def test_fft_simple(ctx, cq, thread, real_t, complex_t, shape, adc_short, 
                    markers, padding, reps, spec_output):
    stf = ShortToFloat(ctx)
    float_data = cl.array.Array(cq, shape, real_t) # no padding
    fft_ = FFT(Type(real_t, shape), axes=(-2,))
    fft = fft_.compile(thread)
    fft_data = cl.array.Array(cq, shape, complex_t)
    predicate = reikna.algorithms.predicate_sum(real_t)
    #sum_ = reikna.algorithms.Reduce(float_data, predicate, axes=(0,))
    #sum = sum_.compile(thread)
    #spec_data = cl.array.Array(cq, shape[1:], real_t)
    mean = Mean(ctx)

    results = []
    for _ in range(reps):
        stf(cq, adc_short, out=float_data, markers=markers, padding=padding)
        fft(fft_data, float_data, False)
        spec_data = mean(cq, abs(fft_data), ndim=adc_short.ndim-1)
        #sum(spec_data, abs(fft_data)) # / real_t(fft_data.shape[0])
        results.append(spec_data.get())
    good = [np.allclose(result, spec_output, rtol=1e-3, atol=1e-3) 
            for result in results]
    assert np.all(good)

#
# Test with data from hardware
#
def multiple_trigger_fifo_acquisition(self, segments, samples, blocksize, posttrigger=None):
    '''
    Multiple recording acquisition with background DMA data transfer.
    
    Input
    -----
    segments: `int`
        Total number of segments measured
    blocksize: `int`
        Number of segments per block dispatched for data processing
    samples: `int`, min 32 step 16
        Number of samples per segment
    posttrigger: `int`, range 16 to samples-16 in steps of 16
        Number of samples recorded after each trigger. Defaults to samples-16. 
        
    Yields
    ------
    data: `np.ndarray`, dtype=int16, shape=(blocksize, samples, channels)
        Acquired data as a numpy array. If blocksize does not divide segments, the
        last block will have shape (segments%blocksize, samples, channels).
        
    Notes
    -----
    * Returns a generator expression that can be used like an iterator in a for loop.
    * The internal buffer is twice the size of data.
    '''
    import pyspcm
    import ctypes as ct
    import time
    
    # set card to multiple recording to fifo mode
    self.card_mode(pyspcm.SPC_REC_FIFO_MULTI)

    # * SPC_LOOPS is the total number of segments, can be 0=inf
    # * SPC_MEMSIZE is ignored
    # * SPC_SEGMENTSIZE is # of samples/segment
    # * SPC_TRIGGERCOUNTER returns # of acquisitions (uint_48)
    self.total_segments(segments)
    self.segment_size(samples)
    if posttrigger is None:
        posttrigger = samples - 16
    self.posttrigger_memory_size(posttrigger)
    numch = bin(self.enable_channels()).count("1")
    if not numch:
        raise RuntimeError('No channels are enabled.')

    # setup software buffer(s)
    nbuffers = 2
    buffer_samples = blocksize * samples * numch
    buffer_bytes = 2*buffer_samples
    data_buffers = (ct.c_int16*buffer_samples*nbuffers)()

    self._def_transfer64bit(
        pyspcm.SPCM_BUF_DATA, pyspcm.SPCM_DIR_CARDTOPC, buffer_bytes, ct.byref(data_buffers), 
        0, buffer_bytes*nbuffers
    )

    try:
        # start data acquisition & transfer
        self.general_command(pyspcm.M2CMD_CARD_START | pyspcm.M2CMD_CARD_ENABLETRIGGER)
        self.general_command(pyspcm.M2CMD_DATA_STARTDMA)

        # data transfer
        # SPC_M2STATUS: 
        # * M2STAT_DATA_OVERRUN indicates a buffer overrun on the card
        self.general_command(pyspcm.M2CMD_DATA_WAITDMA)
        status = 0
        abort = False
        overrun = False
        while True:
            if overrun:
                # user_length must be queried before status for overrun detection
                user_length = self.user_available_length()
            status = self.card_status()
            #print('status=0x{:02x}, available={}k'.format(status, self.user_available_length()//1024))
            #if (not overrun) and (status & (pyspcm.M2STAT_DATA_OVERRUN | pyspcm.M2STAT_CARD_READY)):
            if status & (pyspcm.M2STAT_DATA_OVERRUN):
                # M2STAT_DATA_OVERRUN is only returned once and may be lost
                # if WAITDMA if used or the user queries the card status
                # M2STAT_CARD_READY indicates that the card has stopped
                # which may be the result of a overrun or end of the measurement
                logging.error('A buffer overrun occured on the digitizer. Aborting.')
                overrun = True
            if status & pyspcm.M2STAT_DATA_BLOCKREADY:
                # user_length must be queried after status for this
                # clip shape[0] of the output array to block_size
                user_length = min(self.user_available_length(), buffer_bytes)
                user_position = self.user_available_position()
                data_buffer = ct.cast(ct.addressof(data_buffers) + user_position, 
                                      ct.POINTER(user_length//2*ct.c_int16))
                data = np.frombuffer(data_buffer.contents, dtype=ct.c_int16)
                # abort acquisition if the user sends False
                abort = yield data.reshape(user_length//samples//numch//2, samples, numch)
                # free yielded portion of the buffer
                self._set_param32bit(pyspcm.SPC_DATA_AVAIL_CARD_LEN, user_length)
            else:
                # in normal operation, user_length != 0 always coincides with a new block
                # if a buffer overrun occurs, no new block is reported and 
                # user_length != 0 marks the last (incomplete) block
                if overrun and (user_length != 0):
                    break
                time.sleep(10e-3)
            if abort or (status & pyspcm.M2STAT_DATA_END): 
                break
            #if self.general_command(pyspcm.M2CMD_DATA_WAITDMA):
            #    raise RuntimeError('M4i WAITDMA returned an error.')
    finally:
        # stop transfer before invalidating buffer
        self._stop_acquisition()


@fixture#(scope='module')
def adc_device():
    import qcodes
    from qcodes.instrument_drivers.Spectrum.M4i import M4i    
    return M4i('digitizer')
    
@fixture
def adc_physical(adc_device, shape, reps):
    import pyspcm as spcm
    print(shape)
    blocksize, samples, channels = shape
    channel_list = [spcm.CHANNEL0, spcm.CHANNEL1, spcm.CHANNEL2, spcm.CHANNEL3]
    adc_device.enable_channels(np.bitwise_or.reduce(channel_list[:channels]))
    for channel in range(channels):
        adc_device.set_channel_settings(channel, mV_range=1000., input_path=1, 
                                        termination=0, coupling=0)
    adc_device.set_ext0_OR_trigger_settings(spcm.SPC_TM_HIGH, level0=500, 
                                            termination=0, coupling=0)
    return multiple_trigger_fifo_acquisition(adc_device, segments=reps*blocksize, 
                                             samples=samples, blocksize=blocksize)

@mark.parametrize('channels', [2])
@mark.parametrize('samples', [(1<<13)+128])
@mark.parametrize('dsamples', [128]) # 128
@mark.parametrize('reps', [32])
@mark.parametrize('blocksize', [((1<<13)+128)*(1<<11)*2])
def test_tvmode_physical(ctx, cq, adc_device, adc_physical, markers, padding, 
                         if_freq, fir_coeffs, decimation, reps, shape):
    '''Test speed without preallocated buffers.'''
    stf = ShortToFloat(ctx)
    ddc = DDC(ctx)
    convolve = Convolve(ctx)
    mean = Mean(ctx)

    adc_data = cl.array.Array(cq, shape, np.int16)
    float_data = cl.array.Array(cq, shape, np.float32)
    ddc_data = cl.array.Array(cq, shape, np.complex64)
    fir_coeffs = cl.array.to_device(cq, fir_coeffs, async=True)
    fir_data_shape = shape[:-2] + (shape[-2]-fir_coeffs.shape[0]+1, shape[-1])
    fir_data = cl.array.Array(cq, fir_data_shape, np.complex64)
    out_data = cl.array.Array(cq, fir_data_shape[1:], np.complex64)

    results = []
    for idx, adc_short in enumerate(adc_physical):
        print('index={}, card buffer={:.3f}'.format(idx, adc_device.buffer_fill_size()/1000))
        adc_data.set(adc_short, async=True)
        stf(cq, adc_data, markers=markers, padding=padding, out=float_data)
        ddc(cq, float_data, if_freq=if_freq, out=ddc_data)
        convolve(cq, ddc_data, fir_coeffs, decimation, out=fir_data)
        mean(cq, fir_data, adc_short.ndim-1, out=out_data)
        results.append(out_data.get())
        
        #float_data = stf(cq, adc_short, markers=markers, padding=padding)
        #ddc_data = ddc(cq, float_data, if_freq=if_freq)
        #fir_data = convolve(cq, ddc_data, fir_coeffs, decimation)
        #tvmode_data = mean(cq, fir_data, ndim=adc_short.ndim-1)
        #results.append(tvmode_data.get())

