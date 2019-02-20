from .ADCProcessor import (
    Unpacker, DigitalDownconversion, Filter, Mean, Synchronizer, TvMode, 
    Instrument, ManualParameter, vals
)
from . import dsp
import numpy as np
import pyopencl as cl

def get_cl_context():
    '''get an opencl context for a gpu device'''
    for platform in cl.get_platforms():
        try:
            return cl.Context(
                dev_type=cl.device_type.GPU, 
                properties=[(cl.context_properties.PLATFORM, platform)]
            )
        except cl.cffi_cl.RuntimeError:
            pass
    raise RuntimeError('Unable to create an opencl context for a GPU device.')

def get_cl_queue(cl_context):
    '''create a new opencl command queue'''
    return cl.CommandQueue(cl_context)


class UnpackerGPU(Unpacker):
    '''Offload floating-point math to GPU, perform marker extraction on CPU'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._to_float_gpu = dsp.ShortToFloat(self.parent.gpu_context)
        
    def to_float(self, raw_samples, out=None):
        return self._to_float_gpu(
            self.parent.gpu_queue, raw_samples, out=out, 
            markers=False, bits=16-self.markers.get())

class DigitalDownconversionGPU(DigitalDownconversion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ddc = dsp.DDC(self.parent.gpu_context)
        
    def __call__(self, float_samples, out=None):
        return self.ddc(self.parent.gpu_queue, float_samples, out=out, 
                        if_freq=self.intermediate_frequency.get(), 
                        scale=self.scale.get())

class FilterGPU(Filter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convolve = dsp.Convolve(self.parent.gpu_context)
        
    def __call__(self, samples, out=None):
        if self.mode.get() != 'valid':
            raise NotImplementedError('Only convolution mode "valid" is supported.')
        coeffs = self.coefficients.get().astype(np.float32)
        coeffs = np.tile(coeffs[:,None], (1, samples.shape[-1]))
        return self.convolve(self.parent.gpu_queue, samples, coeffs, 
                             self.decimation.get(), out=out)

class MeanGPU(Mean):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = dsp.Mean(self.parent.gpu_context)

    def __call__(self, data, ndim, out=None):
        return self.mean(self.parent.gpu_queue, data, ndim=ndim, out=out).get()

class SumGPU(Mean):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sum = dsp.Sum(self.parent.gpu_context, np.complex64, np.complex64)

    def __call__(self, data, ndim, out=None):
        return self.sum(self.parent.gpu_queue, data, ndim=ndim, out=out).get()

class TvModeGPU(TvMode):
    def __init__(self, name, cl_context=None, cl_queue=None, *args, **kwargs):
        Instrument.__init__(self, name, *args, **kwargs)
        # create GPU resources
        if cl_context is None:
            if cl_queue is not None:
                cl_context = cl_queue.context
            else:
                cl_context = get_cl_context()
        if cl_queue is None:
            cl_queue = get_cl_queue(cl_context)
        self.gpu_context = cl_context
        self.gpu_queue = cl_queue
        
        self.add_parameter('segments', ManualParameter, 
                           label='Number of segments. Zero for automatic.', 
                           vals=vals.Numbers(min_value=0), default_value=0)
        self.segments.set(0)
        
        self.add_submodule('unpacker', UnpackerGPU(self, 'unpacker'))
        self.add_submodule('ddc', DigitalDownconversionGPU(self, 'ddc'))
        self.add_submodule('filter', FilterGPU(self, 'filter'))
        self.add_submodule('sync', Synchronizer(self, 'sync'))
        #self.add_submodule('mean', MeanGPU(self, 'mean'))
        self.add_submodule('sum', SumGPU(self, 'sum'))
        self.connect_message()

    def generate(self, source, mean=False):
        '''Process blocks of samples provided by source.

        Each block is processed through the following processing pipeline:
        * `unpacker` splits analog and digital data in each sample and converts 
          the analog data to float.
        * `ddc` mixes the analog data with scale*exp(-2 pi if_freq t).
        * `filter` applies an FIR filter on the ddc output with decimation
        * `sync` finds synchronization markers in the digital data and uses it 
          to synchronize the segment index with an external waveform generator.
          The block is reshaped to (iteration, segment, sample, channel).
        * `sum` takes the sum or mean over the iteration axis of the reshaped 
          data.

        Arguments:
            source: `iterable`
                An iterable (typically a generator) that provides multiplexed
                analog and digital data. 
                Each block is indexed by (segment, sample, channel)
            mean: `bool`, default False
                If True, return the block mean. 
                If False, return the block sum and the number of iterations.
        Returns:
            analog_mean: `np.ndarray`
            (analog_sum, iterations): (`np.ndarray`, int)
                The mean or sum of the block over the iteration axis.
                

        Note:
            The current implementation requires all blocks generated by source 
            to have the same shape.
        '''
        # using single buffering, with buffer allocation by the submodules
        analog = None
        digital = None
        analog_ddc = None
        analog_fir = None

        segments = self.segments()

        for block in source:
            # separate analog and digital data
            #analog, digital = self.unpacker(block, out=(analog, digital))
            analog = self.unpacker.to_float(block, out=analog)
            # process samples through the queue
            analog_ddc = self.ddc(analog, out=analog_ddc)
            analog_fir = self.filter(analog_ddc, out=analog_fir)
            analog_math = analog_fir # TODO: channel maths goes here

            # find first segment and number of segments
            if segments == 1:
                first_segment = 0
            else:
                digital = self.unpacker.pack_markers(block, out=digital)
                if digital is None:
                    raise ValueError('Enable marker extraction in unpacker '
                                     'to use auto segments.')
                marked_segments = self.sync(digital)
                if len(marked_segments) == 0:
                    raise ValueError('No synchronization markers received.')
                first_segment = marked_segments[0]
                if (segments == 0) or (segments is None):
                    if len(marked_segments) < 2:
                        raise ValueError('Need at least two synchronization '
                                         'markers to determine the number of '
                                         'segments.')
                    segments = marked_segments[1] - marked_segments[0]
            if np.prod(analog.shape[:-2]) < segments:
                raise ValueError('Each input block must contain at least '
                                 '`segments` ({}) segments.'.format(segments))
            # truncate & reshape to (iteration, segment, sample, channel)
            repetitions = analog.shape[0] // segments
            analog_trunc = (analog_math[:repetitions*segments,...]
                            .reshape((repetitions, segments)+analog_math.shape[1:]))
            analog_sum = self.sum(analog_trunc, analog_trunc.ndim-1)
            if first_segment:
                analog_sum = np.roll(analog_sum, -first_segment, axis=0)
            if mean:
                yield analog_sum / repetitions
            else:
                yield analog_sum, repetitions

    def __call__(self, source):
        '''Processes all blocks generated by source through `generate`.
        
        Processes all blocks generated by source through `generate` and 
        returns the weighted mean. See `generate` for a more details.

        Returns:
            analog_mean: `np.ndarray`
        '''
        repetitions_total = 0
        analog_total = None
        for analog_sum, repetitions in self.generate(source):
            repetitions_total += repetitions
            if analog_total is None:
                analog_total = analog_sum
            else:
                analog_total += analog_sum
        return analog_total / repetitions_total
