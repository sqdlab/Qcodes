from .ADCProcessor import *
from . import dsp
import numpy as np
import pyopencl as cl

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
                        if_freq=self.intermediate_frequency.get())

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
    
class TvModeGPU(TvMode):
    def __init__(self, name, cl_context, cl_queue, *args, **kwargs):
        Instrument.__init__(self, name, *args, **kwargs)
        self.gpu_context = cl_context
        self.gpu_queue = cl_queue
        
        self.add_parameter('segments', ManualParameter, 
                           label='Number of segments. Zero for automatic.', 
                           vals=vals.Numbers(min_value=0))
        
        self.add_submodule('unpacker', UnpackerGPU(self, 'unpacker'))
        self.add_submodule('ddc', DigitalDownconversionGPU(self, 'ddc'))
        self.add_submodule('filter', FilterGPU(self, 'filter'))
        self.add_submodule('sync', Synchronizer(self, 'sync'))
        self.add_submodule('mean', MeanGPU(self, 'mean'))

    def __call__(self, source):
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
            analog_mean = self.mean(analog_trunc, analog_trunc.ndim-1)
            if first_segment:
                analog_mean = np.roll(analog_mean, -first_segment, axis=0)
            yield analog_mean

