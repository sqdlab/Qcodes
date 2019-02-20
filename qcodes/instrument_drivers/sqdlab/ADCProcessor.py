import textwrap
import functools

from qcodes import (Instrument, InstrumentChannel, Parameter, ManualParameter, 
                    validators as vals)
import numpy as np
import scipy.signal
import numba


class Unpacker(InstrumentChannel):
    '''Separate analog and digital data, convert analog to float.
    
    Demultiplexes analog and digital data in raw samples received from an ADC,
    and converts the analog data into a floating-point format.

    Input samples are signed 16bit integers with the most significant bits
    replaced by digital data. Supported data formats are
        A15 A14 A13 A12 A11 A10 A09 A08 A07 A06 A05 A04 A03 A02 A01 A00
        D00 A14 A13 A12 A11 A10 A09 A08 A07 A06 A05 A04 A03 A02 A01 A00
        D01 D00 A13 A12 A11 A10 A09 A08 A07 A06 A05 A04 A03 A02 A01 A00
    The analog bits are split from the digital bits, sign extension is performed
    on the analog part, and the digital bits are returned as a separate uint16 
    array. If the raw data is (at least) 2d, the last axis is assumed to be the 
    channel index and the digital bits from all channels are combined into a 
    single element.
    '''

    def __init__(self, parent:Instrument, name:str, **kwargs):
        super().__init__(parent, name, **kwargs)
        self.add_parameter(
            'markers', ManualParameter, 
            vals=vals.Ints(0, 2), default_value=0, 
            docstring='Number of bits of digital data per sample.'
        )
        self.add_parameter(
            'out_dtype', ManualParameter, default_value=np.float32, 
            vals=vals.Enum(np.float16, np.float32, np.float64),
            docstring='Output floating point type for analog data.'
        )
        self.markers.set(0)
        self.out_dtype.set(np.float32)

    @staticmethod
    @functools.lru_cache()
    def sign_extension_factory(bits):
        '''Compile a sign extension function for int<bits>'''
        @numba.vectorize(['int16(int16)', 'int32(int32)'])
        def sign_extension(x):
            sign_bit = 1 << (bits - 1)
            return (x & (sign_bit - 1)) - (x & sign_bit)
        return sign_extension

    def to_float(self, raw_samples, out=None):
        dtype = self.out_dtype.get()
        markers = self.markers.get()
        if markers != 0:
            sign_extension = self.sign_extension_factory(16-markers)
            raw_samples = sign_extension(raw_samples, out=out)
        return raw_samples.astype(dtype)

    @staticmethod
    @functools.lru_cache()
    def pack_markers_factory(channels, markers, in_bits, out_bits=None):
        '''Make a function that packs markers from multiple channels, fast.

        Uses numba.njit to generate machine code for a packing function for 
        inputs with a fixed number of channels, markers and bit width, enabling
        the compiler to optimize as much as possible.

        Arguments:
            channels: `int`
                number of channels in the input data
            markers: `int`
                number of marker bits in input each sample
            in_bits: `int`
                input dtype is np.uint<in_bits> or np.int<in_bits>
            out_bits: `int`, optional
                output dtype is np.uint<out_bits>. Defaults to the smallest
                possible size.
        Returns:
            pack_markers: `function`
                A function that takes a 2d ndarray and packs the markers along
                the second axis into a single number for each sample.

        '''
        if out_bits is None:
            out_bits = [v for v in [8, 16, 32, 64] if v>channels*markers][0]
        locals_ = dict(np=np, numba=numba)
        template = '''
            @numba.njit()
            def pack_markers(digital, out=None):
                """Pack digital bits for each sample into a single number."""
                if digital.ndim != 2:
                    raise ValueError('Input must be 2d.')
                samples, channels = digital.shape
                if channels != {channels}:
                    raise ValueError('Expected {channels} channels.')
                mask = np.uint{in_bits}(-1) ^ ((1<<({in_bits}-{markers})) - 1)
                if out is None:
                    out = np.empty((samples,), np.uint{out_bits})
                else:
                    #out = out.ravel()
                    if out.size != samples:
                        raise ValueError('out has the wrong size.')
                    if out.itemsize != {out_bits}//8:
                        raise ValueError('out dtype must be uint{out_bits}.')
                for idx in range(samples):
                    out_elem = np.uint{out_bits}(0)
                    for channel in range({channels}):
                        out_elem |= ((digital[idx,channel] & mask) >> 
                                    ({in_bits}-{markers}*(channel+1)))
                    out[idx] = out_elem
                return out
            '''
        code = template.format(channels=channels, markers=markers, 
                               in_bits=in_bits, out_bits=out_bits)
        bytecode = compile(textwrap.dedent(code), '<string>', 'exec')
        exec(bytecode, locals_, locals_)
        return locals_['pack_markers']

    def pack_markers(self, raw_samples, out=None):
        markers = self.markers.get()
        if markers == 0:
            return None
        if raw_samples.ndim == 1:
            pack_markers = self.pack_markers_factory(1, markers, 16)
            return pack_markers(raw_samples[:,None])
        else:
            channels = raw_samples.shape[-1]
            pack_markers = self.pack_markers_factory(channels, markers, 16)
            if out is not None:
                out = out.ravel()
            packed_markers = pack_markers(raw_samples.reshape((-1,channels)), 
                                          out=out)
            return np.reshape(packed_markers, raw_samples.shape[:-1])

    def __call__(self, raw_samples, out=(None,None)):
        '''Separate analog and marker data, convert analog to float.
        
        Arguments:
            raw_samples: `np.ndarray`, dtype=int16
                Fixed-point samples with some bits replaced by digital data
            out: (ndarray, ndarray), optional
                `analog` and `digital` output buffers
        Returns:
            analog: `np.ndarray`, dtype=out_dtype
            digital: `np.ndarray`
        '''
        return (self.to_float(raw_samples, out=out[0]), 
                self.pack_markers(raw_samples, out=out[1]))



class DigitalDownconversion(InstrumentChannel):
    r'''Mix analog samples with a software-generated local oscillator sigal.

    Multiplies floating-point input samples with \exp(-1j \pi if_frequency i)
    '''
    def __init__(self, parent:Instrument, name:str, **kwargs):
        super().__init__(parent, name, **kwargs)
        self.add_parameter(
            'intermediate_frequency', ManualParameter, 
            vals=vals.Numbers(-1.0, 1.0), default_value=-0.5, 
            docstring='Intermediate frequency as a multiple of the Nyquist '
                      'frequency, equal to half the sample rate.')
        self.add_parameter(
            'scale', ManualParameter, vals=vals.Numbers(), default_value=1., 
            docstring='Global scaling factor for code-to-voltage conversion.'
        )

    def __call__(self, float_samples):
        '''Shift frequencies in the input samples by -intermediate_frequency.

        Arguments:
            float_samples: `np.ndarray`, real or complex floating-point dtype
                Must be at least 2d, indexed (sample, channel)
        Returns:
            complex_samples: `np.ndarray`, complex dtype
                float_samples * np.exp(-1j pi if_frequency sample)
        '''
        if float_samples.ndim < 2:
            raise ValueError('input samples must be at least two-dimensional')
        lo_waveform = np.exp(-1j*np.pi*self.intermediate_frequency.get()*
                             np.arange(float_samples.shape[-2]))*self.scale()
        if float_samples.dtype in [np.float16, np.float32]:
            lo_waveform = lo_waveform.astype(np.complex64)
        # broadcasting by element duplication is faster than numpy broadcasting
        lo_waveform = np.tile(lo_waveform[:,None], (1, float_samples.shape[-1]))
        return float_samples*lo_waveform



class Filter(InstrumentChannel):
    '''Apply a FIR filter to analog data, with optional decimation.

    Calculates the convolution of input samples with `coefficients`, and 
    returns every `decimation`th output sample. 
    '''
    def __init__(self, parent:Instrument, name:str, **kwargs):
        super().__init__(parent, name, **kwargs)
        self.add_parameter(
            'decimation', ManualParameter, vals=vals.Ints(1), initial_value=1, 
            docstring='The sample rate is reduced by the decimation factor '
                      'after filtering.'
        )
        self.add_parameter(
            'coefficients', ManualParameter, vals=vals.Arrays(),
            docstring='FIR filter coefficients'
        )
        self.add_parameter(
            'description', ManualParameter, vals=vals.Strings(),
            docstring='FIR filter description'
        )
        self.add_parameter(
            'mode', ManualParameter, 
            vals=vals.Enum('full', 'valid'), initial_value='full',
            docstring='Convolution mode. If `valid`, only points where the '
                      'filter and signal overlap completely are returned.'
        )
        self.coefficients.set(np.array([1.]))

    def set_default_filter(self, length=None):
        '''Set scipy.signal.decimate's default filter

        Set filter coefficients to a Hamming window. The number of taps are
        length+1 if length is given or 20*decimation+1 if omitted.
        Note that the filter is not automatically updated when decimation is 
        changed.
        '''
        decimation = self.decimation.get()
        if length is None:
            length = 2 * 10 * decimation
        if decimation == 1:
            self.coefficients.set(np.array([1.]))
            self.description.set('No filter')
        else:
            coeffs = scipy.signal.firwin(length+1, 1. / decimation, 
                                         window='hamming')
            self.coefficients.set(coeffs)
            self.description.set('Hamming(length={}, cutoff=1/{})'
                                .format(length, decimation))

    def __call__(self, samples):
        if self.mode.get() != 'full':
            raise NotImplementedError('Only convolution mode "full" is supported.')
        dtype = samples.dtype.type().real.dtype
        coefficients = self.coefficients().astype(dtype)
        return scipy.signal.upfirdn(
            coefficients, samples, 
            up=1, down=self.decimation(), axis=-2
        )



class Synchronizer(InstrumentChannel):
    '''Extract synchronization markers from a digital data stream.

    Analyzes a multiplexed digital data stream indexed by (segment, time)
    to find which segments are marked. Markers are assumed to indicate the
    start of a sequence of different experiments, with the whole sequence
    repeating to acquire statistics. The value of `method` decides how much
    effort is spent to gain information:
    * method='one' extracts only the first sequence marker.
    * method='two' extracts the first two markers, allowing the sequence length
        to be determined automatically.
    * method='all' extracts all markers, additionally allowing the detection of
        missing or spurious segments in each repetition.
    '''
    def __init__(self, parent:Instrument, name:str, **kwargs):
        super().__init__(parent, name, **kwargs)
        self.add_parameter(
            'method', ManualParameter, 
            vals=vals.Enum('one', 'two', 'all'), default_value='one', 
            docstring='Number of synchronization markers to analyze. '
                      '`one` uses the first marker found as the start marker, '
                      '`two` allows automatic segment count detection, '
                      '`all` allows missing trigger detection.'
        )
        self.add_parameter(
            'mask', ManualParameter, vals=vals.Ints(), default_value=0xffff,
            docstring='Bit mask to extract the desired digital input channel.'
        )
        self.method.set('one')
        self.mask.set(0xffff)

    @staticmethod
    @numba.njit
    def sync_first(data, blocksize, mask=0xffff):
        '''
        Find index of the first block containing the selected marker.
        
        Arguments:
            data: `ndarray` of integer dtype
                Array containing marker data. May be raw data from the ADC.
            mask: `int`
                Bit mask to extract markers from data.
            
        Returns:
            index: `int`
                Index of the first block contining the selected marker. 
        '''
        data = data.ravel()
        for offset in range(len(data)):
            if data[offset] & mask:
                return offset // blocksize
        return None

    @staticmethod
    @numba.njit
    def sync_segments(data, blocksize, mask=0xffff, limit=0):
        '''
        Find offsets of all blocks containing the selected marker.
        
        Input
        -----
        data: `ndarray` of uint16_t
            Raw data from the ADC
        blocksize: `int`
            Number of samples per block
        mask: `int`
            Bit mask to extract markers from data.
        limit: `int`
            Maximum number of offsets returned.
            
        Returns
        -------
        offsets: `list` of `int`
            Offsets of all blocks containing the selected marker.
        '''
        data = data.ravel()
        offsets = []
        for blockoffset in range(0, len(data), blocksize):
            if limit and (len(offsets) >= limit):
                break
            for offset in range(blockoffset, blockoffset+blocksize):
                if data[offset] & mask:
                    offsets.append(blockoffset // blocksize)
                    break
        return offsets

    def __call__(self, digital, blocksize=None):
        '''Extract synchronization markers.

        Arguments:
            digital: `ndarray` of an integer type
                Packed digital signal. Each element is a bitfield containing 
                the values of all channels sampled at the same time.
            blocksize: `int`, optional
                Number of samples per segment. Defaults to digital.shape[-1]
        Returns:
            index: `list` of `int`
                offset of the first marked segment when method='one'
                offsets of all marked segments when method='two' or 'all'.
                    If method='two', all other offsets are extrapolated.
                an empty list when no markers are found.
        '''
        method = self.method.get()
        mask = self.mask.get()
        if blocksize is None:
            blocksize = digital.shape[-1]
        if method == 'one':
            idx0 = self.sync_first(digital, blocksize, mask)
            if idx0 is None:
                return []
            else:
                return [idx0]
        elif (method == 'two'):
            idxs = self.sync_segments(digital, blocksize, mask, 2)
            if len(idxs) < 2:
                return idxs
            return range(idxs[0], np.prod(digital.shape[:-1]), idxs[1]-idxs[0])
        elif (method == 'all'):
            return self.sync_segments(digital, blocksize, mask)
        raise ValueError('unknown method {}'.format(method))

    

class Mean(InstrumentChannel):
    @staticmethod
    def __call__(data, ndim):
        '''Calculate the mean over the leading data.ndim-ndim axes of data.'''
        shape = data.shape
        try:
            data.shape = (np.prod(shape[:-ndim]),) + shape[-ndim:]
            mean = np.mean(data, axis=0)
        finally:
            data.shape = shape
        return mean



class TvMode(Instrument):
    '''Instrument that performs block-averaging data aquisition.

    Receive input samples indexed by (iteration, segment, time, channel)
    and reduce it to (segment, time, channel) by averaging over the first axis.

    If a marker array is provided in addition to the sample array, the markers
    identify the first segment. 
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_parameter('segments', ManualParameter, 
                           label='Number of segments. Zero for automatic.', 
                           vals=vals.Numbers(min_value=0))

        self.add_submodule('unpacker', Unpacker(self, 'unpacker'))
        self.add_submodule('ddc', DigitalDownconversion(self, 'ddc'))
        self.add_submodule('filter', Filter(self, 'filter'))
        self.add_submodule('sync', Synchronizer(self, 'sync'))
        self.add_submodule('mean', Mean(self, 'mean'))

    def __call__(self, source):
        '''Process blocks of samples provided by source.

        Requests blocks of data from source and processes it through
        `unpacker`, `ddc`, `filter`, and `synchronize`.

        Arguments:
            source: `iterable`
                An iterable (typically a generator) that provides multiplexed
                analog and digital data. Each block is indexed by 
                (segment, sample, channel)

        Note:
            The current implementation requires each block returned by source
            to contain a number of samples that is an integer multiple of 
            segments*samples.
        '''
        for block in source:
            analog, digital = self.unpacker(block)
            # extract synchronization markers
            segments = self.segments()
            marked_segments = self.sync(digital)
            if len(marked_segments) == 0:
                if segments != 1:
                    raise ValueError('No synchronization markers received.')
                first_segment = 0
            else:
                first_segment = marked_segments[0]
            if (segments == 0) or (segments is None):
                if len(marked_segments) < 2:
                    raise ValueError('Unable to determine number of segments.')
                segments = marked_segments[1] - marked_segments[0]
            if len(segments) > 2:
                if np.any(np.diff(marked_segments) - segments):
                    raise ValueError('Distance between (some) synchronization '
                                     'markers does not equal number of segments.')
            if np.prod(analog.shape[:-2]) < segments:
                raise ValueError('Each input block must contain at least '
                                 '`segments` ({}) segments.'.format(segments))
            # truncate block to a multiple of `segments` segments
            repetitions = analog.shape[0] // segments
            analog = np.reshape(analog[:repetitions*segments,...], 
                                (repetitions, segments)+analog.shape[1:])
            # process samples through the queue
            analog_ddc = self.ddc(analog)
            analog_fir = self.filter(analog_ddc)
            # TODO: channel maths goes here
            analog_mean = self.mean(analog_fir, analog.ndim-1)
            if first_segment:
                analog_mean = np.roll(analog_mean, -first_segment, axis=0)
            yield analog_mean
