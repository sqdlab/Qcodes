import scipy
import logging
import numpy as np
from qcodes import validators as vals, ManualParameter, ArrayParameter
from qcodes.instrument_drivers.sqdlab.ADCProcessorGPU import TvModeGPU
from qcodes.instrument_drivers.Spectrum.M4i import M4i
import qcodes.instrument_drivers.Spectrum.pyspcm as spcm
from qcodes.instrument.base import Instrument
import qcodes
import gc

class M4iprocessorGPU(Instrument):
    class M4iprocessorGPUException(Exception):
        pass

    class DataArray(ArrayParameter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, snapshot_value=False, **kwargs)
            self.get = self.get_raw       
            
        def get_raw(self):
            if 'singleshot' in self.name:
                self.instrument.processor.singleshot(True)
                if 'time_integrat' in self.name:
                    self.instrument.processor.enable_time_integration(True)
                else:
                    self.instrument.processor.enable_time_integration(False)
                data = self.instrument.get_data()
                self.shape = data.shape
            else:
                self.instrument.processor.singleshot(False)
                data = self.instrument.get_data()
                self.shape = data.shape
            gc.collect()
            return data            

    class FFTArray(ArrayParameter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.freqs = None        
            
        def get_raw(self):
            old_fft_enabled = self.instrument.fft_enabled()
            self.instrument.fft_enabled(True)

            data = self.instrument.get_data()
            self.freqs = np.fft.fftfreq(data.shape[-2], 1/self.instrument._digi.sample_rate.get())
            self.shape = data.shape
            self.instrument.fft_enabled(old_fft_enabled)
            return data

    class PSDArray(FFTArray):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)            
            
        def get_raw(self):
            data = super().get_raw()
            return np.abs(data)**2

    def __init__(self, name, *args, digi=None, processor=None, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._digi = digi
        self.processor = processor

        self.check_digi()
        self.check_processor()

        self.override_card_lock = False
        
        self.add_parameter(
            'if_freq', 
            label='Intermediate frequency.\
                   The digital down conversion is performed at this frequency.', 
            set_cmd=self._set_if_freq,
            vals=vals.Numbers(0,self._digi.sample_rate.get()//2), 
            initial_value=25e6)

        self.add_parameter(
            'decimation', 
            label='Number of points to decimate.', 
            set_cmd=self.processor.filter.decimation.set,
            vals=vals.Ints(1,),
            initial_value=1)

        self.add_parameter(
            'fir_coeffs',
            label='FIR coefficients.',
            shape=(1,), 
            set_cmd=self.processor.filter.coefficients.set,
            vals=vals.Arrays(), 
            initial_value=self.processor.filter.coefficients.get(),
            docstring="FIR coefficients. You can obtain your own \
                       coefficients using scipy.signal.firwin.\
                        The default coefficients use taps=128 and cutoff=0.05 (value is in multiples of Nyquist frequency)\
                        (cutoff corresponds to 5MHz for a sampling rate of 500 MSamples/s)")

        self.add_parameter(
            'fft_enabled',
            set_cmd=self.processor.fft.enabled,
            label='Enable or disable FFT.', 
            vals=vals.Bool(),
            initial_value=False)

        self.add_parameter(
            'samples', ManualParameter, 
            label='Number of samples per trigger.', 
            vals=vals.Multiples(divisor=16, min_value=32), 
            default_value=2048)
        self.samples(2048)

        self.add_parameter(
            'averages', ManualParameter,
            label='Number of blocks acquired.',
            vals=vals.Ints(1), default_value=1)
        self.averages(1024)

        self.add_parameter(
            'channels',
            label='Number of channels',
            set_cmd=self._set_channels,
            vals=vals.Ints(1,2), initial_value=1)

        self.add_parameter(
            'segments',
            set_cmd=self._set_segments,
            label='Number of Segments',
            vals=vals.Ints(0,2**17), initial_value=1,
            docstring="Number of segments.\
                       Set to zero for autosegmentation.\
                       Connect the sequence start trigger to X0 (X1) for channels 0 (1) of the digitizer.")

        self.add_parameter(
            'sample_rate',
            get_cmd=self._digi.sample_rate.get,
            set_cmd=self._set_sample_rate,
            label='Sampling rate',
            vals=vals.Numbers())

        self.add_parameter(
            'analog', self.DataArray, shape=(1,1,1),
            setpoint_names=('segment', 'sample', 'channel'), 
            label='Analog data array returned by processor.',
        )
        # Makes sure that uqtools accepts this as a compatible parameter
        self.analog.settable = False

        self.add_parameter(
            'singleshot_analog', self.DataArray, shape=(1,1,1,1),
            setpoint_names=('iterations', 'segment', 'sample', 'channel'), 
            label='Analog data array returned by processor.',
        )
        # Makes sure that uqtools accepts this as a compatible parameter
        self.singleshot_analog.settable = False

        self.add_parameter(
            'time_integrated_singleshot_analog', self.DataArray, shape=(1,1,1),
            setpoint_names=('iterations', 'segment', 'channel'), 
            label='Analog data array returned by processor.',
        )
        # Makes sure that uqtools accepts this as a compatible parameter
        self.time_integrated_singleshot_analog.settable = False

        self.add_parameter(
            'fft', self.FFTArray, shape=(1,1,1),
            setpoint_names=('segment', 'sample', 'channel'), 
            label='FFT of data returned by processor.',
        )
        # Makes sure that uqtools accepts this as a compatible parameter
        self.fft.settable = False

        self.add_parameter(
            'psd', self.PSDArray, shape=(1,1,1),
            setpoint_names=('segment', 'sample', 'channel'), 
            label='Power Spectral Density returned by processor.',
        )
        # Makes sure that uqtools accepts this as a compatible parameter
        self.psd.settable = False

    def check_digi(self):
        if self._digi is None:
            logging.info("Setting default processor as TvModeGPU")
            try:
                station = qcodes.config.current_config['user']['station']
                try:
                    station.close_and_remove_instrument(self.name+'_digi')
                except KeyError:
                    pass
                self._digi = M4i(self.name+'_digi')
                station.add_component(self._digi, update_snapshot=False)
            except KeyError:
                self._digi = M4i(self.name+'_digi')
            

            # digitizer setup (one analog, one digital channel)
            self._digi.clock_mode.set(spcm.SPC_CM_EXTREFCLOCK)
            self._digi.reference_clock.set(10000000)#10 MHz ref clk
            self._digi.sample_rate.set(500000000) #500 Msamples per second
            assert self._digi.sample_rate.get() == 500000000, "The digitizer could not be acquired. \
                                                    It might be acquired by a different kernel."

            self._digi.set_ext0_OR_trigger_settings(spcm.SPC_TM_POS, termination=0, 
                                            coupling=0, level0=500)
        
            self._digi.multipurpose_mode_0.set('disabled')
            self._digi.multipurpose_mode_1.set('disabled')
            self._digi.multipurpose_mode_2.set('disabled')
           
            #self._digi.enable_channels(spcm.CHANNEL0 | spcm.CHANNEL1) # spcm.CHANNEL0 | spcm.CHANNEL1
            self._digi.set_channel_settings(1, mV_range=1000., input_path=1, 
                                    termination=0, coupling=1)
            self._digi.set_channel_settings(0, mV_range=1000., input_path=1, 
                                    termination=0, coupling=1)  
        else:
            if not isinstance(self._digi, M4i):
                raise Exception("Digitizer needs a processor")
            else :
                # TODO: create checks for (analog, digital) channel mapping!!
                pass

        
    def check_processor(self):
        if self.processor is None:
            logging.info("Setting default processor as TvModeGPU")
            try:
                station = qcodes.config.current_config['user']['station']
                try:
                    station.close_and_remove_instrument(self.name+'_tvmode')
                except KeyError:
                    pass
                self.processor = TvModeGPU(self.name+'_tvmode')
                station.add_component(self.processor, update_snapshot=True)
            except KeyError:
                self.processor = TvModeGPU(self.name+'_tvmode')
            
            # intermediate frequency
            # ideally, this is half the Nyquist frequency, because
            # * an even FIR filter can exactly filter out the image 
            #   frequency after down-conversion
            # * it gives the maximum bandwidth
            if_freq = -25e+6 # 125MHz at 500MS/s
            self.processor.ddc.intermediate_frequency.set(2*if_freq/self._digi.sample_rate.get())
            # convert ADC codes to voltages
            self.processor.ddc.scale.set(self._digi.range_channel_1.get() * 2**(-13)) 

            # when using GPU, the only supported filter mode is 'valid', 
            # which returns only samples where the signal and filter 
            # waveforms overlap completely. the length of each output 
            # segment is thus reduced by the length of the filter-1
            self.processor.filter.mode.set('valid') # 'full' for CPU
            self.processor.filter.decimation.set(1)
            #tvmode.filter.set_default_filter()

            # generate a low-pass filter using scipy
            # an even number of taps cancels the Nyquist frequency perfectly
            taps = 128
            #cut-off frequency in units of the Nyquist frequency
            cutoff = 0.05 # 0.02*Nyquist = 5MHz

            self.processor.filter.coefficients.set(scipy.signal.firwin(taps, [cutoff]))
            # tvmode.filter.coefficients.set(np.array([1,1,1,1]))

            self.processor.filter.description.set('firwin with {} taps, {} cutoff'
                                        .format(taps, cutoff))
        else:
            if not isinstance(self.processor, TvModeGPU):
                raise Exception("Digitizer needs a processor")
            else :
                # TODO: create checks for (unpacker, sync) depending upon digitizer parameter (channels,(analog & digital))
                pass
        
    def _set_segments(self, segments):
        if segments == 0:
            # use autosegmentation. Assuming X0 is the sequence start trigger.
            # ADC input has one marker, which indicates sequence start
            self._digi.multipurpose_mode_0.set('digital_in')
            self.processor.unpacker.markers.set(1)
            self.processor.sync.method.set('all')
            self.processor.sync.mask.set(0x01)
            logging.warning("Autosegmentation enabled. The number of acquisitions is \
                            set to number of averages in total. Number of acquisitions \
                            per segment will be averages//segments")
        elif segments == 1:
            # only one segment. Not checking for sequence start trigger
            self._digi.multipurpose_mode_0.set('disabled')
            self.processor.unpacker.markers.set(0)
            # self.processor.sync.method.set('all')
            self.processor.sync.mask.set(0x00)
        else:
            # setting number of segments to specific value.
            # Assuming X0 is the sequence start trigger.
            self._digi.multipurpose_mode_0.set('digital_in')
            self.processor.unpacker.markers.set(1)
            self.processor.sync.method.set('all')
            self.processor.sync.mask.set(0x01)
        self.processor.segments.set(segments)

    def _set_channels(self, num_of_channels):
        if num_of_channels == 1:
            self._digi.enable_channels(spcm.CHANNEL0) # spcm.CHANNEL0 | spcm.CHANNEL1
        if num_of_channels == 2:
            self._digi.enable_channels(spcm.CHANNEL0 | spcm.CHANNEL1) # spcm.CHANNEL0 | spcm.CHANNEL1

    def _set_if_freq(self, freq):
        self.processor.ddc.intermediate_frequency.set(2*-freq/self._digi.sample_rate.get())
    
    def _set_sample_rate(self, rate):
        self._digi.sample_rate.set(rate)
        new_rate = self.sample_rate.get()
        logging.warning(f"Cannot set sampling rate to {rate}, it is set to {new_rate} instead")

    def get_data(self):
        '''
        Gets processed data from the GPU. Processing involves gathering data and passing it through TvMode
        '''
        assert len(self.processor.filter.coefficients.get()) <= self.samples.get(), "The length of \
            the fir filter window must be less than or equal to the number of samples per acquisition"
        # TODO: segments can be arbitrary on the adc, but tvmode 
        # does not yet support variable-sized blocks
        num_of_acquisitions = self.averages.get()*max(1, self.segments.get())
        # assert num_of_acquisitions >= 16, "Number of acquisitions must be greater than or equal to 16."
        assert (num_of_acquisitions*self.samples.get()%4096 == 0) or (num_of_acquisitions*self.samples.get() in [2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11]), "The number of total samples requested to the card is not valid.\nThis must be 16, 32, 64, 128, 256, 512, 1k ,2k or any multiple of 4k.\nThe easiest way to ensure this is to use powers of 2 for averages, samples and segments, probably in that order of priority."
        max_blocksize = min(2**28//2**int(np.ceil(np.log2(self.samples.get()))), self.samples.get()*num_of_acquisitions)
        assert self.segments.get()*self.samples.get() <= max_blocksize, "The number of segments does not fit in 1 block of GPU processing. Reduce number of segments or samples."
        blocks = max(num_of_acquisitions//max_blocksize, 1)
        blocksize = min(max_blocksize, num_of_acquisitions)
        
        # getting last error. If an error occurred, the card is locked to all functions until the error is read.
        # Note: You may get the last error multiple times.
        # This function is only enabled if you want to override the card lock.
        if self.override_card_lock:
            self._digi.get_error_info32bit()

        try:
            source = self._digi.multiple_trigger_fifo_acquisition(
                segments=blocks*blocksize,
                samples=self.samples.get(), 
                blocksize=blocksize
            )
            self.processor._singleshot_shape = (self.averages.get(),
                                                self.segments.get(),
                                                (self.samples.get() - len(self.fir_coeffs.get()) + self.processor.filter.decimation.get())//self.processor.filter.decimation.get(),
                                                self.channels.get())
            self.processor._blocksize = blocksize
            return self.processor(source)
            # return source
        except ValueError as e:
            # self.processor.close()
            raise self.M4iprocessorGPUException('Check if the sequece start trigger is connected and is arriving after the acquisition trigger') from e
        except RuntimeError as e:
            msg = ("The card raised an exception. It may be locked. Try getting the error using "
                "{}._digi.get_error_info32bit(verbose=True) or "
                "setting {}.override_card_lock=True.".format(self.name, self.name))
            raise self.M4iprocessorGPUException(msg) from e
    
    def manual_close(self):
        self._digi.close()
        self.processor.close()
        

def runme():
    new_digi = M4iprocessorGPU("one")
    new_digi.segments(4) 
    new_digi.averages(2**17)
    new_digi.samples(2**10-16)
    new_digi.fir_coeffs(np.array([1]))
    # new_digi.sample_rate(100e6)

    import uqtools as uq

    # switch to no data files, just memory
    uq.config.store = 'MemoryStore'
    uq.config.store_kwargs = {}

    tv = uq.DigiTvModeMeasurement(new_digi, singleshot=False, data_save=True)
    tv_ss = uq.DigiTvModeMeasurement(new_digi, singleshot=True, data_save=True)

    # store = tv()
    # print(store)
    # store = tv()
    # print(store)

#     tv = uq.ParameterMeasurement(new_digi.analog, data_save=True)
#     tv_sample_av = uq.Integrate(tv, 'sample', average=True)
#     tv_segment_av = uq.Integrate(tv, 'segment', average=True)
#     tv_channel_av = uq.Integrate(tv, 'channel', average=True)

#     print("Ithee bro")
    # data = new_digi.get_data()
    # import time
    # starttime = time.time()
    # data = new_digi.singleshot_analog()
    # print(time.time()-starttime)
    # print(data.shape)
    # starttime = time.time()
    # new_digi.processor.time_integrate.start=100
    # new_digi.processor.time_integrate.stop=101
    data = new_digi.time_integrated_singleshot_analog()
    # print(time.time()-starttime)
    # print(data.shape)
    # data = np.array(new_digi.time_integrated_singleshot_analog())
    # print(data.shape)
    # data = np.array(new_digi.analog())
    print(data.shape)

if __name__ == '__main__':
    runme()