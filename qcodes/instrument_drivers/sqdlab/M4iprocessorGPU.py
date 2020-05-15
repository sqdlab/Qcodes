import scipy
import logging
from qcodes import validators as vals, ManualParameter, ArrayParameter
from .ADCProcessorGPU import TvModeGPU
from qcodes.instrument_drivers.Spectrum.M4i import M4i
import qcodes.instrument_drivers.Spectrum.pyspcm as spcm
from qcodes.instrument.base import Instrument
import qcodes

class M4iprocessorGPU(Instrument):
    class DataArray(ArrayParameter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)            
            
        def get_raw(self):
            data = self.instrument.get_data()
            self.shape = data.shape
            return data

    def __init__(self, name, *args, digi=None, processor=None, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._digi = digi
        self.processor = processor

        self.check_digi()
        self.check_processor()
        
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
            'fft',
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
            'analog', self.DataArray, shape=(1,1,1),
            setpoint_names=('segment', 'sample', 'channel'), 
            label='Analog data array returned by processor.',
        )
        # Makes sure that uqtools accepts this as a compatible parameter
        self.analog.settable = False

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

    def get_data(self):
        '''
        Gets processed data from the GPU. Processing involves gathering data and passing it through TvMode
        '''

        # TODO: segments can be arbitrary on the adc, but tvmode 
        # does not yet support variable-sized blocks
        num_of_acquisitions = self.averages.get()*max(1, self.segments.get())
        max_blocksize = min(2**15, self.samples.get()*num_of_acquisitions)
        blocks = max(num_of_acquisitions//max_blocksize, 1)
        # WARNING DO NOT SET blocksize = 1; it needs AT to be LEAST 2!!!!!!
        blocksize = min(max_blocksize, num_of_acquisitions)
        source = self._digi.multiple_trigger_fifo_acquisition(
            segments=blocks*blocksize,
            samples=self.samples.get(), 
            blocksize=blocksize
        )
        return self.processor(source)
    
    def manual_close(self):
        self._digi.close()
        self.processor.close()
        

# def runme():
#     new_digi = Digitizer("one")
#     new_digi.segments(5)

#     new_digi.decimation.set(3)

#     import uqtools as uq

#     tv = uq.ParameterMeasurement(new_digi.analog, data_save=True)
#     tv_sample_av = uq.Integrate(tv, 'sample', average=True)
#     tv_segment_av = uq.Integrate(tv, 'segment', average=True)
#     tv_channel_av = uq.Integrate(tv, 'channel', average=True)

#     # data = new_digi.get_data()
#     # print(data.shape)
#     # new_digi.manual_close()

# if __name__ == '__main__':
#     runme()