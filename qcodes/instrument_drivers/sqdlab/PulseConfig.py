import copy
import math

from qcodes import Instrument, InstrumentChannel
from qcodes.utils import validators as vals

import pulsegen
import pulsegen.pulse as pp
PULSE_SHAPES = dict((p.__name__, p) for p in (pp.ZeroPulse, pp.SquarePulse, pp.GaussianPulse, pp.ArbitraryPulse, pp.DRAGGaussPulse))


AWG_MAP = copy.copy(pulsegen.sequence.AWG_MAP)
AWG_MAP['Trigger'] = pulsegen.sequence.AWG(model='Trigger', ch_count=0, marker_count=0, granularity=1)

class ChannelPair(InstrumentChannel):

    def __init__(self, parent, name, awg_model):
        super().__init__(parent, name)
        #self.parent = parent
        self.awg_model = awg_model
        self.channels = []

        self.parameters['separation'] = self.parent.separation

        self.add_parameter('mixer_power', unit='dBm',
                           set_cmd=None,
                           get_parser=float,
                           vals=vals.Numbers(-120, 16),
                           docstring='Power output for mixer.')
        self.mixer_power(16.)
        
        self.add_parameter('sigma', unit='s',
                           set_cmd=None,
                           get_parser=float,
                           vals=vals.Numbers(),
                           docstring='Length unit of GaussianPulse and DRAGGaussPulse pulses.')
        self.sigma(2e-8)

        self.add_parameter('truncate',
                           set_cmd=None,
                           get_parser=float,
                           vals=vals.Numbers(),
                           docstring='Truncate GaussianPulse at truncate sigmas from centre.')
        self.truncate(2)

        self.add_parameter('qscale',
                           set_cmd=None,
                           get_parser=float,
                           vals=vals.Numbers(),
                           docstring='Quadrature scale factor of DRAGGaussPulse')
        self.qscale(0.5)

        self.add_parameter('anharmonicity', unit='rad/s',
                           set_cmd=None,
                           get_parser=float,
                           vals=vals.Numbers(),
                           docstring='Anharmonicity argument to DRAGGaussPulse')
        self.anharmonicity(2*math.pi*300e6)

        self.add_parameter('length', unit='s',
                           set_cmd=None,
                           get_parser=float,
                           vals=vals.Numbers(),
                           docstring='Default length of SquarePulse')
        self.length(20e-9)

        self.add_parameter('pulse_shape',
                           set_cmd=None,
                           docstring='Supported pulse shapes are:\n\t'+'\n\t'.join(PULSE_SHAPES.keys()))
        self.pulse_shape('GaussianPulse')

        self.add_parameter('pi_amp',
                           set_cmd=None,
                           get_parser=float,
                           vals=vals.Numbers(),
                           docstring='Amplitude of pi pulses created by pix() and piy().')
        self.pi_amp(0.9)

        self.add_parameter('pi_half_amp',
                           set_cmd=None,
                           get_parser=float,
                           vals=vals.Numbers(),
                           docstring='Amplitude of pi/2 pulses created by pihalfx() and pihalfy().')
        self.pi_half_amp(0.45)

        self.add_parameter('mixer_calibration',
                           set_cmd=None,
                           get_parser=list,
                           vals=vals.Lists())
        self.mixer_calibration([[1., math.pi/2.], [1., 0.]])

        self.add_parameter('if_freq', unit='rad/s',
                           set_cmd=None,
                           get_parser=float,
                           vals=vals.Numbers())
        self.if_freq(100e6)

    def add_channel(self, channel):
        if len(self.channels) < 2:
            self.channels.append(channel)
            channel.channel_pair = self
            channel.channel_type('mixer')
            return True
        return False

    def get_config(self):
        config = {}
        for k,p in self.parameters.items():
            config[k] = p()
        config['awg_models'] = self.awg_model
        config['pulse_shape'] = PULSE_SHAPES[config['pulse_shape']]
        return config

class Channel(InstrumentChannel):

    def __init__(self, parent, name, awg_model, awg_channel):
        super().__init__(parent, name)
        #self.parent = parent
        self.awg_model = awg_model
        self.awg_channel = awg_channel
        # self._sampling_rate = self.parent.sampling_rate()
        self.parameters['pattern_length'] = self.parent.pattern_length
        self.parameters['fixed_point'] = self.parent.fixed_point
        self.parameters['sampling_rate_multiplier'] = self.parent.sampling_rate_multiplier
        self.parameters['prepend'] = self.parent.prepend
        self.parameters['append'] = self.parent.append
        self.parameters['use_pinv'] = self.parent.use_pinv
        self.parameters['ir_tlim'] = self.parent.ir_tlim
        self.channel_pair = None

        self.add_parameter('channel_type',
                           set_cmd=self._set_channel_type,
                           get_parser=str,
                           val_mapping={'single' : 'single',
                                        'mixer' : 'mixer'})

        self.add_parameter('sampling_rate', unit='Hz',
                           # getcmd=self._get_sampling_rate,
                           set_cmd=self._set_sampling_rate,
                           get_parser=float,
                           vals=vals.Numbers())
        self.sampling_rate(1.2e9)

        self.add_parameter('delay', unit='s',
                           set_cmd=None,
                           get_parser=float,
                           vals=vals.Numbers(),
                           docstring='Analog output delay')
        self.delay(0)

        self.add_parameter('delay_m0', unit='s',
                           set_cmd=None,
                           get_parser=float,
                           vals=vals.Numbers(),
                           docstring='Marker output delay')
        self.delay_m0(0)

        self.add_parameter('delay_m1', unit='s',
                           set_cmd=None,
                           get_parser=float,
                           vals=vals.Numbers(),
                           docstring='Marker output delay')
        self.delay_m1(0)

        self.add_parameter('use_optimal_control',
                           set_cmd=None,
                           get_parser=bool,
                           vals=vals.Bool())
        self.use_optimal_control(False)

        self.add_parameter('impulse_response',
                           set_cmd=None,
                           get_parser=str,
                           vals=vals.Strings(),
                           docstring='Path to a file containing the measured impulse response of the awg channel.')
        self.impulse_response('')

        self.add_parameter('use_fir',
                           set_cmd=None,
                           get_parser=bool,
                           vals=vals.Bool(),
                           docstring='Activate FIR filter. FIR filters are only applied when optimal control is active.')
        self.use_fir(False)

        self.add_parameter('fir_window',
                           set_cmd=None,
                           get_parser=str,
                           vals=vals.Strings(),
                           docstring='FIR window type.')
        self.fir_window('')

        self.add_parameter('fir_cutoff', unit='rad/s',
                           set_cmd=None,
                           get_parser=float,
                           vals=vals.Numbers(),
                           docstring='FIR cutoff frequency.')
        self.fir_cutoff(2*math.pi*300e6)

    def _set_channel_type(self, ch_type):
        if ch_type == 'mixer':
            if self.channel_pair is not None:
                return
            else:
                raise Exception('Trying to set channel_type to "mixer". '
                                +'Create a channel_pair and use its add_channel() method instead.')
        if ch_type == 'single':
            if self.channel_pair is not None:
                for ch in self.channel_pair.channels:
                    if ch is not self:
                        ch.channel_pair = None
                        ch.channel_type('single')
                        break
                self.parent.submodules = {key:val for key, val in self.parent.submodules.items() if val is not self.channel_pair}
                self.channel_pair = None
            else:
                return



    def _set_sampling_rate(self, rate):
        for _, parameter in self.parent.parameters.items():
            if isinstance(parameter, (Channel)) and parameter._awg_model==self._awg_model:
                parameter._sampling_rate = rate

    def get_config(self):
        config = {}
        if self.channel_pair is not None:
            for channel in self.channel_pair.channels:
                if channel is not self:
                    paired_channel = channel
                    break
            # Set paired channel settings to same as this one
            for k, p in paired_channel.parameters.items():
                if k not in ['delay', 'delay_m1', 'delay_m0']:
                    p(getattr(self, k)())

        for k, p in self.parameters.items():
            if k not in ['delay', 'delay_m1', 'delay_m0', 'fir_cutoff', 'fir_window']:
                config[k] = p()
            elif k == 'delay':
                config[k] = self.delay()
            elif k == 'delay_m0':
                config['marker_delays'] = (self.delay_m0(), self.delay_m1())
            elif k=='fir_window':
                config['fir_parameters'] = (self.fir_window(), self.fir_cutoff())

        return config

class PulseConfig(Instrument):

    def __init__(self, name):
        super().__init__(name)

        self.nchannels = 0
        self._channel_config = []
        self._channel_pair_config = []

        self.add_parameter('pattern_length', unit='s',
                           set_cmd=None,
                           get_parser=float,
                           vals=vals.Numbers())
        self.pattern_length(5e-6)

        self.add_parameter('fixed_point', unit='s',
                           set_cmd=None,
                           get_parser=float,
                           vals=vals.Numbers())
        self.fixed_point(4e-6)

        self.add_parameter('sampling_rate_multiplier',
                           set_cmd=None,
                           get_parser=int,
                           vals=vals.Ints())
        self.sampling_rate_multiplier(10)

        self.add_parameter('prepend',
                           set_cmd=None,
                           get_parser=float,
                           vals=vals.Numbers())
        self.prepend(50e-9)

        self.add_parameter('append',
                           set_cmd=None,
                           get_parser=float,
                           vals=vals.Numbers())
        self.append(100e-9)

        self.add_parameter('use_pinv',
                           set_cmd=None,
                           get_parser=bool,
                           vals=vals.Bool(),
                           docstring='Use pseudo-inverse for optimal control.')
        self.use_pinv(True)

        self.add_parameter('ir_tlim',
                           set_cmd=None,
                           docstring='Start and stop times of the impulse response used for matrix inversion.')
        self.ir_tlim((-1e-8, 2e-7))

        self.add_parameter('separation', unit='s',
                           set_cmd=None,            
                           get_parser=float,
                           vals=vals.Numbers(),
                           docstring='Default delay between successive pulses.')
        self.separation(5e-9)

        self.add_parameter('low',
                           set_cmd=None,            
                           get_parser=float,
                           vals=vals.Numbers())
        self.low(-1)

        self.add_parameter('high',
                           set_cmd=None,            
                           get_parser=float,
                           vals=vals.Numbers())
        self.high(1)
        
        self.patch_pulsegen()

    def patch_pulsegen(self):
        pulsegen.sequence.cfg = self

    def add_awg(self, awg, model, sampling_rate):
        '''
        Append an AWG to the configuration.
        
        Input:
            awg - AWG model as defined in pulsegen.
                See get_awg_models for supported models.
            sampling_rate - sampling rate of all channels on this AWG
        '''
        # append an awg (and its channels) to the configuration
        if model not in AWG_MAP:
            raise ValueError('unknown awg model {0:s}.'.format(model))
        if not hasattr(self, '_awg_models'):
            self._awgs = []
            self._awg_models = []
            self._awg_channels = []
        self._awgs.append(awg)
        self._awg_models.append(model)

        chpair_name = 'chpair{:d}'.format(self.nchannels//2)
        chpair = ChannelPair(self, chpair_name, model)
        self.add_submodule(chpair_name, chpair)

        awg_chs = []
        for ch in range(AWG_MAP[model].ch_count):
            ch_name = 'ch{:d}'.format(self.nchannels)
            channel = Channel(self, ch_name, model, getattr(awg, 'ch{:d}'.format(ch+1)))
            self.add_submodule(ch_name, channel)
            if not chpair.add_channel(channel):
                chpair_name = 'chpair{:d}'.format(self.nchannels//2)
                chpair = ChannelPair(self, chpair_name, model)
                self.add_submodule(chpair_name, chpair)
                chpair.add_channel(channel)
            awg_chs.append(channel)
            self.nchannels += 1

        self._awg_channels.append(awg_chs)
        channel.sampling_rate(sampling_rate)

    def get_awgs(self):
        ''' return list of awgs '''
        if not hasattr(self, '_awgs'):
            return []
        else:
            return list(self._awgs)

    def remove_awg(self, index):
        ''' Remove all channels of awg[index]'''
        chpairs = []
        for _, submodule in self.submodules.items():
            if isinstance(submodule, (ChannelPair)):
                chpairs.append(submodule)

        for ch in range(AWG_MAP[self._awg_models[index]].ch_count):
            channel = self._awg_channels[index][ch]
            for chpair in chpairs:
                if channel in chpair.channels:
                    self.submodules = {key:val for key, val in self.submodules.items() if val is not chpair}
            self.submodules = {key:val for key, val in self.submodules.items() if val is not channel}
        self._awgs.pop(index)
        self._awg_models.pop(index)
        self._awg_channels.pop(index)

    def get_config(self):
        config = {}
        for k,p in self.parameters.items():
            if k == 'low':
                pass
            elif k == 'high':
                config['lowhigh'] = (self.low(), self.high())
            else:
                config[k] = p()
        return config

    def get_channel_config(self):
        cfg = []
        i=0
        for _, m in self.submodules.items():
            if isinstance(m, (Channel)):
                if m.channel_pair is not None:
                    ch_config = m.channel_pair.get_config()
                ch_config.update(m.get_config())
                ch_config.update(self.get_config())
                ch_config['mixer_calibration'] = ch_config['mixer_calibration'][i%2]
                i+=1
                cfg.append(ch_config)
        return cfg

    def get_channel_pair_config(self):
        cfg = []
        for _, m in self.submodules.items():
            if isinstance(m, (ChannelPair)):
                ch_config = m.channels[0].get_config()
                ch_config['delay'] = [ch_config['delay'], m.channels[1].delay()]
                ch_config['marker_delays'] = [ch_config['marker_delays'], (m.channels[1].delay_m0(), m.channels[1].delay_m1())]
                ch_config.update(m.get_config())
                ch_config.update(self.get_config())
                cfg.append(ch_config)
        return cfg