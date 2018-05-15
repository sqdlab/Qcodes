import numpy as np

from qcodes import (
    ArrayParameter, InstrumentChannel, VisaInstrument, validators as vals,
    DataArray
)

class DPOCurve(ArrayParameter):
    def __init__(self, name, source, **kwargs):
        '''
        Input
        -----
        parent: `Instrument`
            Object used to communicate with the instrument
        name: `str`
            Name of the Parameter
        source: `str`
            Channel, math or reference waveform to retrieve.
            One of CH<x>, MATH<x>, REF<x>, DIGITALALL.
        '''
        super().__init__(name, shape=(), **kwargs)
        self.source = source

    def preamble(self):
        '''
        Request and format waveform output preamble.
        '''
        preamble_keys_parsers = list(zip(*[
            ('BYT_NR', int), ('BIT_NR', int), ('ENCDG', str), ('BN_FMT', str),
            ('BYT_OR', str), ('WFID', str), ('NR_PT', int), ('PT_FMT', str),
            ('XUNIT', str), ('XINCR', float), ('XZERO', float), ('PT_OFF', int), 
            ('YUNIT', str), ('YMULT', float), ('YOFF', float), ('YZERO', float), 
            ('NR_FR', int)
        ]))
        preamble_values = self._instrument.ask('WFMO?').split(';')
        return dict((key, parser(value)) for value, key, parser 
                    in zip(preamble_values, *preamble_keys_parsers))

    def prepare(self, data_start=1, data_stop=(1<<31)-1, frame_start=1, 
                frame_stop=(1<<31)-1):
        '''
        Prepare curve for data acquisition.

        Sets data start/stop and frame start/stop on the instrument
        and sets up shape and setpoints of the parameter.

        Input
        -----
        data_start, data_stop: `int`
            First and last sample to transfer
        frame_start, frame_stop: `int`
            First and last frame to transfer in FastFrame mode
        '''
        # set data source and range
        self._instrument.write('DATA:SOURCE {}'.format(self.source))
        self._instrument.write('DATA:START {}'.format(data_start))
        self._instrument.write('DATA:STOP {}'.format(data_stop))
        self._instrument.write('DATA:FRAMESTART {}'.format(frame_start))
        self._instrument.write('DATA:FRAMESTOP {}'.format(frame_stop))
        # Choose encoding
        if self._instrument._parent.acq_mode() == 'average':
            # int16 for average data
            self._instrument.write('DATA:ENC SRI; :WFMO:BYT_NR 2')
        else:
            # int8 for other data
            self._instrument.write('DATA:ENC SRI; :WFMO:BYT_NR 1')
        # determine label and data unit
        pre = self.preamble()
        self.label = pre['WFID']
        self.unit = pre['YUNIT']
        # determine setpoints
        sp_times = DataArray(
            name='X', unit=pre['XUNIT'], is_setpoint=True, 
            preset_data=pre['XZERO'] + pre['XINCR']*np.arange(pre['NR_PT'])
        )
        if pre['NR_FR'] < 2: #not fastframe:
            # output is 1d if FastFrame is disabled
            self.shape = (pre['NR_PT'],)
            self.setpoints = (sp_times,)
        else:
            # output is 2d if FastFrame is enabled
            self.shape = (pre['NR_FR'], pre['NR_PT'])
            sp_frames = DataArray(
                name='frame', unit='', is_setpoint=True, 
                preset_data=np.arange(pre['NR_FR'])
            )
            sp_times.nest(len(sp_frames))
            self.setpoints = (sp_frames, sp_times)


    def get_raw(self, scale=True):
        '''
        Retrieve data from instrument.

        * acq_mode==sample, hires, average: ok
        * acq_mode==peakdetect, envelope: 0::2 and 1::2 are min/max
        * acq_mode==wfmdb: result is a histogram, samples must not be scaled
        '''
        self._instrument.write('DATA:SOURCE {}'.format(self.source))
        # get raw data from device
        pre = self.preamble()
        if pre['ENCDG'] == 'BIN':
            #order = dict('MSB':'>', 'LSB':'<')[pre['BYT_OR']]
            dtype = {'RI':{1:'b', 2:'h', 4:'i', 8:'q'}[pre['BYT_NR']], 
                     'RP':{1:'B', 2:'H', 4:'I', 8:'Q'}[pre['BYT_NR']], 
                     'FP':'f'}[pre['BN_FMT']]
            data = self._instrument._parent.visa_handle.query_binary_values(
                'CURVE?', datatype=dtype, is_big_endian=(pre['BYT_OR']=='MSB'),
                container=np.array
            )
        elif pre['ENCDG'] == 'ASC':
            converter = float if pre['BN_FMT'] == 'fp' else int
            data = self._instrument._parent.visa_handle.query_ascii_values(
                'CURVE?', converter=converter, container=np.array
            )
        else:
            raise ValueError('Invalid encoding {}.'.format(pre['ENCDG']))
        # shift and scale
        data.shape = self.shape
        if scale:
            data = pre['YZERO'] + pre['YMULT']*data.astype(np.float32)
        return data


class DPOPixmap(DPOCurve):
    def prepare(self, rows=252, columns=1000):
        '''
        Prepare pixmap for data acquisition.

        Sets up shape and setpoints of the parameters.
        '''
        # set data source and range
        self._instrument.write('DATA:SOURCE {}'.format(self.source))
        self._instrument.write('DATA:START {}'.format(1))
        self._instrument.write('DATA:STOP {}'.format((1<<31)-1))
        # Data encoding set to ASCII because
        # * The data is corrupted when choosing binary mode
        # * It is faster for clean waveforms ('0' is 2 bytes instead of 8)
        self._instrument.write('WFMO:ENC ASCII')
        # determine label and data unit
        pre = self.preamble()
        if pre['NR_PT'] != rows*columns:
            raise ValueError('Pixmap size is not equal to rows*columns.')
        self.shape = (columns, rows)
        self.label = pre['WFID']
        self.unit = ''
        # determine setpoints
        sp_xs = DataArray(
            name='X', unit=pre['XUNIT'], is_setpoint=True, 
            preset_data=pre['XZERO'] + pre['XINCR']*np.arange(columns)
        )
        sp_ys = DataArray(
            name='Y', unit=pre['YUNIT'], is_setpoint=True,
            preset_data=pre['YZERO'] + pre['YMULT']*(np.arange(rows)+pre['YOFF'])
        )
        sp_ys.nest(len(sp_xs))
        self.setpoints = (sp_xs, sp_ys)

    def get_raw(self):
        self._instrument.write('WFMO:ENC ASCII')
        return super().get_raw(False)


class DPOChannel(InstrumentChannel):
    def __init__(self, parent, name, channel):
        super().__init__(parent, name)
        self.channel = channel

        # Vertical setup
        self.add_parameter(
            'state', label='State',
            get_cmd='SELECT:CH{}?'.format(channel),
            set_cmd='SELECT:CH{} {}'.format(channel, '{:d}'),
            get_parser=lambda v: bool(int(v)), vals=vals.Bool()
        )
        self.add_parameter(
            'label', label='Label', 
            get_cmd='CH{}:LABEL:NAME?'.format(channel),
            set_cmd='CH{}:LABEL:NAME {}'.format(channel, '{}'),
            get_parser=lambda v: v[1:-1], vals=vals.Strings(0, 32)
        )
        self.add_parameter(
            'offset', label='Vertical Offset', unit='V',
            docstring='Vertical offset applied after digitizing, in volts.', 
            get_cmd='CH{}:OFFSET?'.format(channel), 
            set_cmd='CH{}:OFFSET {}'.format(channel, '{:f}'),
            get_parser=float
        )
        self.add_parameter(
            'vposition', label='Vertical Position', unit='div',
            docstring='Vertical offset applied before digitizing, in divisions.', 
            get_cmd='CH{}:POSITION?'.format(channel), 
            set_cmd='CH{}:POSITION {}'.format(channel, '{:f}'),
            get_parser=float, vals=vals.Numbers(-8., 8.)
        )
        self.add_parameter(
            'vscale', label='Vertical Scale', unit='V', 
            get_cmd='CH{}:SCALE?'.format(channel),
            set_cmd='CH{}:SCALE {}'.format(channel, '{:f}')
        )
        self.add_parameter(
            'bandwidth', label='Bandwidth', unit='Hz',
            get_cmd='CH{}:BANDWIDTH?'.format(channel), get_parser=float,
            set_cmd='CH{}:BANDWIDTH {}'.format(channel, '{:f}')
        )
        self.add_parameter(
            'coupling', label='Input Coupling',
            get_cmd='CH{}:COUPLING?'.format(channel),
            set_cmd='CH{}:COUPLING {}'.format(channel, '{}'),
            val_mapping={'ac':'AC', 'dc':'DC', 'dcreject':'DCREJ', 'gnd':'GND'}
        )
        self.add_parameter(
            'termination', label='Input Termination', unit='Ohm',
            get_cmd='CH{}:TERMINATION?'.format(channel),
            set_cmd='CH{}:TERMINATION {}'.format(channel, '{:f}'),
            get_parser=float, vals=vals.Enum(50, 50., 1000000, 1e6)
        )
        self.add_parameter(
            'deskew', label='Deskew', unit='s',
            get_cmd='CH{}:DESKEW?'.format(channel), 
            set_cmd='CH{}:DESKEW {}'.format(channel, '{:f}'), 
            get_parser=float, vals=vals.Numbers(-25e-9, 25e-9)
        )
        # data transfer
        self.add_parameter('curve', DPOCurve, source='CH{}'.format(channel))
        self.add_parameter('pixmap', DPOPixmap, source='CH{}'.format(channel))


class Tektronix_DPO70000(VisaInstrument):
    def __init__(self, name, address):
        super().__init__(name, address, terminator='\n')

        # Acquisition setup
        self.add_parameter(
            'acq_state', label='Acuqisition state',
            get_cmd='ACQ:STATE?', get_parser=lambda v: bool(int(v)),
            set_cmd='ACQ:STATE {:d}', vals=vals.Bool()
        )
        self.add_parameter(
            'acq_single', label='Single acquisition mode',
            get_cmd='ACQ:STOPAFTER?', set_cmd='ACQ:STOPAFTER {}', 
            val_mapping={False:'RUNST', True:'SEQ'}
        )
        self.add_parameter(
            'acq_mode', label='Acquisition Mode',
            docstring='''
            In each each acquisition interval, show
             * sample: the first sampled value
             * peakdetect: the minimum and maximum samples
             * hires: the average of all samples
             * average: the first sample averaged over separate acquisitions
             * wfmdb: a histogram of all samples of one or more acquisitions
             * envelope: a histogram of the minimum and maximum samples of 
                         multiple acquisitions
            ''',
            get_cmd='ACQ:MODE?', set_cmd='ACQ:MODE {}',
            val_mapping={'sample':'SAM', 'peakdetect':'PEAK', 'hires':'HIR',
                         'average':'AVE', 'wfmdb':'WFMDB', 'envelope':'ENV'}
        )
        self.add_parameter(
            'acq_averages', label='Number of averages',
            get_cmd='ACQ:NUMAVG?', get_parser=int,
            set_cmd='ACQ:NUMAVG {:d}', vals=vals.Numbers(1)
        )
        self.add_parameter(
            'acq_envelopes', label='Number of envelope waveforms',
            get_cmd='ACQ:NUMENV?', get_parser=int,
            set_cmd='ACQ:NUMENV {:d}', vals=vals.Numbers(1, 2e9)
        )
        self.add_parameter(
            'acq_wfmdbs', label='Number of wfmdb waveforms',
            get_cmd='ACQ:NUMSAMPLES?', get_parser=int,
            set_cmd='ACQ:NUMSAMPLES {:d}', vals=vals.Numbers(5000, 2147400000)
        )
        self.add_parameter(
            'acq_sampling', label='Sampling mode', 
            get_cmd='ACQ:SAMPLINGMODE?', set_cmd='ACQ:SAMPINGMODE {}', 
            val_mapping={'realtime':'RT', 'equivalent':'ET', 'interpolated':'IT'}
        )

        # Trigger setup

        # Horizontal setup
        self.add_parameter(
            'hmode', label='Horizontal Mode',
            docstring='Selects the automatic horzontal model.'
                      'auto: Set time/division. Keeps the record length constant.'
                      'constant: Set time/division. Keeps the sample rate constant.'
                      'manual: Set record length and sample rate.', 
            get_cmd='HOR:MODE?', set_cmd='HOR:MODE {}', 
            val_mapping={'auto':'AUTO', 'constant':'CONS', 'manual':'MAN'}
        )
        self.add_parameter(
            'hscale', label='Horizontal Scale', unit='s', 
            docstring='Horizontal scale in seconds per division.', 
            get_cmd='HOR:MODE:SCALE?', get_parser=float,
            set_cmd='HOR:MODE:SCALE {}'
        )
        self.add_parameter(
            'samplerate', label='Sample Rate', unit='1/s',
            get_cmd='HOR:MODE:SAMPLERATE?', get_parser=float,
            set_cmd='HOR:MODE:SAMPLERATE {}', set_parser=float
        )
        self.add_parameter(
            'recordlength', label='Record Length', 
            get_cmd='HOR:MODE:RECORDLENGTH?', get_parser=int,
            set_cmd='HOR:MODE:RECORDLENGTH {}', set_parser=int
        )
        self.add_parameter(
            'hposition', label='Horizontal Position', unit='%', 
            docstring='Position of the trigger point on screen in %.',
            get_cmd='HOR:POS?', get_parser=float,
            set_cmd='HOR:POS {}', vals=vals.Numbers(0., 100.)
        )
        self.add_parameter(
            'hdelay_status', label='Horizontal Delay Status',
            get_cmd='HOR:DELAY:MODE?', get_parser=lambda v: bool(int(v)),
            set_cmd='HOR:DELAY:MODE {:d}', vals=vals.Bool()
        )
        self.add_parameter(
            'hdelay_pos', label='Horizontal Position', unit='%', 
            docstring='Position of the trigger point on screen in %.',
            get_cmd='HOR:DELAY:POS?', get_parser=float,
            set_cmd='HOR:DELAY:POS {}', vals=vals.Numbers(0., 100.)
        )
        self.add_parameter(
            'hdelay_time', label='Horizontal Delay', unit='s',
            get_cmd='HOR:DELAY:TIME?', get_parser=float,
            set_cmd='HOR:DELAY:TIME {}'
        )

        for ch_id, ch_name in enumerate(['ch1', 'ch2', 'ch3', 'ch4'], 1):
            self.add_submodule(ch_name, DPOChannel(self, ch_name, ch_id))

    def autoset(self):
        self.write('AUTOSET')

    def clear(self):
        self.write('CLEAR')