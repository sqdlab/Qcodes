import math
import logging

from qcodes import Instrument, InstrumentChannel, ManualParameter, validators as vals

class TimingChannel(InstrumentChannel):
    def __init__(self, parent:Instrument, name:str, duration_get_cmd:callable=None) -> None:
        super().__init__(parent, name)

        self.add_parameter(
            'channel', ManualParameter, 
            docstring='Output channel of the pulse generator'
        )
        if name == 'awg':
            # the primary awg must always be triggered at time zero
            self.add_parameter(
                'reference',  
                docstring='Time relative to which the in/output starts.',
                get_cmd=lambda:'awg output start'
            )
            self.add_parameter(
                'offset', unit='s', 
                docstring='In/output time offset from reference point.',
                get_cmd=lambda: 0.
            )
        else:
            self.add_parameter(
                'reference', ManualParameter, 
                docstring='Time relative to which the in/output starts.',
                vals=vals.Enum('zero', 'awg output start', 'fixed point'),
                initial_value='fixed point'
            )
            self.add_parameter(
                'offset', ManualParameter, unit='s', 
                docstring='In/output time offset from reference point.',
                vals=vals.Numbers(), initial_value=0.
            )
        if duration_get_cmd is not None:
            # if duration_get_cmd is given, use it to compute duration
            # and make duration only gettable
            self.add_parameter(
                'duration', unit='s', get_cmd=duration_get_cmd, 
                docstring='Duration of the in/output.',
            )
        else:
            self.add_parameter(
                'duration', ManualParameter, unit='s', 
                docstring='Duration of the in/output.',
                vals=vals.Numbers(min_value=0.), initial_value=None
            )
        self.add_parameter(
            'trigger_delay', ManualParameter, unit='s', 
            docstring='Time delay between sending a trigger pulse and start of '
                      'in/output on the instrument. Usually found in the data sheet.',
            vals=vals.Numbers(min_value=0.), initial_value=0.
        )
        self.add_parameter(
            'trigger_duration', ManualParameter, unit='s', 
            docstring='Duration of the trigger pulse sent to the instrument.',
            vals=vals.Numbers(min_value=0.), initial_value=100e-9
        )
        self.add_parameter(
            'trigger_holdoff', ManualParameter, unit='s', 
            docstring='Minimum time required before a trigger can be processed '
                      'after in/output finishes.',
            vals=vals.Numbers(min_value=0.), initial_value=0.
        )

    def get_timings(self, ref_points):
        '''
        Return channel timings.

        Arguments
        ---------
        ref_points: `dict` of `str`:`float` pairs
            Absolute times of possible reference points. Should contain values
            for all options of the `reference` parameter.

        Returns
        -------
        timings: `dict` of `str`:`float` pairs
            Absolute times for reference, out_start, out_end, trig_start and 
            trig_end.
        '''
        T_reference = ref_points[self.reference.get()]
        T_out_start = T_reference + self.offset.get()
        if self.duration.get() is None:
            T_out_stop = None
        else:
            T_out_stop = T_out_start + self.duration.get()
        T_trig_start = T_out_start - self.trigger_delay.get()
        T_trig_stop = T_trig_start + self.trigger_duration.get()

        return dict(reference=T_reference, 
                    out_start=T_out_start, out_end=T_out_stop,
                    trig_start=T_trig_start, trig_end=T_trig_stop)

    def update(self, ref_points):
        '''
        Push channel timings to the pulse generator.

        See `get_timings` for arguments.
        '''
        timings = self.get_timings(ref_points)
        channel = self.channel.get()
        if channel is None: 
            return
        if not hasattr(channel, 'delay'):
            if timings['trig_start'] != 0:
                print(channel)
                raise ValueError('Unable to set delay on channel {} to {}'
                                 .format(channel.name, timings['trig_start']))
        else:
            channel.delay.set(timings['trig_start'])
        if hasattr(channel, 'duration') and hasattr(channel.duration, 'set'):
            channel.duration.set(timings['trig_end'] - timings['trig_start'])



class TimingConfig(Instrument):
    '''
    qcodes port of the timing configuration tool.

    In contrast to the qtlab version, parameter setting does not update the
    pulse generator. Changes must be pushed to the instrument by calling 
    update().

    Input
    -----
    pulser: `qcodes.Instrument`
        Instance of a DG645 pulse generator driver.
    '''

    PULSER_HOLDOFF = 25e-9
    MIN_ACQ_DURATION = 20e-9

    def __init__(self, name, pulser, fixed_point_get_cmd, awg_duration_get_cmd=None, 
                 acq_duration_get_cmd=None):
        super().__init__(name)
        self.pulser = pulser
        self.fixed_point_get_cmd = fixed_point_get_cmd

        self.add_parameter(
            'clock_interval', ManualParameter, unit='s', 
            docstring='Interval between triggers RECEIVED BY the pulse generator. '
                      'Usually 100ns or 200ns for 10MHz or 5MHz input clocks.',
            vals=vals.Numbers(min_value=0.), initial_value=200e-9
        )
        self.add_parameter(
            'repetition_time', unit='s', vals=vals.Numbers(min_value=0.), 
            docstring='Interval of the main trigger output (T0) by the pulse '
                      'generator.', 
            get_cmd=self.get_repetition_time, set_cmd=self.set_repetition_time
        )

        # The DG645 pulse generator used has 4+1 channels, so we duplicate that here
        self.add_submodule('awg', TimingChannel(self, 'awg', duration_get_cmd=awg_duration_get_cmd))
        self.add_submodule('acq', TimingChannel(self, 'acq', duration_get_cmd=acq_duration_get_cmd))
        self.add_submodule('readout', TimingChannel(self, 'readout'))
        self.add_submodule('aux1', TimingChannel(self, 'aux1'))
        self.add_submodule('aux2', TimingChannel(self, 'aux2'))

    def get_repetition_time(self):
        return self.clock_interval.get() * self.pulser.trigger_prescale_factor.get()

    def set_repetition_time(self, value):
        prescale = value / self.clock_interval.get()
        if abs(prescale - round(prescale)) > 1e-6:
            raise ValueError('Repetition time is not divisible by the clock '
                             'interval.')
        self.pulser.trigger_prescale_factor.set(round(prescale))

    def check_repetition_time(self):
        if self.pulser.T0.duration.get() > self.repetition_time.get():
            logging.warning('Trigger cycle of the pulse generator exceeds the '
                            'requested repetition time. Check the settings of '
                            'unused pulser channels.')
            return False
        return True

    def get_ref_points(self):
        ref_points = {'zero': 0.}
        ref_points['awg output start'] = self.awg.trigger_delay.get()
        ref_points['fixed point'] = ref_points['awg output start'] + self.fixed_point_get_cmd()
        return ref_points

    def get_timings(self):
        ref_points = self.get_ref_points()
        timings = {}
        for name, ch in self.submodules.items():
            timings[name] = ch.get_timings(ref_points)
        return timings

    def check_timings(self):
        '''Check timing for consistency.'''
        warnings = 0
        T_rep = self.repetition_time.get()
        timings = self.get_timings()
        # check per-channel constraints
        for channel, timing in timings.items():
            if timing['trig_start'] < 0.:
                raise ValueError('Start of the trigger for {} is before the '
                                 'start of the trigger cycle.'.format(channel))
            if timing['trig_end'] > T_rep - self.PULSER_HOLDOFF:
                raise ValueError('End of the trigger for {} is greater than the '
                                 'repetition time.'.format(channel))
            if (timing['out_end'] is not None) and (timing['out_end'] > T_rep):
                warnings += 1
                logging.warning('In/output of {} extends into the following '
                                'shot.'.format(channel))
        # check special constraints
        if (timings['acq']['trig_start'] < 
            timings['awg']['out_start'] + self.MIN_ACQ_DURATION):
            warnings += 1
            logging.warning('The acquisition trigger must be ' + 
                            'after the start of the AWG output to guarantee ' + 
                            'correct operation of the sequence start trigger.')
        return warnings == 0

    def update(self):
        '''Push timing configuration to pulse generator.'''
        # check for duplicate channel assignments
        for idx, chA in enumerate(self.submodules.values()):
            for chB in tuple(self.submodules.values())[idx+1:]:
                if (chA.channel.get() and 
                    chB.channel.get() and 
                    (chA.channel.get() == chB.channel.get())
                ):
                    raise ValueError(
                        'Channels {} and {} are assigned to the same output on '
                        'the pulse generator.'.format(chA.name, chB.name)
                    )
        # check timings
        self.check_timings()
        # update pulser
        ref_points = self.get_ref_points()
        for ch in self.submodules.values():
            ch.update(ref_points)
        # check if repetition time can be met
        self.check_repetition_time()

    def plot(self):
        '''
        Generate a representation of the timing configuration with matplotlib.
        
        Output:
            (Figure) matplotlib figure showing the timing configuration
        '''
        from matplotlib.patches import Rectangle, Path, PathPatch
        import matplotlib.pyplot as plt
        
        fig = plt.figure()
        ax = fig.add_axes((0.25, 0.125, 0.7, 0.78))
        plt.close(fig)
        
        # scale all timings
        timings = self.get_timings()
        scale = 1e6
        repetition = scale * self.repetition_time.get()
        for timing in timings.values():
            for key, value in timing.items():
                if value is not None:
                    timing[key] *= scale
        
        fig.suptitle('timing configuration in us')
                    
        # (half the) vertical size of the bars in units of bar spacings
        dy = 0.3
        dy_ref = 0.5
        
        ax.set_xlim((0., repetition))
        ax.set_ylim((-2*dy, 2*dy + 2*len(timings) - 1))
        yticklabels = []
        
        line_fixed = ax.vlines(scale*self.get_ref_points()['fixed point'], 
                               *ax.get_ylim(), linestyle='--')

        fill_kwargs_template = dict(facecolor='none', edgecolor='black', 
                                    linestyle='solid', hatch='//')
                
        # iterate over channels/instruments
        for channel_idx, timing_dict in enumerate(reversed(list(timings.items()))):
            channel, timing = timing_dict
            # iterate over trigger and signal bars
            loop_vars = [(0, 'signal', timing['out_start'], timing['out_end']),
                         (1, 'trig', timing['trig_start'], timing['trig_end'])]
            for type_idx, type_name, x0, x1 in loop_vars:
                y = 2*channel_idx + type_idx
        
                # draw labels
                if type_name == 'trig':
                    pulser_ch = self.submodules[channel].channel.get()
                    pulser_ch = 'None' if pulser_ch is None else pulser_ch.name
                    yticklabels.append('{0} {1} ({2})'.format(channel, type_name, 
                                                              pulser_ch))
                else:
                    yticklabels.append('{0} {1}'.format(channel, type_name))
        
                # draw bars
                fill_kwargs = dict(fill_kwargs_template)
                if x1 is None:
                    # if the duration is unknown, draw a box with 10% of the width,
                    # an angled right bound and a question mark
                    x1 = min(repetition, x0 + repetition/10)
                    clip_path = Path([(x0, y - dy), 
                                      (x0 + 1.0*(x1 - x0), y - dy), 
                                      (x0 + 0.7*(x1 - x0), y + dy),
                                      (x0, y + dy), 
                                      (x0, y - dy)])
                    clip_path = PathPatch(clip_path, facecolor='none', 
                                          linestyle='dashed')
                    ax.add_patch(clip_path)
                    ax.text(x1, y, '?', fontsize=12)
                    fill_kwargs.update(clip_path=clip_path, linestyle='dashed')
                if x1 > repetition:
                    # if the pulse ends after the repetition time, fold it over
                    ax.fill([x0, x0, repetition, repetition], 
                            [y - dy, y + dy, y + dy, y - dy], **fill_kwargs)
                    ax.fill([0., 0., x1 - repetition, x1 - repetition], 
                            [y - dy, y + dy, y + dy, y - dy], **fill_kwargs)
                else:
                    ax.fill([x0, x0, x1, x1], [y - dy, y + dy, y + dy, y - dy],
                            **fill_kwargs)
        
                reference = timing['reference']
                line_reference = ax.vlines(reference, y - dy_ref, y + dy_ref, 
                                           linestyle=':')
        
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels, size=12)

        # place a legend below the axes
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        line_out = Rectangle((0, 0), 1., 1., **fill_kwargs_template)
        ax.legend((line_fixed, line_reference, line_out),
                  ('fixed point', 'reference point', 'signal on'), ncol=3, 
                  loc='upper right', bbox_to_anchor=(1.02, -0.075))

        return fig

    def awg_plot(self, seq, interactive=True):
        if not seq.sampled_sequences:
            seq.sample() 

        import numpy as np

        fig = self.plot()
        ax = fig.axes[0]
        
        awg_start = self.get_timings()['awg']['out_start']

        scale = 1e6

        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top + 3.2)

        yticklabels = list(ax.get_yticklabels())
        yticks = list(ax.get_yticks())

        # for waveform in seq.sampled_sequences[channel].waveforms:
        waveform = seq.sampled_sequences[0].waveforms[0]
        sampling_rate = seq.channels[0].sampling_freq
        waveform_line, = ax.plot(np.arange(scale*awg_start, scale*awg_start + scale*len(waveform)/sampling_rate, scale/sampling_rate),  top + 2 + waveform)
        yticks.append(top + 2)
        yticklabels.append('waveform')

        # for markers in seq.sampled_sequences[channel].markers:
        markers = seq.sampled_sequences[0].markers[0]
        sampling_rate = seq.channels[0].sampling_freq / seq.channels[0].awg.granularity
        marker_lines = []
        for i, marker in enumerate(markers[::-1]):
            marker_line, = ax.plot(np.arange(scale*awg_start, scale*awg_start + scale*len(marker)/sampling_rate, scale/sampling_rate), top + 0.5*i + 0.4*marker)
            marker_lines.append(marker_line)
            yticks.append(top + 0.5*i)
            yticklabels.append('marker{:}'.format(i))

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, size=12)

        def update_plots(channel, segment):
            try:
                waveform = seq.sampled_sequences[channel].waveforms[segment]
                markers = seq.sampled_sequences[channel].markers[segment]
            except IndexError:
                waveform = 0*seq.sampled_sequences[0].waveforms[0]
                markers = [0*seq.sampled_sequences[0].markers[0][0], 0*seq.sampled_sequences[0].markers[0][0]]

            waveform_line.set_ydata(top + 2 + waveform)

            for i, marker in enumerate(markers[::-1]):
                marker_lines[i].set_ydata(top + 0.5*i + 0.4*marker)

            fig.canvas.draw_idle()

        if not interactive:
            return fig, update_plots
        import matplotlib.pyplot as plt
        from ipywidgets import interact
        import ipywidgets as widgets
        interact(update_plots, channel=widgets.IntSlider(min=0, max=len(seq.sampled_sequences) - 1, step=1),
                       segment=widgets.IntSlider(min=0, max=len(seq.sampled_sequences[0].waveforms) - 1, step=1))
        # dummy = plt.figure()
        # new_manager = dummy.canvas.manager
        # new_manager.canvas.figure = fig
        # fig.set_canvas(new_manager.canvas)
        # plt.show()
        return fig
