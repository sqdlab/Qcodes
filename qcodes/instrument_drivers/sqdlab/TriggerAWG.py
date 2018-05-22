from qcodes import Instrument

class TriggerAWG(Instrument):
    def __init__(self, name, parameter, disable_val, enable_val, **kwargs):
        '''
        A trigger source with AWG interface. Can be added to the list of AWGs 
        to inhibit triggers while the AWGs are programmed.

        Arguments:
            parameter: `Parameter`
                parameter that is set to start/stop triggering
            stop_value
                value of parameter that disables the trigger source
            start_value
                value of parameter that enables the trigger_source
        '''
        super().__init__(name, **kwargs)
        self._parameter = parameter
        self._disable_val = disable_val
        self._enable_val = enable_val

    def run(self):
        '''Enable triggers.'''
        self._parameter.set(self._enable_val)
        
    def stop(self):
        '''Disable triggers.'''
        self._parameter.set(self._disable_val)
        
    #def load_sequence(self, path, filename, append=False):
    #    pass