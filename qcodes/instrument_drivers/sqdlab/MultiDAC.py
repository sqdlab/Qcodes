from qcodes import VisaInstrument, validators as vals

class MultiDAC(VisaInstrument):
    '''
    Control DACs connected to Raspberry PI SPI pins running a DAC SCPI server
    '''
    def __init__(self, name, address, nch, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)
        # add parameters
        self.add_parameter(
            'nch', label='Number of outputs',
            get_cmd='SOUR:NCH?', get_parser=int,
            set_cmd='SOUR:NCH {:d}', vals=vals.Numbers(min_value=0))

        self.nch.set(nch)
        for ch in range(1, 1+nch):
            prefix = 'SOUR:VOLT{:d}'.format(ch)
            self.add_parameter(
                'voltage{:d}'.format(ch), unit='V', 
                get_cmd=prefix+'?', get_parser=float, 
                set_cmd=prefix+' {:f}', vals=vals.Numbers(-10., 10.*(1-2**(-15)))
            )

        self.connect_message()