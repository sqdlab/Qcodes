from qcodes import VisaInstrument, validators as vals

class YIG(VisaInstrument):
    '''
    Control YIG DAC connected to Raspberry PI SPI pins running a DAC SCPI server

    The old driver asked for *STB? before every set, add that back if required.
    '''
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)
        # add parameters
        self.add_parameter(
            'frequency', label='Center Frequency', unit='Hz',
            get_cmd='SOUR:FREQ?', get_parser=float,
            set_cmd='SOUR:FREQ {:f}', vals=vals.Numbers(2.37e9, 18.6e9))
        self.connect_message()