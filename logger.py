__author__ = 'sei'

import NanoControl as nano
import oceanoptics
from datetime import datetime
import pandas
import math

class logger(object):
    # Spectrum Aquisition
    _spectrometer = None
    _wl = None
    _filename = None
    _integration_time = 10
    _scan_index = 0
    _number_of_samples = 3000

    #Stage control
    _stage_amplitude = 2047 # amplitude in nm
    _cycle_time = 5 # cycle duration in s

    #General
    _starttime = None
    _juststarted = True

    def __init__(self):

        self._init_spectrometer()
        self._init_nanocontrol()

    def _init_nanocontrol(self):
        self.stage = nano.NanoControl()

    def millis(self):
        dt = datetime.now() - self._starttime
        ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        return ms

    def _init_spectrometer(self):
        try:
            #self.spectrometer = oceanoptics.STS()
            self._spectrometer = oceanoptics.QE65000()
            self._spectrometer.integration_time(self._integration_time)
            sp = self._spectrometer.spectrum()
            self._wl = sp[0]
            #serial = self.spectrometer.Serial
            self._filename = 'samples_' + str(self._number_of_samples) + '_inttime_' + str(
                self._integration_time) + '__' + str(datetime.now().year) + str(datetime.now().month) \
                + str(datetime.now().day) + '.csv'
            self.spectra = pandas.DataFrame()
            #self._starttime = datetime.now()
        except:
            raise RuntimeError("Error opening spectrometer. Exiting...")

    def stage_to_starting_point(self):
        self.stage._fine('B',2047)

    def save_data(self):
        self.spectra.to_csv(self._filename, header=False, mode = 'a')

    def reset(self):
        self.spectra = pandas.DataFrame()
        self._juststarted = True
        self._scan_index = 0

    def measure_spectrum(self):
        if self._juststarted:
            self._starttime = datetime.now()
            self._juststarted = False

        self._scan_index += 1

        t = self._millis()/1000

        sin_value = math.cos(2*math.pi/float(self._cycle_time)*t)
        #print "Val: {0:6} | t: {1:.3f}".format(int(A*sin_value),t) + '  ' + '#'.rjust(int(10*sin_value+10))
        self.stage._fine('B',self._stage_amplitude*sin_value)

        #print("Aquiring: %s" % self.scan_index)
        int = self._spectrometer.intensities()
        self.spectra.append((self._scan_index,t,sin_value, int))

        if self._scan_index >= self._number_of_samples:
            print("%s spectra aquired" % self._scan_index)
            print("time taken: %s s" % t )
            self.stage_to_starting_point()
            return int,False

        return int,True

