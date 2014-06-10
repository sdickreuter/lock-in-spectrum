__author__ = 'sei'

from datetime import datetime
import math

import NanoControl as nano
import oceanoptics
import pandas
import numpy as np

class logger(object):
    # Spectrum Aquisition
    _spectrometer = None
    _wl = None
    _filename = None
    _integration_time = 100
    _scan_index = 0
    _number_of_samples = 3000

    # Stage control
    _stage_amplitude = 2047  # amplitude in nm
    _cycle_time = 5  # cycle duration in s

    #General
    _starttime = None
    _juststarted = True

    def __init__(self):

        self._init_spectrometer()
        self._init_nanocontrol()

    def _init_nanocontrol(self):
        #self.stage = nano.NanoControl()
        self.stage = nano.NanoControl_Dummy()

    def _millis(self):
        dt = datetime.now() - self._starttime
        ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        return ms

    def _init_spectrometer(self):
        try:
            #self.spectrometer = oceanoptics.STS()
            #self._spectrometer = oceanoptics.QE65000()
            self._spectrometer = oceanoptics.Dummy()
            self._spectrometer.integration_time(self._integration_time)
            sp = self._spectrometer.spectrum()
            self._wl = sp[0]
            #serial = self.spectrometer.Serial
            self._filename = self._gen_filename()
            #self.spectra = pandas.DataFrame()
            self.spectra = None
            self.data = np.zeros((self._number_of_samples, 1026))
            #self._starttime = datetime.now()
        except:
            raise RuntimeError("Error opening spectrometer. Exiting...")

    def _gen_filename(self):
        #return 'samples_' + str(self._number_of_samples) + '_inttime_' + str(
        #        self._integration_time) + '__' + str(datetime.now().year) + str(datetime.now().month) \
        #                     + str(datetime.now().day) + '.csv'
        return str(datetime.now().year) + str(datetime.now().month).zfill(2) \
               + str(datetime.now().day).zfill(2)+'_'+str(datetime.now().hour).zfill(2) +\
               str(datetime.now().minute).zfill(2) + str(datetime.now().second).zfill(2) + '.csv'


    def get_wl(self):
        return self._wl

    def get_spec(self):
        return self._spectrometer.intensities()

    def get_scan_index(self):
        return self._scan_index

    def get_number_of_samples(self):
        return self._number_of_samples

    def stage_to_starting_point(self):
        self.stage._fine('B', 2047)

    def save_data(self):
        self._filename = self._gen_filename()
        cols = ('t','ref')+ tuple(map(str,self._wl))
        self.spectra = pandas.DataFrame(self.data,columns=cols)
        self.spectra.to_csv(self._filename, header=False)
        self.spectra = None
        #print self.spectra

    def reset(self):
        self.spectra = None
        self.data = np.zeros((self._number_of_samples, 1026))
        self._juststarted = True
        self._scan_index = 0

    def measure_spectrum(self):
        if self._juststarted:
            self._starttime = datetime.now()
            self._juststarted = False


        t = self._millis() / 1000

        sin_value = math.cos(2 * math.pi / float(self._cycle_time) * t)
        #print "Val: {0:6} | t: {1:.3f}".format(int(A*sin_value),t) + '  ' + '#'.rjust(int(10*sin_value+10))
        self.stage._fine('B', self._stage_amplitude * sin_value)

        #print("Aquiring: %s" % self.scan_index)
        data = self._spectrometer.intensities()

        self.data[self._scan_index, 0] = t
        self.data[self._scan_index, 1] = sin_value
        self.data[self._scan_index, 2:] = data

        self._scan_index += 1

        if self._scan_index >= self._number_of_samples:
            print("%s spectra aquired" % self._scan_index)
            print("time taken: %s s" % t )
            self.spectra = pandas.DataFrame()
            self.stage_to_starting_point()
            return data, False

        return data, True

