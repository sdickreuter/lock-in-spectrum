__author__ = 'sei'

from datetime import datetime
import math
import threading

#from PIStage._base import Controller
import oceanoptics
import numpy as np

class logger(object):
    # Spectrum Aquisition
    _spectrometer = None
    _wl = None
    _filename = None
    _integration_time = 0.1
    _scan_index = 0
    _number_of_samples = 1000

    # Stage control
    _stage_amplitude = 5  # amplitude in um
    # _cycle_time = 0  # cycle duration in s
    #_cycle_time_start = 10 # cycle duration in s, staring value
    #_cycle_factor = 0.2 # cycle time is calculated using this factor
    _cycle_time = 0  # cycle duration in s
    _cycle_time_start = 3 * _integration_time * 1000 / 10  # cycle duration in s, starting value
    _cycle_factor = -float(180) / _number_of_samples  # cycle time is calculated using this factor

    #General
    _starttime = None
    _juststarted = True
    _new_spectrum = False

    def __init__(self, stage, settings):
        self.worker_thread = None
        self._init_spectrometer()
        self.stage = stage
        self.settings = settings
        #self._init_stage()

    def _init_stage(self):
        #self.stage = nano.NanoControl()
        #self.stage = PI.Dummy()
        pass

    def _millis(self):
        dt = datetime.now() - self._starttime
        ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        return ms

    def set_integration_time(self, integration_time):
        self._integration_time = integration_time
        self._cycle_time_start = 3 * self._integration_time * 1000 / 10  # cycle duration in s, starting value
        self._spectrometer.integration_time(self._integration_time)

    def set_number_of_samples(self, number_of_samples):
        self._number_of_samples = number_of_samples
        self._cycle_factor = -float(180) / self._number_of_samples  # cycle time is calculated using this factor

    def _init_spectrometer(self):
        try:
            #self._spectrometer = oceanoptics.QE65000()
            self._spectrometer = oceanoptics.Dummy()
            self._spectrometer.integration_time(self._integration_time)
            sp = self._spectrometer.spectrum()
            self._wl = sp[0]
            self.spectra = None
            self.data = np.zeros((self._number_of_samples, 1026), dtype=np.float64)
        except:
            raise RuntimeError("Error opening spectrometer. Exiting...")

    def get_wl(self):
        return self._wl

    def get_spec(self):
        return self._spectrometer.intensities()

    def get_scan_index(self):
        return self._scan_index

    def get_number_of_samples(self):
        return self._number_of_samples

    def reset(self):
        self._new_spectrum = False
        self.data = np.zeros((self._number_of_samples, 1026), dtype=np.float64)
        self._juststarted = True
        self._scan_index = 0

    def move_stage(self, amplitude):
        dx = self._stage_amplitude * amplitude * self.settings.direction_x
        dy = self._stage_amplitude * amplitude * self.settings.direction_y
        dz = self._stage_amplitude * amplitude * self.settings.direction_z
        self.stage.moverel(dx, dy, dz)

    def measure_spectrum(self):
        if self._juststarted:
            self._starttime = datetime.now()
            self._juststarted = False
            self._scan_index = 0

        t = self._millis() / 1000

        self._cycle_time = self._cycle_factor * t + self._cycle_time_start
        ref = math.cos(2 * math.pi / self._cycle_time * t)
        #print "Val: {0:6} | t: {1:.3f}".format(int(A*sin_value),t) + '  ' + '#'.rjust(int(10*sin_value+10))
        if not self.worker_thread is None: self.worker_thread.join(1)
        self.worker_thread = threading.Thread(target=self.move_stage,args=(ref,))
        self.worker_thread.daemon = True
        self.worker_thread.start()

        data = self._spectrometer.intensities()

        self.data[self._scan_index, 0] = t
        self.data[self._scan_index, 1] = ref
        self.data[self._scan_index, 2:] = data

        self._scan_index += 1

        if self._scan_index >= self._number_of_samples:
            print("%s spectra aquired" % self._scan_index)
            print("time taken: %s s" % t )
            self.worker_thread.join(1)
            self._new_spectrum = True
            return data, False

        return data, True

