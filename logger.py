__author__ = 'sei'

from datetime import datetime
import math
import time

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

    _cycle_time = 0  # cycle duration in s
    _cycle_time_start = 0  # cycle duration in s, starting value
    _cycle_factor = 0  # cycle time is calculated using this factor

    #General
    _starttime = None
    _juststarted = True
    _new_spectrum = False



    def __init__(self, stage, settings):
        self.settings = settings
        self.worker_thread = None
        self.stage = stage
        self._init_spectrometer()
        self._cycle_time_start = 250
        self._startx = 10
        self._starty = 10
        self._startz = 10


    def _millis(self):
        dt = datetime.now() - self._starttime
        ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        return ms

    def set_integration_time(self, integration_time):
        self._integration_time = integration_time/1000
        #self._cycle_time_start = 3 * self._integration_time * 1000 / 10  # cycle duration in s, starting value
        self._spectrometer.integration_time(self._integration_time)

    def _init_spectrometer(self):
        try:
            #self._spectrometer = oceanoptics.QE65000()
            #self._spectrometer = oceanoptics.Dummy()
            self._spectrometer = oceanoptics.ParticleDummy(stage=self.stage)
            self._spectrometer.integration_time(self._integration_time)
            sp = self._spectrometer.spectrum()
            self._wl = sp[0]
            self.spectra = None
            self.data = np.zeros((self.settings.number_of_samples, 1026), dtype=np.float64)
        except:
            raise RuntimeError("Error opening spectrometer. Exiting...")

    def get_wl(self):
        return self._wl

    def get_spec(self):
        return self._spectrometer.intensities()

    def get_scan_index(self):
        return self._scan_index

    def reset(self):
        self._new_spectrum = False
        self.data = np.zeros((self.settings.number_of_samples, 1026), dtype=np.float64)
        self._juststarted = True
        self._scan_index = 0
        #self.stage.moveabs(self._startx,self._starty,self._startz)

    def move_stage(self, dist):
        x = self._startx + self.settings.amplitude/2 * (dist) * self.settings.direction_x
        y = self._starty + self.settings.amplitude/2 * (dist) * self.settings.direction_y
        z = self._startz + self.settings.amplitude/2 * (dist) * self.settings.direction_z
        #print "X: {0:+8.4f} | Y: {1:8.4f} | Z: {2:8.4f} || X: {3:+8.4f} | Y: {4:8.4f} | Z: {5:8.4f}".format(x,y,z,self._startx,self._starty,self._startz)
        self.stage.moveabs(x=x, y=y, z=z)
        #self.stage.moveabs(x=x)

    def measure_spectrum(self):
        if self._juststarted:
            self._cycle_factor = -1.0 / (6.0 * self.settings.number_of_samples/1000)  # cycle time is calculated using this factor
            self._juststarted = False
            self._scan_index = 0
            pos = self.stage.pos()
            self._startx = pos[0]
            self._starty = pos[1]
            self._startz = pos[2]
            #print "measure_spectrum {0:+8.4f} {1:+8.4f} {2:+8.4f}".format(self._startx, self._starty, self._startz)
            self._starttime = datetime.now()

        self._cycle_time = self._cycle_factor * self._scan_index + self._cycle_time_start
        ref = math.cos(2 * math.pi * self._scan_index / self._cycle_time)
        self.move_stage((-ref+1)/2)
        #self.stage.pos()
        data = self._spectrometer.intensities()

        self.data[self._scan_index, 0] = self._scan_index
        self.data[self._scan_index, 1] = ref
        self.data[self._scan_index, 2:] = data

        self._scan_index += 1

        if self._scan_index >= self.settings.number_of_samples:
            print("%s spectra aquired" % self._scan_index)
            print("time taken: %s s" % (self._millis() / 1000) )
            #self.worker_thread.join(1)
            self.stage.moveabs(self._startx,self._starty,self._startz)
            self._new_spectrum = True
            return data, False

        return data, True

