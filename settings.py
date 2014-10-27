__author__ = 'sei'

import configparser
import os

class Settings(object):
    _filename = "config.ini"

    def __init__(self):
        self.config = configparser.ConfigParser()
        try:
            self.config.read(self._filename)
            self.integration_time = self.config.getfloat('spectrum', 'integration_time')
            self.number_of_samples = self.config.getint('spectrum', 'number_of_samples')

            self.direction_x = self.config.getfloat('direction', 'x')
            self.direction_y = self.config.getfloat('direction', 'y')
            self.direction_z = self.config.getfloat('direction', 'z')
            self.amplitude = self.config.getint('direction', 'amplitude')

            self.stepsize = self.config.getfloat('stage', 'stepsize')

            self.sigma = self.config.getfloat('searchmax', 'sigma')
            self.rasterdim = self.config.getint('searchmax', 'rasterdim')
            self.rasterwidth = self.config.getfloat('searchmax', 'rasterwidth')


        except:
            RuntimeError("Error loading settings.")
            # self.integration_time = 0.08
            # self.number_of_samples = 1000
            return

    def save(self):
        try:
            self.config.set('spectrum', 'integration_time', self.integration_time)
            self.config.set('spectrum', 'number_of_samples', self.number_of_samples)
            self.config.set('direction', 'x', self.direction_x)
            self.config.set('direction', 'y', self.direction_y)
            self.config.set('direction', 'z', self.direction_z)
            self.config.set('direction', 'amplitude', int(self.amplitude))
            self.config.set('stage', 'stepsize', self.stepsize)
            self.config.set('searchmax', 'sigma', self.sigma)
            self.config.set('searchmax', 'rasterdim', int(self.rasterdim))
            self.config.set('searchmax', 'rasterwidth', self.rasterwidth)
            f = open(self._filename, "wb")
            self.config.write(f)
            f.close()
        except:
            print("Error saving settings.")
            print(os.getcwd())