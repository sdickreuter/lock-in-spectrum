__author__ = 'sei'

import configparser
import os

class Settings(object):
    _filename = "config.ini"

    def __init__(self):
        self.config = configparser.ConfigParser()
        try:
            self.config.read(self._filename)

        except:
            print("Error loading settings.")
            RuntimeError("Error loading settings.")
            # self.integration_time = 0.08
            # self.number_of_samples = 1000
            return

        self.integration_time = int(self.config['spectrum']['integration_time'])
        self.number_of_samples = int(self.config['spectrum']['number_of_samples'])

        self.direction_x = float(self.config['direction']['x'])
        self.direction_y = float(self.config['direction']['y'])
        self.direction_z = float(self.config['direction']['z'])
        self.amplitude = float(self.config['direction']['amplitude'])

        self.stepsize = float(self.config['stage']['stepsize'])

        self.sigma = float(self.config['searchmax']['sigma'])
        self.rasterdim = int(self.config['searchmax']['rasterdim'])
        self.rasterwidth = float(self.config['searchmax']['rasterwidth'])



    def save(self):
        self.config.set('spectrum', 'integration_time', str(self.integration_time))
        self.config.set('spectrum', 'number_of_samples', str(self.number_of_samples))
        self.config.set('direction', 'x', str(self.direction_x))
        self.config.set('direction', 'y', str(self.direction_y))
        self.config.set('direction', 'z', str(self.direction_z))
        self.config.set('direction', 'amplitude', str(self.amplitude))
        self.config.set('stage', 'stepsize', str(self.stepsize))
        self.config.set('searchmax', 'sigma', str(self.sigma))
        self.config.set('searchmax', 'rasterdim', str(self.rasterdim))
        self.config.set('searchmax', 'rasterwidth', str(self.rasterwidth))
        try:
            with open(self._filename, 'w') as configfile:
                self.config.write(configfile)
        except:
            print("Error saving settings.")
            print(os.getcwd())