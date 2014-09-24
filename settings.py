__author__ = 'sei'

import ConfigParser

class Settings(object):
    _filename = "config.ini"

    def __init__(self):
        self.config = ConfigParser.ConfigParser()
        try:
            self.config.read(self._filename)
            self.integration_time = self.config.getfloat('spectrum', 'integration_time')
            self.number_of_samples = self.config.getint('spectrum', 'number_of_samples')
            self.direction_x = self.config.getfloat('direction', 'x')
            self.direction_y = self.config.getfloat('direction', 'y')
            self.direction_z = self.config.getfloat('direction', 'z')
            self.amplitude = self.config.getint('direction', 'amplitude')

        except:
            RuntimeError("Error loading settings.")
            #self.integration_time = 0.08
            #self.number_of_samples = 1000
            return

    def save(self):
        """cPickle the information into the file"""
        try:
            self.config.set('spectrum', 'integration_time', self.integration_time)
            self.config.set('spectrum', 'number_of_samples', self.number_of_samples)
            self.config.set('direction', 'x', self.direction_x)
            self.config.set('direction', 'y', self.direction_y)
            self.config.set('direction', 'z', self.direction_z)
            self.config.set('direction', 'amplitude', self.amplitude)
            f = open(self._filename,"wb")
            self.config.write(f)
            f.close()
        except:
            print "Error saving settings."