__author__ = 'dunkelfeld'

import time
from gi.repository import Gtk
import seabreeze.spectrometers as sb

devices = sb.list_devices()
s = sb.Spectrometer(devices[0])
print(s.intensities())
print(s.wavelengths())
