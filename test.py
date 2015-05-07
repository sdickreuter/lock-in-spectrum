__author__ = 'dunkelfeld'

import time
import seabreeze.spectrometers as sb
from gi.repository import Gtk


devices = sb.list_devices()
s = sb.Spectrometer(devices[0])
s.tec_set_temperature_C(-17)
s.tec_set_enable(True)
for i in range(10) :
    print(s.tec_get_temperature_C())
    time.sleep(0.3)

s.integration_time_micros(100000)
print(s.intensities())
print(s.wavelengths())
