__author__ = 'sei'

from gi.repository import Gtk
from gi.repository import GObject
from gi.repository import GLib
import math

class Settings_Dialog(Gtk.Dialog):

    def __init__(self, parent, settings):
       self.settings = settings
       #super(Settings_Dialog, self).__init__(self, "Settings", parent, 0, (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,Gtk.STOCK_OK, Gtk.ResponseType.OK))
       Gtk.Dialog.__init__(self, "Settings", parent, 0, (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OK, Gtk.ResponseType.OK))
       #self.set_default_size(150, 100)

       self.box = self.get_content_area()
       self.box.set_spacing(6)

       # Spinbuttons
       self.integration_time_adj = Gtk.Adjustment(value=10, lower=80, upper=1000, step_incr=10, page_incr=10,
                                                   page_size=0)
       self.integration_time_spin = Gtk.SpinButton(adjustment=self.integration_time_adj, climb_rate=0.1, digits=0)
       self.integration_time_spin.set_tooltip_text("Set Integration time of the spectrometer")
       self.number_of_samples_adj = Gtk.Adjustment(value=1000, lower=100, upper=10000, step_incr=100, page_incr=10,
                                                   page_size=0)
       self.number_of_samples_spin = Gtk.SpinButton(adjustment=self.number_of_samples_adj, climb_rate=0.1, digits=0)
       self.number_of_samples_spin.set_tooltip_text("Set how many samples are taken at each run")

       self.number_of_samples_spin.set_value(self.settings.number_of_samples)
       self.integration_time_spin.set_value(self.settings.integration_time)

       self.box.add(Gtk.Label(label="Integration Time [s]"))
       self.box.add(self.integration_time_spin)
       self.box.add(Gtk.Label(label="Number of Samples"))
       self.box.add(self.number_of_samples_spin)

       self.hide()


    def rundialog(self):
        self.show_all()
        result = self.run()
        if (result==Gtk.ResponseType.OK):
            self.settings.number_of_samples = self.number_of_samples_spin.get_value_as_int()
    	    self.settings.integration_time = self.integration_time_spin.get_value_as_int()
            self.settings.save()
        else:
           self.number_of_samples_spin.set_value(self.settings.number_of_samples)
           self.integration_time_spin.set_value(self.settings.integration_time)

        self.hide()

    def enable_number_of_samples(self):
        self.number_of_samples_spin.set_sensitive(False)  # disable spinbutton which sets number of samples

    def disable_number_of_samples(self):
        self.number_of_samples_spin.set_sensitive(True)  # re-enable spinbutton which sets number of samples



class Direction_Dialog(Gtk.Dialog):

    def __init__(self, parent, settings):
       self.settings = settings
       #super(Settings_Dialog, self).__init__(self, "Settings", parent, 0, (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,Gtk.STOCK_OK, Gtk.ResponseType.OK))
       Gtk.Dialog.__init__(self, "Direction of Stage Movement", parent, 0, (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OK, Gtk.ResponseType.OK))
       #self.set_default_size(150, 100)

       self.box = self.get_content_area()
       self.box.set_spacing(6)

       #buttons

       # Spinbuttons
       self.x_adj = Gtk.Adjustment(value=1, lower=0, upper=1, step_incr=0.1, page_incr=0.1, page_size=0)
       self.y_adj = Gtk.Adjustment(value=0, lower=0, upper=1, step_incr=0.1, page_incr=0.1, page_size=0)
       self.z_adj = Gtk.Adjustment(value=0, lower=0, upper=1, step_incr=0.1, page_incr=0.1, page_size=0)

       self.x_spin = Gtk.SpinButton(adjustment=self.x_adj, climb_rate=0.1, digits=2)
       self.y_spin = Gtk.SpinButton(adjustment=self.y_adj, climb_rate=0.1, digits=2)
       self.z_spin = Gtk.SpinButton(adjustment=self.z_adj, climb_rate=0.1, digits=2)

       #self.number_of_samples_spin.set_value(self.settings.number_of_samples)
       #self.integration_time_spin.set_value(self.settings.integration_time)
       self.box.add(Gtk.Label(label="x"))
       self.box.add(self.x_spin)
       self.box.add(Gtk.Label(label="y"))
       self.box.add(self.y_spin)
       self.box.add(Gtk.Label(label="z"))
       self.box.add(self.z_spin)

       self.x_spin.connect("value-changed", self.on_change)
       self.y_spin.connect("value-changed", self.on_change)
       self.z_spin.connect("value-changed", self.on_change)

       self.hide()

    def _normalize(self,x,y,z):
        l = math.sqrt( math.pow(x,2) + math.pow(y,2) + math.pow(z,2))
        x = x/l
        y = y/l
        z = z/l
        return x,y,z

    def on_change(self, widget):
        x = self.x_spin.get_value()
        y = self.y_spin.get_value()
        z = self.z_spin.get_value()
        x,y,z = self._normalize(x,y,z)
        self.x_spin.set_value(x)
        self.y_spin.set_value(y)
        self.z_spin.set_value(z)


    def rundialog(self):
        self.show_all()
        result = self.run()
        if (result==Gtk.ResponseType.OK):
            self.settings.direction_x = self.x_spin.get_value()
    	    self.settings.direction_y = self.y_spin.get_value()
    	    self.settings.direction_z = self.z_spin.get_value()
    	    self.settings.save()
        else:
            self.x_spin.set_value(self.settings.direction_x)
            self.y_spin.set_value(self.settings.direction_y)
            self.z_spin.set_value(self.settings.direction_z)
        self.hide()
