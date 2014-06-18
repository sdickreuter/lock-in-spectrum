import pandas

__author__ = 'sei'

import time
from logger import logger
import threading
from itertools import cycle
import numpy as np
from datetime import datetime

from gi.repository import Gtk
from gi.repository import GObject
from gi.repository import GLib

class mpl:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas


class lockin_gui(object):
    _window_title = "Lock-in Spectrum"
    _heartbeat = 250  # s

    def __init__(self):
        GObject.threads_init()  # all Gtk is in the main thread;
        # only GObject.idle_add() is in the background thread
        self.window = Gtk.Window(title=self._window_title)
        self.window.set_resizable(False)
        self.window.set_border_width(3)

        self.grid = Gtk.Grid()
        self.grid.set_row_spacing(5)
        self.grid.set_column_spacing(5)
        self.window.add(self.grid)

        self.button_live = Gtk.Button(label="Liveview")
        self.button_aquire = Gtk.Button(label="Aquire Spectrum")
        self.button_stagetostart = Gtk.Button(label="Stage to Start Pos.")
        self.button_save = Gtk.Button(label="Save Data")
        self.button_dark = Gtk.Button(label="Take Dark Spectrum")
        self.button_lamp = Gtk.Button(label="Take Lamp Spectrum")
        self.button_normal = Gtk.Button(label="Take Normal Spectrum")


        self.window.connect("delete-event", self.quit)
        self.button_aquire.connect("clicked", self.on_aquire_clicked)
        self.button_live.connect("clicked", self.on_live_clicked)
        self.button_stagetostart.connect("clicked", self.on_stagetostart_clicked)
        self.button_save.connect("clicked", self.on_save_clicked)
        self.button_dark.connect("clicked", self.on_dark_clicked)
        self.button_lamp.connect("clicked", self.on_lamp_clicked)
        self.button_normal.connect("clicked", self.on_normal_clicked)


        self.status = Gtk.Label(label="Initialized")
        self.progress = Gtk.ProgressBar()
        self._progress_fraction = 0
        self.progress.set_fraction(self._progress_fraction)

        self.sidebox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        self.sidebox.add(self.button_live)
        self.sidebox.add(self.button_aquire)
        self.sidebox.add(self.button_stagetostart)
        self.sidebox.add(self.button_save)
        self.sidebox.add(self.button_dark)
        self.sidebox.add(self.button_lamp)
        self.sidebox.add(self.button_normal)


        # MPL stuff
        self.figure = mpl.Figure()
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.ax.grid(True)
        self.canvas = mpl.FigureCanvas(self.figure)
        # self.line, = self.ax.plot(self.wl, self.sp[:,0])

        self.canvas.set_size_request(600, 600)
        self.sidebox.set_size_request(100, -1)
        self.progress.set_size_request(-1, 15)
        self.status.set_size_request(100, -1)

        self.grid.add(self.canvas)
        self.grid.attach_next_to(self.sidebox, self.canvas, Gtk.PositionType.RIGHT, 1, 1)
        self.grid.attach_next_to(self.progress, self.canvas, Gtk.PositionType.BOTTOM, 1, 1)
        self.grid.attach_next_to(self.status, self.sidebox, Gtk.PositionType.BOTTOM, 1, 1)

        self.window.show_all()


        # Thread stuff
        self.worker_running_event = threading.Event()
        self.worker_thread = None
        self.worker_mode = None
        self.worker_lock = threading.Lock()

        self._plotting = True

        self.log = logger()
        self._spec = self.log.get_spec()
        self._wl = self.log.get_wl()
        self.line, = self.ax.plot(self._wl, self._spec)

        self.lamp = None
        self.dark = None
        self.normal = None


    def quit(self,*args):
        if not self.worker_thread is None:
            self.stop_thread()
        self.log = None
        Gtk.main_quit(*args)

    def start_thread(self, target, mode):
        self.worker_mode = mode
        self.worker_running_event.clear()
        if not self.worker_thread is None: self.worker_thread.join(0.2)
        self.worker_thread = threading.Thread(target=target, args=(self.worker_running_event,))
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def stop_thread(self):
        self.worker_running_event.set()
        self.worker_thread.join(1)
        self.worker_thread = None

    def on_aquire_clicked(self, widget):
        if self.worker_thread is None:
            self.log.reset()
            self.status.set_label('Acquiring ...')
            self.start_thread(self.acquire_spectrum, 'acquire')
        else:
            self.status.set_label('Paused')
            self.stop_thread()
            if not self.worker_mode is 'acquire':
                self.log.reset()
                self.status.set_label('Acquiring ...')
                self.start_thread(self.acquire_spectrum, 'acquire')

    def on_live_clicked(self, widget):
        if self.worker_thread is None:
            self.status.set_label('Liveview')
            self.start_thread(self.live_spectrum, 'live')
        else:
            self.status.set_label('Paused')
            self.stop_thread()
            if not self.worker_mode is 'live':
                self.status.set_label('Liveview')
                self.start_thread(self.live_spectrum, 'live')

    def on_save_clicked(self, widget):
        if self.log._new_spectrum:
            self.status.set_label("Saving Data ...")
            self.save_data()
            self.log.reset()
            self.status.set_label('Data saved')
        else:
            self.status.set_label('No Data found')

    def on_stagetostart_clicked(self, widget):
        self.log.stage_to_starting_point()

    def on_dark_clicked(self, widget):
        if self.worker_thread is None:
            self.status.set_label('Taking Dark Spectrum')
            self.start_thread(self.take_spectrum, 'dark')
        else:
            self.status.set_label('Paused')
            self.stop_thread()
            if not self.worker_mode is 'dark':
                self.status.set_label('Taking Dark Spectrum')
                self.start_thread(self.take_spectrum, 'dark')

    def on_lamp_clicked(self, widget):
         if self.worker_thread is None:
            self.status.set_label('Taking Lamp Spectrum')
            self.start_thread(self.take_spectrum, 'lamp')
         else:
            self.status.set_label('Paused')
            self.stop_thread()
            if not self.worker_mode is 'lamp':
                self.status.set_label('Taking Lamp Spectrum')
                self.start_thread(self.take_spectrum, 'lamp')

    def on_normal_clicked(self, widget):
         if self.worker_thread is None:
            self.status.set_label('Taking Normal Spectrum')
            self.start_thread(self.take_spectrum, 'normal')
         else:
            self.status.set_label('Paused')
            self.stop_thread()
            if not self.worker_mode is 'normal':
                self.status.set_label('Taking Normal Spectrum')
                self.start_thread(self.take_spectrum, 'normal')

    def take_spectrum(self, e):
        data = np.zeros(1024,dtype=np.float32)
        for i in range(self.log._number_of_samples):
            data = (data + self.log.get_spec())/2

            self._progress_fraction =  float(i+1) / self.log._number_of_samples

            if e.is_set():
                if self.worker_mode is 'dark':
                    self.dark = None
                if self.worker_mode is 'lamp':
                    self.lamp = None
                if self.worker_mode is 'normal':
                    self.normal = None
                break

        if self.worker_mode is 'dark':
            self.dark = data
        if self.worker_mode is 'lamp':
            self.lamp = data
        if self.worker_mode is 'normal':
            self.normal = data
            #if not self.dark is None:
            #    data = data - self.dark
            #    if not self.lamp is None:
            #        data = data/(self.lamp)

        self._spec = data
        self.status.set_label('Spectra taken')
        return True

    def acquire_spectrum(self, e):
        #self._plotting = False
        while True:
            self._spec, running = self.log.measure_spectrum()

            self._progress_fraction =  float(self.log.get_scan_index()) / self.log.get_number_of_samples()

            if not self.dark is None:
                self._spec = self._spec - self.dark
                if not self.lamp is None:
                    self._spec = self._spec/(self.lamp)

            if not running:
                self._spec = self.calc_lock_in()
                self.status.set_label('Spectra acquired')
                break

            if e.is_set():
                self.log.reset()
                break
        return True

    def live_spectrum(self, e):
        while not e.is_set():
            self._spec = self.log.get_spec()
            if not self.dark is None:
                self._spec = self._spec - self.dark
                if not self.lamp is None:
                    self._spec = self._spec/(self.lamp)
        return True

    def update_plot(self):
        if self._plotting:
            self.line.set_ydata(self._spec)
            self.ax.relim()
            self.ax.autoscale_view(False, False, True)
            self.canvas.draw()
        return True

    def update_progress(self):
        self.progress.set_fraction(self._progress_fraction)
        return True

    def run(self):
        """	run main gtk thread """
        try:
            GLib.timeout_add(self._heartbeat, self._update)
            Gtk.main()
        except KeyboardInterrupt:
            pass


    def _update(self, _suff=cycle('/|\-')):
        #self.window.set_title('%s %s' % (self._window_title, next(_suff)))
        self.update_plot()
        self.progress.set_fraction(self._progress_fraction)
        return True

    def calc_lock_in(self):
        shape = self.log.data.shape
        res = np.empty(1024)
        diff = np.diff( np.append(0,self.log.data[:,0]) )
        ref = self.log.data[:,1]
        for i in range(2,1026):
            buf = self.log.data[:,i]
            if not self.dark is None:
                buf = buf - self.dark[i-2]
                if not self.lamp is None:
                    buf = buf/(self.lamp[i-2])
            buf = buf*diff*ref
            buf = np.sum(buf)
            res[i-2] = buf
        return res

    def _gen_filename(self):
        return str(datetime.now().year) + str(datetime.now().month).zfill(2) \
               + str(datetime.now().day).zfill(2)+'_'+str(datetime.now().hour).zfill(2) +\
               str(datetime.now().minute).zfill(2) + str(datetime.now().second).zfill(2) + '.csv'


    def save_data(self):
        filename = self._gen_filename()
        cols = ('t','ref')+ tuple(map(str,np.round(self._wl,1)))
        data = pandas.DataFrame(self.log.data,columns=cols)
        data.to_csv('spectrum_'+filename, header=True,index=False)
        if not self.dark is None:
            data = np.append(np.round(self._wl,1).reshape(self._wl.shape[0],1),self.dark.reshape(self.dark.shape[0],1), 1)
            data = pandas.DataFrame(data,columns=('wavelength','intensity'))
            data.to_csv('dark_'+filename, header=True,index=False)
        if not self.lamp is None:
            data = np.append(np.round(self._wl,1).reshape(self._wl.shape[0],1),self.lamp.reshape(self.lamp.shape[0],1), 1)
            data = pandas.DataFrame(data,columns=('wavelength','intensity'))
            data.to_csv('lamp_'+filename, header=True,index=False)
        if not self.normal is None:
            data = np.append(np.round(self._wl,1).reshape(self._wl.shape[0],1),self.normal.reshape(self.normal.shape[0],1), 1)
            data = pandas.DataFrame(data,columns=('wavelength','intensity'))
            data.to_csv('normal_'+filename, header=True,index=False)



if __name__ == "__main__":
    gui = lockin_gui()
    gui.run()
