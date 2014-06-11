__author__ = 'sei'

import time
from logger import logger
import threading
from itertools import cycle
import numpy as np

from gi.repository import Gtk
from gi.repository import GObject
from gi.repository import GLib

class mpl:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas


class lockin_gui(object):
    _window_title = "GTK_CV_test"
    _heartbeat = 200  # s

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

        self.status = Gtk.Label(label="Initialized")
        self.progress = Gtk.ProgressBar()
        self._progress_fraction = 0
        self.progress.set_fraction(self._progress_fraction)

        self.sidebox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        self.sidebox.add(self.button_live)
        self.sidebox.add(self.button_aquire)
        self.sidebox.add(self.button_stagetostart)
        self.sidebox.add(self.button_save)

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

        self.window.connect("delete-event", Gtk.main_quit)
        self.button_aquire.connect("clicked", self.on_aquire_clicked)
        self.button_live.connect("clicked", self.on_live_clicked)
        self.button_stagetostart.connect("clicked", self.on_stagetostart_clicked)
        self.button_save.connect("clicked", self.on_save_clicked)


        self.window.show_all()


        # Thread stuff
        self.worker_running_event = threading.Event()
        self.worker_thread = None
        self.worker_mode = None
        self.worker_lock = threading.Lock()

        self._plotting = True

        self.log = logger()
        self._spec = self.log.get_spec()
        self.line, = self.ax.plot(self.log.get_wl(), self._spec)


        time.sleep(2)

    def start_thread(self, target, mode):
        self.worker_mode = mode
        self.worker_running_event.clear()
        self.worker_thread = threading.Thread(target=target, args=(self.worker_running_event,))
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def stop_thread(self):
        self.worker_running_event.set()
        self.worker_thread.join(1)
        self.worker_thread = None

    def on_aquire_clicked(self, button):

        if self.worker_thread is None:
            self.log.reset()
            self.status.set_label('Acquiring ...')
            self.start_thread(self.acquire_spectrum, 'acquire')
        else:
            self.status.set_label('Paused')
            self.stop_thread()
            if self.worker_mode is 'live':
                self.status.set_label('Acquiring ...')
                self.start_thread(self.acquire_spectrum, 'acquire')

    def on_live_clicked(self, button):

        if self.worker_thread is None:
            self.status.set_label('Liveview')
            self.start_thread(self.live_spectrum, 'live')
        else:
            self.status.set_label('Paused')
            self.stop_thread()
            if self.worker_mode is 'acquire':
                self.status.set_label('Liveview')
                self.start_thread(self.live_spectrum, 'live')

    def on_save_clicked(self, button):
        if not self.log.spectra is None:
            self.log.save_data()
            self.log.reset()
            self.status.set_label('Data saved')
        else:
            self.status.set_label('No Data found')

    def on_stagetostart_clicked(self, button):
        self.log.stage_to_starting_point()

    def acquire_spectrum(self, e):
        self._plotting = False
        while True:
            #self._spec, running = self.log.measure_spectrum()
            running = self.log.measure_spectrum()

            self._progress_fraction =  float(self.log.get_scan_index()) / self.log.get_number_of_samples()

            if not running:
                self._spec = self.calc_lock_in()
                self.status.set_label('Spectra acquired')
                break

            if e.is_set():
                self.log.reset()
                break
        self._plotting = True
        return True

    def live_spectrum(self, e):
        while not e.is_set():
            self._spec = self.log.get_spec()
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
            GLib.timeout_add(self._heartbeat, self._update_title)
            #GLib.timeout_add(self._heartbeat, self.update_progress)
            GLib.timeout_add(self.log._integration_time/1000, self.update_plot)
            Gtk.main()
        except KeyboardInterrupt:
            pass

    def _update_title(self, _suff=cycle('/|\-')):
        self.window.set_title('%s %s' % (self._window_title, next(_suff)))
        self.progress.set_fraction(self._progress_fraction)
        return True

    def calc_lock_in(self):
        shape = self.log.data.shape
        res = np.empty(1024)
        diff = np.diff( np.append(0,self.log.data[:,0]) )
        ref = self.log.data[:,1]
        print diff
        for i in range(2,1026):
            buf = self.log.data[:,i]
            buf = buf*diff*ref
            buf = np.sum(buf)
            #res = np.append(res,buf)
            res[i-2] = buf
        return res


if __name__ == "__main__":
    gui = lockin_gui()
    gui.run()
