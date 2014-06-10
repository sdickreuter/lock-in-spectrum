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
    _heartbeat = 100  # s

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

        self.status = Gtk.Label(label="bla")
        self.progress = Gtk.ProgressBar()
        self._progress_fraction = 0
        self.progress.set_fraction(self._progress_fraction)

        self.sidebox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        self.sidebox.add(self.button_live)
        self.sidebox.add(self.button_aquire)
        self.sidebox.add(self.button_stagetostart)

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

        self.window.show_all()


        # Thread stuff
        self.worker_running_event = threading.Event()
        self.worker_thread = None
        self.worker_mode = None


        self.log = logger()
        self.line, = self.ax.plot(self.log.get_wl(), self.log.get_spec())

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
            self.start_thread(self.acquire_spectrum, 'acquire')
        else:
            self.stop_thread()
            if self.worker_mode is 'live':
                self.start_thread(self.acquire_spectrum, 'acquire')

    def on_live_clicked(self, button):

        if self.worker_thread is None:
            self.start_thread(self.live_spectrum, 'live')
        else:
            self.stop_thread()
            if self.worker_mode is 'acquire':
                self.start_thread(self.live_spectrum, 'live')

    def on_stagetostart_clicked(self, button):
        self.log.stage_to_starting_point()

    def acquire_spectrum(self, e):
        while True:
            y, running = self.log.measure_spectrum()
            GLib.idle_add(self.update_plot(),y)
            self._progress_fraction = self.log.get_number_of_samples()/self.log.get_scan_index()
            GLib.idle_add(self.progress.set_fraction, self._progress_fraction)

            if not running:
                self.log.save_data()
                break

            if e.is_set:
                break
        return

    def live_spectrum(self, e):
        while not e.is_set():
            y = self.log.get_spec()
            GLib.idle_add(self.update_plot(),y)
        return

    def update_plot(self,data):
        self.line.set_ydata(data)
        self.ax.relim()
        self.ax.autoscale_view(False, False, True)
        self.canvas.draw()
        return

    def update_progress(self):
        self.progress.set_fraction(self._progress_fraction)


    def run(self):
        """	run main gtk thread """
        try:
            GLib.timeout_add(self._heartbeat, self._update_title)
            Gtk.main()
        except KeyboardInterrupt:
            pass

    def _update_title(self, _suff=cycle('/|\-')):
        self.window.set_title('%s %s' % (self._window_title, next(_suff)))
        return True

    @staticmethod
    def calc_lock_in(x, sig, ref):
        diff = np.concatenate((0, np.diff(x)))
        diff = sig * ref * diff
        return np.sum(diff)


if __name__ == "__main__":
    gui = lockin_gui()
    gui.run()
