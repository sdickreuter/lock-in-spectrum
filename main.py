

import pandas
import time
from logger import logger
import threading
from itertools import cycle
import numpy as np
from datetime import datetime
import os
import PIStage
import math

from gi.repository import Gtk
from gi.repository import GObject
from gi.repository import GLib

import dialogs
from settings import Settings

class mpl:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas

from scipy.interpolate import interp1d

class lockin_gui(object):
    _window_title = "Lock-in Spectrum"
    _heartbeat = 250  # ms delay at which the plot/gui is refreshed

    def __init__(self):
        self.settings = Settings()

        #self.stage = PIStage.Dummy();
        self.stage = PIStage.E545();

        GObject.threads_init()  # all Gtk is in the main thread;
        # only GObject.idle_add() is in the background thread
        self.window = Gtk.Window(title=self._window_title)
        self.window.set_resizable(False)
        self.window.set_border_width(3)

        self.grid = Gtk.Grid()
        self.grid.set_row_spacing(5)
        self.grid.set_column_spacing(5)
        self.window.add(self.grid)

        # Buttons
        self.button_live = Gtk.Button(label="Liveview")
        self.button_live.set_tooltip_text("Start/Stop Liveview of Spectrum")
        self.button_aquire = Gtk.Button(label="Aquire Spectrum")
        self.button_aquire.set_tooltip_text("Start/Stop aquiring Lock-In Spectrum")
        self.button_direction = Gtk.Button(label="Set Direction")
        self.button_direction.set_tooltip_text("Set Direction of Stage Movement")
        self.button_settings = Gtk.Button(label="Settings")
        self.button_settings.set_tooltip_text("Set Integration Time and Number of Samples")
        self.button_search = Gtk.Button(label="Search for Max")
        self.button_search.set_tooltip_text("Search for position with maximum Intensity")
        self.button_save = Gtk.Button(label="Save Data")
        self.button_save.set_tooltip_text("Save all spectral Data in .csv")
        self.button_dark = Gtk.Button(label="Take Dark Spectrum")
        self.button_dark.set_tooltip_text("Take dark spectrum which will substracted from spectrum")
        self.button_lamp = Gtk.Button(label="Take Lamp Spectrum")
        self.button_lamp.set_tooltip_text("Take lamp spectrum to normalize spectrum")
        self.button_normal = Gtk.Button(label="Take Normal Spectrum")
        self.button_normal.set_tooltip_text("Start/Stop taking a normal spectrum as comparison to the Lock-In spectrum")
        self.button_reset = Gtk.Button(label="Reset")
        self.button_reset.set_tooltip_text("Reset all spectral data (if not saved data is lost!)")
        self.button_loaddark = Gtk.Button(label="Loard Dark Spectrum")
        self.button_loaddark.set_tooltip_text("Load Dark Spectrum from file")
        self.button_loadlamp = Gtk.Button(label="Loard Lamp Spectrum")
        self.button_loadlamp.set_tooltip_text("Load Lamp Spectrum from file")
        # Stage Control Buttons
        self.button_xup = Gtk.Button(label="x+")
        self.button_xdown = Gtk.Button(label="x-")
        self.button_yup = Gtk.Button(label="y+")
        self.button_ydown = Gtk.Button(label="y-")
        self.button_zup = Gtk.Button(label="z+")
        self.button_zdown = Gtk.Button(label="z-")
        self.button_stepup = Gtk.Button(label="+")
        self.button_stepdown = Gtk.Button(label="-")
        self.label_stepsize = Gtk.Label(label = str(self.settings.stepsize))
        self.button_moverel = Gtk.Button(label="Move Stage rel.")
        self.button_moveabs = Gtk.Button(label="Move Stage abs.")
        # Stage position labels
        self.label_x = Gtk.Label()
        self.label_y = Gtk.Label()
        self.label_z = Gtk.Label()
        self.show_pos()

        # Connect Buttons
        self.window.connect("delete-event", self.quit)
        self.button_aquire.connect("clicked", self.on_aquire_clicked)
        self.button_direction.connect("clicked", self.on_direction_clicked)
        self.button_live.connect("clicked", self.on_live_clicked)
        self.button_settings.connect("clicked", self.on_settings_clicked)
        self.button_search.connect("clicked", self.on_search_clicked)
        self.button_save.connect("clicked", self.on_save_clicked)
        self.button_dark.connect("clicked", self.on_dark_clicked)
        self.button_lamp.connect("clicked", self.on_lamp_clicked)
        self.button_normal.connect("clicked", self.on_normal_clicked)
        self.button_reset.connect("clicked", self.on_reset_clicked)
        self.button_loaddark.connect("clicked", self.on_loaddark_clicked)
        self.button_loadlamp.connect("clicked", self.on_loadlamp_clicked)
        # Connect Stage Control Buttons
        self.button_xup.connect("clicked", self.on_xup_clicked)
        self.button_xdown.connect("clicked", self.on_xdown_clicked)
        self.button_yup.connect("clicked", self.on_yup_clicked)
        self.button_ydown.connect("clicked", self.on_ydown_clicked)
        self.button_zup.connect("clicked", self.on_zup_clicked)
        self.button_zdown.connect("clicked", self.on_zdown_clicked)
        self.button_stepup.connect("clicked", self.on_stepup_clicked)
        self.button_stepdown.connect("clicked", self.on_stepdown_clicked)
        self.button_moverel.connect("clicked", self.on_moverel_clicked)
        self.button_moveabs.connect("clicked", self.on_moveabs_clicked)

        #Stage Control Button Table
        self.table_stagecontrol = Gtk.Table(3, 4, True)
        self.table_stagecontrol.attach(self.button_xup, 0, 1, 1, 2)
        self.table_stagecontrol.attach(self.button_xdown, 2, 3, 1, 2)
        self.table_stagecontrol.attach(self.button_yup, 1, 2, 0, 1)
        self.table_stagecontrol.attach(self.button_ydown, 1, 2, 2, 3)
        self.table_stagecontrol.attach(self.button_zup, 3, 4, 0, 1)
        self.table_stagecontrol.attach(self.button_zdown, 3, 4, 2, 3)
        #Stage Stepsize Table
        self.table_stepsize = Gtk.Table(1,3, True)
        self.table_stepsize.attach(self.button_stepup, 0, 1, 0, 1)
        self.table_stepsize.attach(self.label_stepsize, 1, 2, 0, 1)
        self.table_stepsize.attach(self.button_stepdown, 2, 3, 0, 1)

        self.status = Gtk.Label(label="Initialized")
        self.progress = Gtk.ProgressBar()
        self._progress_fraction = 0
        self.progress.set_fraction(self._progress_fraction)

        self.sidebox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        self.sidebox.add(Gtk.Separator())
        self.sidebox.add(self.button_live)
        self.sidebox.add(Gtk.Separator())
        self.sidebox.add(Gtk.Label("Lock-In Spectrum"))
        self.sidebox.add(self.button_aquire)
        self.sidebox.add(self.button_direction)
        self.sidebox.add(Gtk.Separator())
        self.sidebox.add(Gtk.Label(label="Additional Spectra"))
        self.sidebox.add(self.button_dark)
        self.sidebox.add(self.button_lamp)
        self.sidebox.add(self.button_normal)
        self.sidebox.add(Gtk.Separator())
        self.sidebox.add(Gtk.Label(label="Miscellaneous"))
        self.sidebox.add(self.button_search)
        self.sidebox.add(self.button_save)
        self.sidebox.add(self.button_settings)
        self.sidebox.add(self.button_reset)
        self.sidebox.add(self.button_loaddark)
        self.sidebox.add(self.button_loadlamp)
        self.sidebox.add(Gtk.Separator())
        self.sidebox.add(Gtk.Label(label="Stage Control"))
        self.sidebox.add(self.table_stagecontrol)
        self.sidebox.add(Gtk.Label(label="Set Stepsize [um]"))
        self.sidebox.add(self.table_stepsize)
        self.sidebox.add(self.button_moverel)
        self.sidebox.add(self.button_moveabs)
        self.sidebox.add(self.label_x)
        self.sidebox.add(self.label_y)
        self.sidebox.add(self.label_z)

        # MPL stuff
        self.figure = mpl.Figure()
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.ax.grid(True)
        #self.ax.set_color_cycle(["b", ""])
        self.canvas = mpl.FigureCanvas(self.figure)
        # self.line, = self.ax.plot(self.wl, self.sp[:,0])

        self.canvas.set_size_request(900, 800)
        self.sidebox.set_size_request(100, -1)
        self.progress.set_size_request(-1, 15)
        self.status.set_size_request(100, -1)

        self.grid.add(self.canvas)
        self.grid.attach_next_to(self.sidebox, self.canvas, Gtk.PositionType.RIGHT, 1, 1)
        self.grid.attach_next_to(self.progress, self.canvas, Gtk.PositionType.BOTTOM, 1, 1)
        self.grid.attach_next_to(self.status, self.sidebox, Gtk.PositionType.BOTTOM, 1, 1)

        self.window.show_all()

        # Thread used for taking spectra
        self.worker_running_event = threading.Event()
        self.worker_thread = None
        self.worker_mode = None
        self.worker_lock = threading.Lock()  # to signal the thread to stop

        self.log = logger(self.stage, self.settings)  # logger class which coordinates the spectrometer and the stage
        self._spec = self.log.get_spec()  # get an initial spectrum for display
        self._wl = self.log.get_wl()  # get the wavelengths
        self.lines = []
        self.lines.extend( self.ax.plot(self._wl, self._spec,"-") )
        self.lines.extend( self.ax.plot(self._wl, self.smooth(self._spec),"-",c="black") )  # plot initial spectrum

        #Dialogs
        self.settings_dialog = dialogs.Settings_Dialog(self.window, self.settings)
        self.direction_dialog = dialogs.Direction_Dialog(self.window, self.settings)
        self.moveabs_dialog = dialogs.MoveAbs_Dialog(self.window, self.stage)
        self.moverel_dialog = dialogs.MoveRel_Dialog(self.window, self.stage)

        # variables for storing the spectra
        self.lamp = None
        self.dark = None
        self.normal = None
        self.lockin = None

    def smooth(self, x):
        """
        modified from: http://wiki.scipy.org/Cookbook/SignalSmooth
        """
        window_len=151

        s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

        window='hanning'
        #window='flat'

        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y = np.convolve(w/w.sum(),s,mode='valid')
        y = y[(window_len/2):-(window_len/2)]
        return y

    def quit(self, *args):
        """
        Function for quitting the program, will also stop the worker thread
        :param args:
        """
        if not self.worker_thread is None:
            self.stop_thread()
        self.log = None
        Gtk.main_quit(*args)

    def start_thread(self, target, mode):
        """
        Starts the working thread which takes spectra
        :param target: function the thread shall execute
        :param mode: which kind of spectrum the thread is taking (dark, lamp, lock-in ...)
        """
        self.log.set_integration_time(self.settings.integration_time)
        self.worker_mode = mode
        self.worker_running_event.clear()
        if not self.worker_thread is None: self.worker_thread.join(0.2)  # wait 200ms for thread to finish
        self.worker_thread = threading.Thread(target=target, args=(self.worker_running_event,))
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def stop_thread(self):
        self.worker_running_event.set()
        self.worker_thread.join(1)  # wait 1 s for thread to finish
        self.worker_thread = None

    def on_integration_time_change(self, widget):
        self.log.set_integration_time(float(self.integration_time_spin.get_value_as_int()) / 1000)
        time.sleep(0.1)

    def on_number_of_samples_change(self, widget):
        self.log.set_number_of_samples(self.number_of_samples_spin.get_value_as_int())

###---------------- button connect functions ----------

    def on_reset_clicked(self, widget):
        self.log.reset()
        self.dark = None
        self.lamp = None
        self.lockin = None
        self.normal = None
        self.settings_dialog.enable_number_of_samples()


    def on_aquire_clicked(self, widget):
        if self.worker_thread is None:
            self.log.reset()
            self.settings_dialog.enable_number_of_samples()
            self.button_direction.set_sensitive(False)
            self.status.set_label('Acquiring ...')
            self.start_thread(self.acquire_spectrum, 'acquire')
        else:
            self.status.set_label('Stopped')
            self.button_direction.set_sensitive(True)
            self.stop_thread()
            if not self.worker_mode is 'acquire':
                self.log.reset()
                self.settings_dialog.enable_number_of_samples()
                self.button_direction.set_sensitive(False)
                self.status.set_label('Acquiring ...')
                self.start_thread(self.acquire_spectrum, 'acquire')

    def on_direction_clicked(self, widget):
        self.direction_dialog.rundialog()

    def on_live_clicked(self, widget):
        if self.worker_thread is None:
            self.status.set_label('Liveview')
            self.start_thread(self.live_spectrum, 'live')
        else:
            self.status.set_label('Stopped')
            self.stop_thread()
            if not self.worker_mode is 'live':
                self.status.set_label('Liveview')
                self.start_thread(self.live_spectrum, 'live')

    def on_search_clicked(self, widget):
        self.status.set_text("Searching Max.")
        self.search_max_int();
        self.status.set_text("Max. approached")


    def on_save_clicked(self, widget):
        if self.log._new_spectrum:
            self.status.set_label("Saving Data ...")
            self.save_data()
            self.log.reset()
            self.settings_dialog.enable_number_of_samples()
            self.status.set_label('Data saved')
        else:
            self.status.set_label('No Data found')

    def on_settings_clicked(self, widget):
        self.settings_dialog.rundialog()

    def on_dark_clicked(self, widget):
        if self.worker_thread is None:
            self.status.set_label('Taking Dark Spectrum')
            self.start_thread(self.take_spectrum, 'dark')
        else:
            self.status.set_label('Stopped')
            self.stop_thread()
            if not self.worker_mode is 'dark':
                self.status.set_label('Taking Dark Spectrum')
                self.start_thread(self.take_spectrum, 'dark')

    def on_lamp_clicked(self, widget):
        if self.worker_thread is None:
            self.status.set_label('Taking Lamp Spectrum')
            self.start_thread(self.take_spectrum, 'lamp')
        else:
            self.status.set_label('Stopped')
            self.stop_thread()
            if not self.worker_mode is 'lamp':
                self.status.set_label('Taking Lamp Spectrum')
                self.start_thread(self.take_spectrum, 'lamp')

    def on_normal_clicked(self, widget):
        if self.worker_thread is None:
            self.status.set_label('Taking Normal Spectrum')
            self.start_thread(self.take_spectrum, 'normal')
        else:
            self.status.set_label('Stopped')
            self.stop_thread()
            if not self.worker_mode is 'normal':
                self.status.set_label('Taking Normal Spectrum')
                self.start_thread(self.take_spectrum, 'normal')

    def on_loaddark_clicked(self, widget):
        buf = self._load_spectrum_from_file()
        if not buf is None: self.dark = buf

    def on_loadlamp_clicked(self, widget):
        buf = self._load_spectrum_from_file()
        if not buf is None: self.lamp = buf

###---------------- END button connect functions ----------

    def _load_spectrum_from_file(self):
        dialog = Gtk.FileChooserDialog("Please choose a file", self.window,
            Gtk.FileChooserAction.OPEN,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
             Gtk.STOCK_OPEN, Gtk.ResponseType.OK))

        filter_text = Gtk.FileFilter()
        filter_text.set_name("CSV Spectrum files")
        filter_text.add_pattern("*.csv")
        dialog.add_filter(filter_text)
        dialog.set_current_folder(os.path.dirname(os.path.abspath(__file__)))
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
           data = pandas.DataFrame(pandas.read_csv(dialog.get_filename()))
           data = data['intensity']
        elif response == Gtk.ResponseType.CANCEL:
           data = None
        dialog.destroy()
        return data


###---------------- Stage Control Button Connect functions ----------

    def show_pos(self):
        pos = self.stage.pos()
        self.label_x.set_text("x: {0:+8.4f}".format(pos[0]))
        self.label_y.set_text("y: {0:+8.4f}".format(pos[1]))
        self.label_z.set_text("z: {0:+8.4f}".format(pos[2]))

    def on_xup_clicked(self, widget):
        self.stage.moverel(dx=self.settings.stepsize)
        self.show_pos()

    def on_xdown_clicked(self, widget):
        self.stage.moverel(dx=-self.settings.stepsize)
        self.show_pos()

    def on_yup_clicked(self, widget):
        self.stage.moverel(dy=self.settings.stepsize)
        self.show_pos()

    def on_ydown_clicked(self, widget):
        self.stage.moverel(dy=-self.settings.stepsize)
        self.show_pos()

    def on_zup_clicked(self, widget):
        self.stage.moverel(dz=self.settings.stepsize)
        self.show_pos()

    def on_zdown_clicked(self, widget):
        self.stage.moverel(dz=-self.settings.stepsize)
        self.show_pos()

    def on_stepup_clicked(self, widget):
        self.settings.stepsize = 10*self.settings.stepsize
        if self.settings.stepsize > 10: self.settings.stepsize = 10.0
        self.label_stepsize.set_text(str(self.settings.stepsize))
        self.settings.save()

    def on_stepdown_clicked(self, widget):
        self.settings.stepsize = self.settings.stepsize/10
        if self.settings.stepsize < 0.001: self.settings.stepsize = 0.001
        self.label_stepsize.set_text(str(self.settings.stepsize))
        self.settings.save()

    def on_moverel_clicked(self, widget):
        self.moverel_dialog.rundialog()
        self.show_pos()

    def on_moveabs_clicked(self, widget):
        self.moveabs_dialog.rundialog()
        self.show_pos()


###---------------- END Stage Control Button Connect functions ------

###---------------- functions for taking and showing Spectra ----------

    def take_spectrum(self, e):
        self.settings_dialog.disable_number_of_samples()
        data = np.zeros(1024, dtype=np.float64)
        for i in range(self.settings.number_of_samples):
            data = (data + self.log.get_spec()) / 2
            self._progress_fraction = float(i + 1) / self.settings.number_of_samples

            if e.is_set():
                if self.worker_mode is 'dark':
                    self.dark = None
                if self.worker_mode is 'lamp':
                    self.lamp = None
                if self.worker_mode is 'normal':
                    self.normal = None
                break

        self.settings_dialog.enable_number_of_samples()

        if self.worker_mode is 'dark':
            self.settings_dialog.disable_number_of_samples()
            self.dark = data
        if self.worker_mode is 'lamp':
            self.settings_dialog.disable_number_of_samples()
            self.lamp = data
        if self.worker_mode is 'normal':
            self.normal = data

        self._spec = data
        self.status.set_label('Spectra taken')
        return True

    def acquire_spectrum(self, e):
        # self._plotting = False
        self.button_direction.set_sensitive(False)
        self.settings_dialog.disable_number_of_samples()
        self.lockin = None
        while True:
            self._spec, running = self.log.measure_spectrum()

            self._progress_fraction = float(self.log.get_scan_index()) / self.settings.number_of_samples

            if not self.dark is None:
                self._spec = self._spec - self.dark
                if not self.lamp is None:
                    self._spec = self._spec / self.lamp

            if not running:
                self._spec = self.calc_lockin()
                self.lockin = self._spec
                self.status.set_label('Spectra acquired')
                break

            if e.is_set():
                self.log.reset()
                self.settings_dialog.enable_number_of_samples()
                break

        self.settings_dialog.enable_number_of_samples()
        self.button_direction.set_sensitive(True)
        return True

    def live_spectrum(self, e):
        while not e.is_set():
            self._spec = self.log.get_spec()
            if not self.dark is None:
                self._spec = self._spec - self.dark
                if not self.lamp is None:
                    self._spec = self._spec / (self.lamp)
        return True

    def update_plot(self):
        self.lines[0].set_ydata(self._spec)
        self.lines[1].set_ydata(self.smooth(self._spec))
        self.ax.relim()
        self.ax.autoscale_view(False, False, True)
        self.canvas.draw()
        return True

    def update_progress(self):
        self.progress.set_fraction(self._progress_fraction)
        return True

###---------------- END functions for taking and showing Spectra ----------

    def run(self):
        """	run main gtk thread """
        try:
            GLib.timeout_add(self._heartbeat, self._update)
            Gtk.main()
        except KeyboardInterrupt:
            pass

    def _update(self, _suff=cycle('/|\-')):
        # self.window.set_title('%s %s' % (self._window_title, next(_suff)))
        self.update_plot()
        self.progress.set_fraction(self._progress_fraction)
        return True

    def calc_lockin(self):
        shape = self.log.data.shape
        res = np.empty(1024)
        diff = np.diff(np.append(0, self.log.data[:, 0]))
        ref = self.log.data[:, 1]
        for i in range(2, 1026):
            buf = self.log.data[:, i]
            if not self.dark is None:
                buf = buf - self.dark[i - 2]
                if not self.lamp is None:
                    buf = buf / (self.lamp[i - 2])
            buf = buf * diff * ref
            buf = np.sum(buf)
            res[i - 2] = buf
        return res

    def _gen_filename(self):
        return str(datetime.now().year) + str(datetime.now().month).zfill(2) \
               + str(datetime.now().day).zfill(2) + '_' + str(datetime.now().hour).zfill(2) + \
               str(datetime.now().minute).zfill(2) + str(datetime.now().second).zfill(2) + '.csv'

    def save_data(self):
        filename = self._gen_filename()
        cols = ('t', 'ref') + tuple(map(str, np.round(self._wl, 1)))
        data = pandas.DataFrame(self.log.data, columns=cols)
        data.to_csv('spectrum_' + filename, header=True, index=False)
        if not self.dark is None:
            data = np.append(np.round(self._wl, 1).reshape(self._wl.shape[0], 1),
                             self.dark.reshape(self.dark.shape[0], 1), 1)
            data = pandas.DataFrame(data, columns=('wavelength', 'intensity'))
            data.to_csv('dark_' + filename, header=True, index=False)
        if not self.lamp is None:
            data = np.append(np.round(self._wl, 1).reshape(self._wl.shape[0], 1),
                             self.lamp.reshape(self.lamp.shape[0], 1), 1)
            data = pandas.DataFrame(data, columns=('wavelength', 'intensity'))
            data.to_csv('lamp_' + filename, header=True, index=False)
        if not self.normal is None:
            data = np.append(np.round(self._wl, 1).reshape(self._wl.shape[0], 1),
                             self.normal.reshape(self.normal.shape[0], 1), 1)
            data = pandas.DataFrame(data, columns=('wavelength', 'intensity'))
            data.to_csv('normal_' + filename, header=True, index=False)
        if not self.normal is None:
            data = np.append(np.round(self._wl, 1).reshape(self._wl.shape[0], 1),
                             self.normal.reshape(self.normal.shape[0], 1), 1)
            data = pandas.DataFrame(data, columns=('wavelength', 'intensity'))
            data.to_csv('normal_' + filename, header=True, index=False)
        if not self.lockin is None:
            data = np.append(np.round(self._wl, 1).reshape(self._wl.shape[0], 1),
                             self.lockin.reshape(self.lockin.shape[0], 1), 1)
            data = pandas.DataFrame(data, columns=('wavelength', 'intensity'))
            data.to_csv('lockin_' + filename, header=True, index=False)

    def _get_hexagon(self, a):
        a = float(a)
        return ( (a,0), (0.5*a,math.sqrt(3)/2*a), (-0.5*a,math.sqrt(3)/2*a), (-a,0), (0.5*a,-math.sqrt(3)/2*a), (-0.5*a,-math.sqrt(3)/2*a) )

    def search_max_int(self):

        if self.worker_thread is not None:
            self.status.set_label('Stopped')
            self.stop_thread()

        sizes = (.1,.05,0.01,0.005)

        for size in sizes:
            hex = self._get_hexagon(size)

            origin = self.stage.pos()

            dirx = 0
            diry = 0
            for pos in hex:
                self.stage.moveabs(origin[0]+pos[0],origin[1]+pos[1])
                int = np.max(self.smooth(self.log.get_spec()))
                dirx += pos[0]*int
                diry += pos[1]*int
                #print (dirx, diry)

            norm = math.sqrt( dirx*dirx + diry*diry)
            dirx = size*dirx/norm
            diry = size*diry/norm
            #print (dirx,diry)
            self.stage.moveabs(origin[0]+dirx,origin[1]+diry,origin[2])

        self._spec = self.log.get_spec()


if __name__ == "__main__":
    gui = lockin_gui()
    gui.run()
