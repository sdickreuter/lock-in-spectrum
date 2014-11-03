import time
from itertools import cycle
from datetime import datetime
import os

import pandas
import numpy as np
import PIStage
from gi.repository import Gtk
from gi.repository import GObject
from gi.repository import GLib

from spectrum import Spectrum
import dialogs
from settings import Settings

class MPL:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas

class LockinGui(object):
    _window_title = "Lock-in Spectrum"
    _heartbeat = 100  # ms delay at which the plot/gui is refreshed

    def __init__(self):
        self.savedir = "./Spectra/"
        self.path = "./"

        self.settings = Settings()

        self.stage = PIStage.Dummy()
        # self.stage = PIStage.E545();

        GObject.threads_init()
        # only GObject.idle_add() is in the background thread
        self.window = Gtk.Window(title=self._window_title)
        # self.window.set_resizable(False)
        self.window.set_border_width(3)

        self.grid = Gtk.Grid()
        self.grid.set_row_spacing(5)
        self.grid.set_column_spacing(5)
        self.window.add(self.grid)

        # Buttons for spectrum stack
        self.button_live = Gtk.Button(label="Liveview")
        self.button_live.set_tooltip_text("Start/Stop Liveview of Spectrum")
        self.button_stop = Gtk.Button(label="Stop")
        self.button_stop.set_tooltip_text("Stop any ongoing Action")
        self.button_stop.set_sensitive(False)
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
        self.label_stepsize = Gtk.Label(label=str(self.settings.stepsize))
        self.button_moverel = Gtk.Button(label="Move Stage rel.")
        self.button_moveabs = Gtk.Button(label="Move Stage abs.")
        # Stage position labels
        self.label_x = Gtk.Label()
        self.label_y = Gtk.Label()
        self.label_z = Gtk.Label()
        self.show_pos()

        # Connect Buttons
        self.window.connect("delete-event", self.quit)
        self.button_aquire.connect("clicked", self.on_lockin_clicked)
        self.button_direction.connect("clicked", self.on_direction_clicked)
        self.button_live.connect("clicked", self.on_live_clicked)
        self.button_stop.connect("clicked", self.on_stop_clicked)
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

        # Stage Control Button Table
        self.table_stagecontrol = Gtk.Table(3, 4, True)
        self.table_stagecontrol.attach(self.button_xup, 0, 1, 1, 2)
        self.table_stagecontrol.attach(self.button_xdown, 2, 3, 1, 2)
        self.table_stagecontrol.attach(self.button_yup, 1, 2, 0, 1)
        self.table_stagecontrol.attach(self.button_ydown, 1, 2, 2, 3)
        self.table_stagecontrol.attach(self.button_zup, 3, 4, 0, 1)
        self.table_stagecontrol.attach(self.button_zdown, 3, 4, 2, 3)
        # Stage Stepsize Table
        self.table_stepsize = Gtk.Table(1, 3, True)
        self.table_stepsize.attach(self.button_stepup, 0, 1, 0, 1)
        self.table_stepsize.attach(self.label_stepsize, 1, 2, 0, 1)
        self.table_stepsize.attach(self.button_stepdown, 2, 3, 0, 1)

        self.status = Gtk.Label(label="Initialized")
        self.progress = Gtk.ProgressBar()
        self._progress_fraction = 0
        self.progress.set_fraction(self._progress_fraction)

        # Box for control of taking single spectra
        self.SpectrumBox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.SpectrumBox.add(Gtk.Separator())
        self.SpectrumBox.add(self.button_live)
        self.SpectrumBox.add(Gtk.Separator())
        self.SpectrumBox.add(Gtk.Label("Lock-In Spectrum"))
        self.SpectrumBox.add(self.button_aquire)
        self.SpectrumBox.add(self.button_direction)
        self.SpectrumBox.add(Gtk.Separator())
        self.SpectrumBox.add(Gtk.Label(label="Additional Spectra"))
        self.SpectrumBox.add(self.button_dark)
        self.SpectrumBox.add(self.button_lamp)
        self.SpectrumBox.add(self.button_normal)
        self.SpectrumBox.add(Gtk.Separator())
        self.SpectrumBox.add(Gtk.Label(label="Miscellaneous"))
        self.SpectrumBox.add(self.button_save)
        self.SpectrumBox.add(self.button_settings)
        self.SpectrumBox.add(self.button_reset)
        self.SpectrumBox.add(self.button_loaddark)
        self.SpectrumBox.add(self.button_loadlamp)
        self.SpectrumBox.add(Gtk.Separator())

        # box for Stage control
        self.stage_hbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        # self.stage_hbox.add(Gtk.Separator())
        self.stage_hbox.add(self.button_search)
        self.stage_hbox.add(Gtk.Label(label="Stage Control"))
        self.stage_hbox.add(self.table_stagecontrol)
        self.stage_hbox.add(Gtk.Label(label="Set Stepsize [um]"))
        self.stage_hbox.add(self.table_stepsize)
        self.stage_hbox.add(self.button_moverel)
        self.stage_hbox.add(self.button_moveabs)
        self.labels_hbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.labels_hbox.add(self.label_x)
        self.labels_hbox.add(self.label_y)
        self.labels_hbox.add(self.label_z)


        # Buttons for scanning stack
        self.button_add_position = Gtk.Button('Add Position to List')
        self.button_spangrid = Gtk.Button('Span Grid')
        self.button_searchonoff = Gtk.Switch()
        self.label_searchonoff = Gtk.Label('Search Max.')
        self.searchonoff_box =Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.searchonoff_box.set_homogeneous(True)
        self.searchonoff_box.add(self.label_searchonoff)
        self.searchonoff_box.add(self.button_searchonoff)
        self.button_lockinonoff = Gtk.Switch()
        self.label_lockinonoff = Gtk.Label('Use Lock-In')
        self.lockinonoff_box =Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.lockinonoff_box.set_homogeneous(True)
        self.lockinonoff_box.add(self.label_lockinonoff)
        self.lockinonoff_box.add(self.button_lockinonoff)
        self.button_scan_start = Gtk.Button('Start Scan')
        self.scan_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.button_scan_add = Gtk.ToolButton(Gtk.STOCK_ADD)
        self.button_scan_remove = Gtk.ToolButton(Gtk.STOCK_REMOVE)
        self.button_scan_clear = Gtk.ToolButton(Gtk.STOCK_DELETE)
        self.scan_hbox.set_homogeneous(True)
        self.scan_hbox.add(self.button_scan_add)
        self.scan_hbox.add(self.button_scan_remove)
        self.scan_hbox.add(self.button_scan_clear)


        # Treeview for showing/settings scanning grid
        self.scan_store = Gtk.ListStore(float, float)
        self.scan_view = Gtk.TreeView(model=self.scan_store)
        self.scan_xrenderer = Gtk.CellRendererText()
        self.scan_xrenderer.set_property("editable", True)
        self.scan_xcolumn = Gtk.TreeViewColumn("x", self.scan_xrenderer, text=0)
        self.scan_xcolumn.set_min_width(100)
        self.scan_xcolumn.set_alignment(0.5)
        self.scan_view.append_column(self.scan_xcolumn)

        self.scan_yrenderer = Gtk.CellRendererText()
        self.scan_yrenderer.set_property("editable", True)
        self.scan_ycolumn = Gtk.TreeViewColumn("y", self.scan_yrenderer, text=1)
        self.scan_ycolumn.set_min_width(100)
        self.scan_ycolumn.set_alignment(0.5)
        self.scan_view.append_column(self.scan_ycolumn)

        self.scan_scroller = Gtk.ScrolledWindow()
        self.scan_scroller.set_vexpand(True)
        self.scan_scroller.add(self.scan_view)

        self.scan_store.append([10.0, 10.0])
        self.scan_store.append([11.0, 10.0])
        self.scan_store.append([10.0, 11.0])


        #Connections for scanning stack
        self.button_add_position.connect("clicked", self.on_add_position_clicked)
        self.button_spangrid.connect("clicked", self.on_spangrid_clicked)
        self.button_scan_start.connect("clicked", self.on_scan_start_clicked)
        self.button_scan_add.connect("clicked", self.on_scan_add)
        self.button_scan_remove.connect("clicked", self.on_scan_remove)
        self.button_scan_clear.connect("clicked", self.on_scan_clear)
        self.scan_xrenderer.connect("edited", self.on_scan_xedited)
        self.scan_yrenderer.connect("edited", self.on_scan_yedited)

        #Box for control of scanning
        self.ScanningBox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.ScanningBox.add(Gtk.Separator())
        self.ScanningBox.add(self.button_add_position)
        self.ScanningBox.add(self.button_spangrid)
        self.ScanningBox.add(self.searchonoff_box)
        self.ScanningBox.add(self.lockinonoff_box)
        self.ScanningBox.add(Gtk.Separator())
        self.ScanningBox.add(Gtk.Label("Scanning Positions"))
        self.ScanningBox.add(self.scan_hbox)
        self.ScanningBox.add(self.scan_scroller)
        self.ScanningBox.add(Gtk.Separator())
        self.ScanningBox.add(self.button_scan_start)


        # MPL stuff
        self.figure = MPL.Figure()
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.ax.grid(True)
        self.canvas = MPL.FigureCanvas(self.figure)
        self.canvas.set_hexpand(True)
        self.canvas.set_vexpand(True)

        self.canvas.set_size_request(800, 800)

        self.stack = Gtk.Stack()
        self.stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        self.stack.set_transition_duration(500)

        self.stack.add_titled(self.SpectrumBox, "spec", "Spectra")
        self.stack.add_titled(self.ScanningBox, "scan", "Scanning")

        self.stack_switcher = Gtk.StackSwitcher()
        self.stack_switcher.set_stack(self.stack)

        self.SideBox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.SideBox.add(self.stack_switcher)
        self.ScanningBox.add(Gtk.Separator())
        self.SideBox.add(self.button_stop)
        self.SideBox.add(self.stack)
        self.SideBox.add(self.stage_hbox)
        self.SideBox.add(self.labels_hbox)

        self.grid.add(self.canvas)
        self.grid.attach_next_to(self.SideBox, self.canvas, Gtk.PositionType.RIGHT, 1, 1)
        self.grid.attach_next_to(self.progress, self.canvas, Gtk.PositionType.BOTTOM, 1, 1)
        self.grid.attach_next_to(self.status, self.SideBox, Gtk.PositionType.BOTTOM, 1, 1)

        self.window.show_all()

        self.spectrum = Spectrum(self.stage, self.settings, self.status, self.progress, self.enable_buttons, self.disable_buttons)  # logger class which coordinates the spectrometer and the stage


        self._spec = self.spectrum.get_spec() # get an initial spectrum for display
        self._wl = self.spectrum.get_wl()  # get the wavelengths
        self.lines = []
        self.lines.extend(self.ax.plot(self._wl, self._spec, "-"))
        self.lines.extend(self.ax.plot(self._wl, self.spectrum.smooth(self._spec), "-", c="black"))  # plot initial spectrum

        #Dialogs
        self.settings_dialog = dialogs.SettingsDialog(self.window, self.settings)
        self.direction_dialog = dialogs.DirectionDialog(self.window, self.settings)
        self.moveabs_dialog = dialogs.MoveAbsDialog(self.window, self.stage)
        self.moverel_dialog = dialogs.MoveRelDialog(self.window, self.stage)
        self.spangrid_dialog = dialogs.SpanGridDialog(self.window)
        self.prefix_dialog = dialogs.PrefixDialog(self.window)


    def quit(self, *args):
        """
        Function for quitting the program, will also stop the worker thread
        :param args:
        """
        self.spectrum = None
        Gtk.main_quit(*args)

    def disable_buttons(self):
        #self.stack_switcher.set_sensitive(False)
        self.scan_hbox.set_sensitive(False)
        self.SpectrumBox.set_sensitive(False)
        self.stage_hbox.set_sensitive(False)
        self.ScanningBox.set_sensitive(False)
        self.button_stop.set_sensitive(True)


    def enable_buttons(self):
        #self.stack_switcher.set_sensitive(True)
        self.scan_hbox.set_sensitive(True)
        self.SpectrumBox.set_sensitive(True)
        self.stage_hbox.set_sensitive(True)
        self.ScanningBox.set_sensitive(True)
        self.button_stop.set_sensitive(False)

    # ##---------------- button connect functions ----------
    def on_scan_start_clicked(self, widget):
        os.chdir(self.savedir)

        prefix = self.prefix_dialog.rundialog()

        if prefix is not None:
            try:
                # os.path.exists(prefix)
                os.mkdir(prefix)
            except:
                print("Error creating directory ./"+prefix)
            path = prefix + '/'
            self.status.set_label('Scanning')
            self.spectrum.make_scan(self.scan_store, self.button_searchonoff.get_active(), self.button_lockinonoff.get_active(), path)
            self.disable_buttons()

        os.chdir('../')


    def on_add_position_clicked(self, widget):
        pos = self.stage.query_pos()
        self.scan_store.append([pos[0], pos[1]])

    def on_spangrid_clicked(self, widget):
        iter = self.scan_store.get_iter_first()
        grid = self.spangrid_dialog.rundialog()
        if (len(self.scan_store) >= 3) & (grid[0] is not 0):
            a = self.scan_store[iter][:]
            iter = self.scan_store.iter_next(iter)
            b = self.scan_store[iter][:]
            iter = self.scan_store.iter_next(iter)
            c = self.scan_store[iter][:]

            grid_vec_1 = [b[0] - a[0], b[1] - b[1]]
            grid_vec_2 = [c[0] - a[0], c[1] - b[1]]

            self.scan_store.clear()

            for x in range(int(grid[0])):
                for y in range(int(grid[1])):
                    vec_x = a[0] + grid_vec_1[0] * x + grid_vec_2[0] * y
                    vec_y = a[1] + grid_vec_1[1] * x + grid_vec_2[1] * y
                    self.scan_store.append([vec_x, vec_y])

    def on_stop_clicked(self, widget):
        self.spectrum.stop_process()
        self.enable_buttons()
        self.status.set_label('Stopped')

    def on_reset_clicked(self, widget):
        self.spectrum.reset()
        self.spectrum.dark = None
        self.spectrum.lamp = None
        self.spectrum.lockin = None
        self.spectrum.normal = None

    def on_lockin_clicked(self, widget):
        self.status.set_label('Acquiring ...')
        self.spectrum.take_lockin()
        self.disable_buttons()

    def on_direction_clicked(self, widget):
        self.direction_dialog.rundialog()

    def on_live_clicked(self, widget):
        self.status.set_label('Liveview')
        self.spectrum.take_live()
        self.disable_buttons()

    def on_search_clicked(self, widget):
        self.status.set_text("Searching Max.")
        self.spectrum.search_max()
        self.disable_buttons()

    def on_save_clicked(self, widget):
        self.status.set_label("Saving Data ...")
        self.save_data()
        self.status.set_label('Data saved')

    def on_settings_clicked(self, widget):
        self.settings_dialog.rundialog()
        self.spectrum.reset()

    def on_dark_clicked(self, widget):
        self.status.set_label('Taking Dark Spectrum')
        self.spectrum.take_dark()
        self.disable_buttons()

    def on_lamp_clicked(self, widget):
        self.status.set_label('Taking Lamp Spectrum')
        self.spectrum.take_lamp()
        self.disable_buttons()

    def on_normal_clicked(self, widget):
        self.status.set_label('Taking Normal Spectrum')
        self.spectrum.take_normal()
        self.disable_buttons()

    def on_loaddark_clicked(self, widget):
        buf = self._load_spectrum_from_file()
        if not buf is None:
            self.spectrum.dark = buf

    def on_loadlamp_clicked(self, widget):
        buf = self._load_spectrum_from_file()
        if not buf is None:
            self.spectrum.lamp = buf

    # ##---------------- END button connect functions ----------

    def _load_spectrum_from_file(self):
        dialog = Gtk.FileChooserDialog("Please choose a file", self.window,
                                       Gtk.FileChooserAction.OPEN,
                                       (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                        Gtk.STOCK_OPEN, Gtk.ResponseType.OK))

        data = None
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

    # ## ----------- scan Listview connect functions

    def on_scan_xedited(self, widget, path, number):
        self.scan_store[path][0] = float(number.replace(',', '.'))
        # self.plotpoints()

    def on_scan_yedited(self, widget, path, number):
        self.scan_store[path][1] = float(number.replace(',', '.'))
        # self.plotpoints()

    def on_scan_add(self, widget):
        self.scan_store.append()

    def on_scan_remove(self, widget):
        select = self.scan_view.get_selection()
        model, treeiter = select.get_selected()
        if treeiter is not None:
            self.scan_store.remove(treeiter)

    def on_scan_clear(self, widget):
        self.scan_store.clear()


    # ## ----------- END scan Listview connect functions


    # ##---------------- Stage Control Button Connect functions ----------

    def show_pos(self):
        pos = self.stage.last_pos()
        #print(pos)
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
        self.settings.stepsize *= 10
        if self.settings.stepsize > 10:
            self.settings.stepsize = 10.0
        self.label_stepsize.set_text(str(self.settings.stepsize))
        self.settings.save()

    def on_stepdown_clicked(self, widget):
        self.settings.stepsize /= 10
        if self.settings.stepsize < 0.001:
            self.settings.stepsize = 0.001
        self.label_stepsize.set_text(str(self.settings.stepsize))
        self.settings.save()

    def on_moverel_clicked(self, widget):
        self.moverel_dialog.rundialog()
        self.show_pos()

    def on_moveabs_clicked(self, widget):
        self.moveabs_dialog.rundialog()
        self.show_pos()


    # ##---------------- END Stage Control Button Connect functions ------

    def run(self):
        """	run main gtk thread """
        try:
            #GLib.timeout_add(self._heartbeat, self._update_plot)
            GLib.idle_add(self._update_plot)
            #GLib.io_add_watch(self.spectrum.conn_for_main, GLib.IO_IN | GLib.IO_PRI, self._update, args=(self,))
            #GLib.io_add_watch(self.spectrum.conn_for_main, GLib.IO_IN | GLib.IO_PRI, self.spectrum.callback, args=(self.spectrum,self.progress,))
            GLib.io_add_watch(self.spectrum.conn_for_main, GLib.IO_IN | GLib.IO_PRI, self.spectrum.callback, args=(self.spectrum,))
            Gtk.main()
        except KeyboardInterrupt:
            pass

    def _update_plot(self):
        self.lines[0].set_ydata(self.spectrum.get_spec())
        self.lines[1].set_ydata(self.spectrum.smooth(self.spectrum.get_spec()))
        self.ax.relim()
        self.ax.autoscale_view(False, False, True)
        self.canvas.draw()
        self.show_pos()
        return True

    def save_data(self):
        os.chdir(self.savedir)
        prefix = self.prefix_dialog.rundialog()
        if prefix is not None:
            try:
                # os.path.exists(prefix)
                os.mkdir(prefix)
            except:
                print("Error creating directory ./"+prefix)
            path = prefix + '/'
            self.spectrum.save_data(path)
        os.chdir('../')

if __name__ == "__main__":
    gui = LockinGui()
    gui.run()
