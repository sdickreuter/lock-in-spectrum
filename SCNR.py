__author__ = 'sei'

import random
import os
import matplotlib
matplotlib.use("Qt5Agg")
import pandas
import PIStage
from pygamepad import Gamepad
from spectrum import Spectrum
from settings import Settings
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot, QTimer, QSocketNotifier
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import dialogs
Ui_MainWindow = uic.loadUiType("ui/SCNR_main.ui")[0]
#from SCNR_main import Ui_MainWindow


class SCNR(QMainWindow):
    _window_title = "SCNR"
    _heartbeat = 100  # ms delay at which the plot/gui is refreshed, and the gamepad moves the stage

    def __init__(self, parent=None):
        super(SCNR, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.fig = Figure()
        self.axes = self.fig.add_subplot(111)
        self.axes.hold(False)

        self.Canvas = FigureCanvas(self.fig)
        self.Canvas.setParent(self.ui.plotwidget)

        self.Canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Canvas.updateGeometry()

        l = QVBoxLayout(self.ui.plotwidget)
        l.addWidget(self.Canvas)

        self.ui.status.setText("Ready")

        self.savedir = "./Spectra/"
        self.path = "./"

        self.settings = Settings()

        self.settings_dialog = dialogs.Settings_Dialog(self.settings)

        self.x_step = .0
        self.y_step = .0
        self.step_distance = 1  # in um
        self.pad = None
        try:
            pass
            #self.pad = Gamepad(True)
        except:
            print("Could not initialize Gamepad")

        try:
            pass
            #self.stage = PIStage.E545(self.settings.stage_ip,self.settings.stage_port)
        except:
            self.stage = None
            self.stage = PIStage.Dummy()
            print("Could not initialize PIStage, using Dummy instead")

        self.stage = PIStage.Dummy()
        self.spectrum = Spectrum(self.stage, self.settings, self.ui.status, self.ui.progressBar, self.enable_buttons,
                                 self.disable_buttons)  # logger class which coordinates the spectrometer and the stage

        spec = self.spectrum.get_spec()  # get an initial spectrum for display
        self._wl = self.spectrum.get_wl()  # get the wavelengths
        self.update_plot(None)
        self.spectrum.getspecthread.dynamicSpecSignal.connect(self.update_plot)

        #timer = QTimer(self)
        #timer.timeout.connect(self.update_plot)
        #timer.start(50)


    def disable_buttons(self):
        self.ui.tabWidget.setDisabled(True)
        self.ui.stage_frame.setDisabled(True)
        self.ui.Button_searchmax.setDisabled(True)
        self.ui.Button_stepup.setDisabled(True)
        self.ui.Button_stepdown.setDisabled(True)
        self.ui.Button_stop.setDisabled(False)

    def enable_buttons(self):
        self.ui.tabWidget.setDisabled(False)
        self.ui.stage_frame.setDisabled(False)
        self.ui.Button_searchmax.setDisabled(False)
        self.ui.Button_stepup.setDisabled(False)
        self.ui.Button_stepdown.setDisabled(False)
        self.ui.Button_stop.setDisabled(True)


    def _on_pad_change(self, io, condition):
        a, b, x, y, ax, ay = self.pad.receiver.recv()
        if a:
            self.on_stepdown_clicked(None)
        if b:
            self.on_add_position_clicked(None)
        if x:
            self.on_search_clicked(None)
        if y:
            self.on_stepup_clicked(None)

        self.x_step = float((ax - 128))
        if abs(self.x_step) > 8:
            self.x_step = self.x_step / 128 * self.settings.stepsize
        else:
            self.x_step = 0.0

        self.y_step = float((ay - 128))
        if abs(self.y_step) > 8:
            self.y_step = self.y_step / 128 * self.settings.stepsize
        else:
            self.y_step = 0.0
        # print('x_step: {0:3.2f} um   y_step: {1:3.2f} um'.format( self.x_step, self.y_step))
        return True

    def _pad_make_step(self):
        if self.pad is not None:
            if abs(self.x_step) > 0.0001:
                if abs(self.y_step) > 0.0001:
                    self.stage.moverel(dx=self.x_step, dy=self.y_step)
                else:
                    self.stage.moverel(dx=self.x_step)
            elif abs(self.y_step) > 0.001:
                self.stage.moverel(dy=self.y_step)
        return True

    # ##---------------- button connect functions ----------

    @pyqtSlot()
    def on_scan_start_clicked(self):
        prefix = self.prefix_dialog.rundialog()

        if prefix is not None:
            try:
                # os.path.exists(prefix)
                os.mkdir(self.savedir+prefix)
            except:
                print("Error creating directory ./" + prefix)
            path = self.savedir + prefix + '/'
            self.status.setText('Scanning')
            self.spectrum.make_scan(self.scan_store, path, self.button_searchonoff.get_active(),
                                    self.button_lockinonoff.get_active())
            self.disable_buttons()

    @pyqtSlot()
    def on_addpos_clicked(self):
        self.stage.query_pos()
        pos = self.stage.last_pos()
        self.scan_store.append([pos[0], pos[1]])

    @pyqtSlot()
    def on_spangrid_clicked(self):
        x, y, ok = dialogs.SpanGrid_Dialog.getXY()
        print((x,y,ok))
        # iterator = self.scan_store.get_iter_first()
        # grid = self.spangrid_dialog.rundialog()
        # if (len(self.scan_store) >= 3) & ((grid[0] is not 0) | (grid[1] is not 0)):
        #     a = self.scan_store[iterator][:]
        #     iterator = self.scan_store.iter_next(iterator)
        #     b = self.scan_store[iterator][:]
        #     iterator = self.scan_store.iter_next(iterator)
        #     c = self.scan_store[iterator][:]
        #
        #     if abs(b[0]) > abs(c[0]):
        #         grid_vec_1 = [b[0] - a[0], b[1] - a[1]]
        #         grid_vec_2 = [c[0] - a[0], c[1] - a[1]]
        #     else:
        #         grid_vec_2 = [b[0] - a[0], b[1] - a[1]]
        #         grid_vec_1 = [c[0] - a[0], c[1] - a[1]]
        #
        #     self.scan_store.clear()
        #
        #     for x in range(int(grid[0])):
        #         for y in range(int(grid[1])):
        #             vec_x = a[0] + grid_vec_1[0] * x + grid_vec_2[0] * y
        #             vec_y = a[1] + grid_vec_1[1] * x + grid_vec_2[1] * y
        #             self.scan_store.append([vec_x, vec_y])

    @pyqtSlot()
    def on_searchgrid_clicked(self):
        pass

    @pyqtSlot()
    def on_stop_clicked(self):
        self.ui.status.setText('Stopped')
        self.spectrum.stop_process()
        self.enable_buttons()

    @pyqtSlot()
    def on_reset_clicked(self):
        self.spectrum.reset()
        self.spectrum.dark = None
        self.spectrum.lamp = None
        self.spectrum.lockin = None
        self.spectrum.mean = None

    @pyqtSlot()
    def on_lockin_clicked(self):
        self.ui.status.setText('Acquiring ...')
        self.spectrum.take_lockin()
        self.disable_buttons()

    @pyqtSlot()
    def on_direction_clicked(self):
        self.direction_dialog.rundialog()

    @pyqtSlot()
    def on_live_clicked(self):
        self.ui.status.setText('Liveview')
        self.spectrum.take_live()
        self.disable_buttons()


    @pyqtSlot()
    def on_search_clicked(self):
        self.ui.status.setText("Searching Max.")
        self.spectrum.search_max()
        self.disable_buttons()

    @pyqtSlot()
    def on_save_clicked(self):
        self.ui.status.setText("Saving Data ...")
        self.save_data()

    @pyqtSlot()
    def on_settings_clicked(self):
        self.settings_dialog.show()
        self.axes.set_xlim([self.settings.min_wl, self.settings.max_wl])
        #self.spectrum.reset()

    @pyqtSlot()
    def on_dark_clicked(self):
        self.ui.status.setText('Taking Dark Spectrum')
        self.spectrum.take_dark()
        self.disable_buttons()

    @pyqtSlot()
    def on_lamp_clicked(self):
        self.ui.status.setText('Taking Lamp Spectrum')
        self.spectrum.take_lamp()
        self.disable_buttons()

    @pyqtSlot()
    def on_mean_clicked(self):
        self.ui.status.setText('Taking Normal Spectrum')
        self.spectrum.take_mean()
        self.disable_buttons()

    @pyqtSlot()
    def on_bg_clicked(self):
        self.ui.status.setText('Taking Background Spectrum')
        self.spectrum.take_bg()
        self.disable_buttons()

    @pyqtSlot()
    def on_series_clicked(self):
        self.ui.status.setText('Taking Time Series')
        prefix = self.prefix_dialog.rundialog()
        if prefix is not None:
            try:
                # os.path.exists(prefix)
                os.mkdir(self.savedir+prefix)
            except:
                print("Error creating directory ./" + prefix)
            path = self.savedir + prefix + '/'
        else:
            self.ui.status.setText("Error")
        self.spectrum.take_series(path)
        self.disable_buttons()

    @pyqtSlot()
    def on_loaddark_clicked(self):
        buf = self._load_spectrum_from_file()
        if not buf is None:
            self.spectrum.dark = buf

    @pyqtSlot()
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

    def on_scan_xedited(self, path, number):
        self.scan_store[path][0] = float(number.replace(',', '.'))
        # self.plotpoints()

    def on_scan_yedited(self, path, number):
        self.scan_store[path][1] = float(number.replace(',', '.'))
        # self.plotpoints()

    def on_scan_add(self):
        self.scan_store.append()

    def on_scan_remove(self):
        select = self.scan_view.get_selection()
        model, treeiter = select.get_selected()
        if treeiter is not None:
            self.scan_store.remove(treeiter)

    def on_scan_clear(self):
        self.scan_store.clear()


    # ## ----------- END scan Listview connect functions


    # ##---------------- Stage Control Button Connect functions ----------

    def show_pos(self):
        pos = self.stage.last_pos()
        # print(pos)
        self.ui.label_x.setText("x: {0:+8.4f}".format(pos[0]))
        self.ui.label_y.setText("y: {0:+8.4f}".format(pos[1]))
        self.ui.label_z.setText("z: {0:+8.4f}".format(pos[2]))

    @pyqtSlot()
    def on_xup_clicked(self):
        self.stage.moverel(dx=self.settings.stepsize)
        self.show_pos()

    @pyqtSlot()
    def on_xdown_clicked(self):
        self.stage.moverel(dx=-self.settings.stepsize)
        self.show_pos()

    @pyqtSlot()
    def on_yup_clicked(self):
        self.stage.moverel(dy=self.settings.stepsize)
        self.show_pos()

    @pyqtSlot()
    def on_ydown_clicked(self):
        self.stage.moverel(dy=-self.settings.stepsize)
        self.show_pos()

    @pyqtSlot()
    def on_zup_clicked(self):
        self.stage.moverel(dz=self.settings.stepsize)
        self.show_pos()

    @pyqtSlot()
    def on_zdown_clicked(self):
        self.stage.moverel(dz=-self.settings.stepsize)
        self.show_pos()

    @pyqtSlot()
    def on_stepup_clicked(self):
        self.settings.stepsize *= 10
        if self.settings.stepsize > 10:
            self.settings.stepsize = 10.0
        self.ui.label_stepsize.setText(str(self.settings.stepsize))
        self.settings.save()

    @pyqtSlot()
    def on_stepdown_clicked(self):
        self.settings.stepsize /= 10
        if self.settings.stepsize < 0.001:
            self.settings.stepsize = 0.001
        self.ui.label_stepsize.setText(str(self.settings.stepsize))
        self.settings.save()

    def on_moverel_clicked(self):
        self.moverel_dialog.rundialog()
        self.show_pos()

    def on_moveabs_clicked(self):
        self.moveabs_dialog.rundialog()
        self.show_pos()

    @pyqtSlot(np.ndarray)
    def update_plot(self, spec):
        spec = self.spectrum.get_spec(self.ui.CheckBox_correct.isChecked())
        self.axes.plot(self._wl, spec)
        self.Canvas.draw()
        #self.show_pos()
        return True







if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    main = SCNR()
    main.show()
    sys.exit(app.exec_())
