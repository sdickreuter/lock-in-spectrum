import datetime

__author__ = 'sei'

import random
import os
import matplotlib
matplotlib.use("Qt5Agg")
#import pandas
import PIStage
from spectrum import Spectrum
from settings import Settings
from PyQt5.QtCore import pyqtSlot, QTimer, QSocketNotifier, QAbstractTableModel, Qt, QVariant, QModelIndex
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy, QVBoxLayout, QFileDialog, QInputDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import dialogs
from threads import GamepadThread
from PyQt5 import uic
#from sys import path
#Ui_MainWindow = uic.loadUiType("ui"+path.sep+"SCNR_main.ui")[0]
Ui_MainWindow = uic.loadUiType("ui/SCNR_main.ui")[0]
#from ui.SCNR_main import Ui_MainWindow


class NumpyModel(QAbstractTableModel):
    def __init__(self, nmatrix, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._matrix = nmatrix

    def update(self, narray):
        self._matrix = narray.copy()
        self.layoutChanged.emit()

    def getMatrix(self):
        return self._matrix.copy()

    def rowCount(self, parent=None):
        return self._matrix.shape[0]

    def columnCount(self, parent=None):
        return self._matrix.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole or role == Qt.EditRole :
                row = index.row()
                col = index.column()
                return QVariant("%.5f"%self._matrix[row, col])
        return QVariant()

    def setData(self, index, value, role = Qt.EditRole):
        if role == Qt.EditRole:
            try:
                val = float(value)
            except:
                return False
            self._matrix[index.row(),index.column()] = val
            self.dataChanged.emit(index, index, ())
            return True
        return False

    def flags(self, index):
        #if (index.column() == 0):
        return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
        #else:
        #    return Qt.ItemIsEnabled

    def headerData(self, col, orientation, role):
        if role != Qt.DisplayRole:
            return QVariant()

        if orientation == Qt.Horizontal:
            if col == 0:
                return 'x'
            if col == 1:
                return 'y'
        return QVariant()



class SCNR(QMainWindow):
    _window_title = "SCNR"
    _heartbeat = 100  # ms delay at which the plot/gui is refreshed, and the gamepad moves the stage

    def __init__(self, parent=None):
        super(SCNR, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.positions = np.matrix([ [0.0,0.0], [0.0,10.0], [10.0,0.0]])
        self.posModel = NumpyModel(self.positions)
        self.ui.posTable.setModel(self.posModel)
        self.vh = self.ui.posTable.verticalHeader()
        self.vh.setVisible(False)
        self.hh = self.ui.posTable.horizontalHeader()
        self.hh.setModel(self.posModel)
        self.hh.setVisible(True)

        self.settings = Settings()

        self.fig = Figure()
        self.axes = self.fig.add_subplot(111)
        self.axes.hold(False)
        #self.axes.autoscale(False)
        #self.axes.set_xlim([self.settings.min_wl, self.settings.max_wl])


        self.Canvas = FigureCanvas(self.fig)
        self.Canvas.setParent(self.ui.plotwidget)

        self.Canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Canvas.updateGeometry()

        l = QVBoxLayout(self.ui.plotwidget)
        l.addWidget(self.Canvas)

        self.ui.status.setText("Ready")

        #self.savedir = "."+path.sep+"Spectra"+path.sep
        #self.path = "."+path.sep

        self.savedir = "./Spectra/"
        self.path = "./"

        self.x_step = .0
        self.y_step = .0
        self.step_distance = 1  # in um

        try:
            #pass
            self.stage = PIStage.E545(self.settings.stage_ip,self.settings.stage_port)
        except:
            self.stage = None
            self.stage = PIStage.Dummy()
            print("Could not initialize PIStage, using Dummy instead")

        self.spectrum = Spectrum(self.stage, self.settings, self.ui.status, self.ui.progressBar, self.enable_buttons,
                                 self.disable_buttons)  # logger class which coordinates the spectrometer and the stage

        self.spec = self.spectrum.get_spec()  # get an initial spectrum for display
        self._wl = self.spectrum.get_wl()  # get the wavelengths
        #self.update_plot(None)
        #self.spectrum.getspecthread.dynamicSpecSignal.connect(self.update_plot)
        self.spectrum.specSignal.connect(self.update_plot)
        self.spectrum.updatePositions.connect(self.update_positions)
        self.padthread = GamepadThread()
        self.padthread.BSignal.connect(self.on_search_clicked)
        self.padthread.XSignal.connect(self.on_addpos_clicked)
        self.padthread.YSignal.connect(self.on_stepup_clicked)
        self.padthread.ASignal.connect(self.on_stepdown_clicked)
        self.padthread.xaxisSignal.connect(self.on_xaxis)
        self.padthread.yaxisSignal.connect(self.on_yaxis)
        self.ax = 0.0
        self.ay = 0.0

        self.padthread.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_pad_analog)
        self.timer.start(100)
        self.pad_active = True

        self.settings_dialog = dialogs.Settings_Dialog(self.settings)
        self.settings_dialog.updateSignal.connect(self.update_settings)
        self.update_settings()
        self.ui.label_stepsize.setText(str(self.settings.stepsize))


    def disable_buttons(self):
        self.ui.tabWidget.setDisabled(True)
        self.ui.stage_frame.setDisabled(True)
        self.ui.Button_searchmax.setDisabled(True)
        self.ui.Button_stepup.setDisabled(True)
        self.ui.Button_stepdown.setDisabled(True)
        self.ui.Button_stop.setDisabled(False)
        #self.pad_active = False


    def enable_buttons(self):
        self.ui.tabWidget.setDisabled(False)
        self.ui.stage_frame.setDisabled(False)
        self.ui.Button_searchmax.setDisabled(False)
        self.ui.Button_stepup.setDisabled(False)
        self.ui.Button_stepdown.setDisabled(False)
        self.ui.Button_stop.setDisabled(True)
        self.pad_active = True

    @pyqtSlot()
    def update_settings(self):
        self.spectrum._spectrometer.integration_time_micros(self.settings.integration_time*1000)

    @pyqtSlot(float)
    def on_xaxis(self, x):
        self.ax = x

    @pyqtSlot(float)
    def on_yaxis(self, y):
        self.ay = y

    @pyqtSlot()
    def check_pad_analog(self):
        if self.pad_active:
            x_step = self.ax
            if abs(x_step) > 0.001:
                x_step = x_step * self.settings.stepsize
            else:
                x_step = 0.0

            y_step = self.ay
            if abs(y_step) > 0.001:
                y_step = y_step * self.settings.stepsize
            else:
                y_step = 0.0

            if abs(x_step) > 0.0001:
                if abs(y_step) > 0.0001:
                    self.stage.moverel(dx=x_step, dy=y_step)
                else:
                    self.stage.moverel(dx=x_step)
            elif abs(y_step) > 0.001:
                self.stage.moverel(dy=y_step)
            self.show_pos()

    # ## ----------- scan Listview connect functions

    @pyqtSlot()
    def on_addpos_clicked(self):
        self.stage.query_pos()
        x,y,z = self.stage.last_pos()
        positions = self.posModel.getMatrix()
        if positions.shape[1] == 2:
            positions = np.append(positions,np.matrix([x,y]), axis = 0)
        else:
            positions = np.matrix([x,y])
        self.posModel.update(positions)

    @pyqtSlot()
    def on_spangrid_clicked(self):
        xl, yl, ok = dialogs.SpanGrid_Dialog.getXY()
        positions = self.posModel.getMatrix()
        if (positions.shape[0] >= 3) & ((xl is not 0) | (yl is not 0)):
            a = np.ravel(positions[0,:])
            b = np.ravel(positions[1,:])
            c = np.ravel(positions[2,:])
            grid = np.zeros((xl*yl,2))
            if abs(b[0]) > abs(c[0]):
                grid_vec_1 = [b[0] - a[0], b[1] - a[1]]
                grid_vec_2 = [c[0] - a[0], c[1] - a[1]]
            else:
                grid_vec_2 = [b[0] - a[0], b[1] - a[1]]
                grid_vec_1 = [c[0] - a[0], c[1] - a[1]]

            print(grid_vec_1)
            print(grid_vec_2)
            i = 0
            for x in range(xl):
                for y in range(yl):
                    vec_x = a[0] + grid_vec_1[0] * x + grid_vec_2[0] * y
                    vec_y = a[1] + grid_vec_1[1] * x + grid_vec_2[1] * y
                    grid[i,0] = vec_x
                    grid[i,1] = vec_y
                    i += 1

            self.posModel.update(grid)

    @pyqtSlot()
    def on_scan_add(self):
        positions = self.posModel.getMatrix()
        if positions.shape[1] == 2:
            positions = np.append(positions,np.matrix([0.0,0.0]), axis = 0)
        else:
            positions = np.matrix([0.0,0.0])
        self.posModel.update(positions)


    @pyqtSlot()
    def on_scan_remove(self):
        indices = self.ui.posTable.selectionModel().selectedIndexes()
        rows = np.array([],dtype=int)
        for index in indices:
            rows = np.append(rows,index.row())
        positions = self.posModel.getMatrix()
        positions = np.delete(positions,rows,axis=0)
        self.posModel.update(positions)

    @pyqtSlot()
    def on_scan_clear(self):
        self.posModel.update(np.matrix([[]]))

    @pyqtSlot(np.ndarray)
    def update_positions(self, pos):
        self.posModel.update(pos)

    # ## ----------- END scan Listview connect functions

    # ##---------------- button connect functions ----------

    @pyqtSlot()
    def on_start_scan_clicked(self):
        prefix, ok = QInputDialog.getText(self, 'Save Folder',
            'Enter Folder to save spectra to:')

        if ok:
            try:
                # os.path.exists(prefix)
                os.mkdir(self.savedir+prefix)
            except:
                #print("Error creating directory ."+path.sep + prefix)
                print("Error creating directory ./" + prefix)
            #path = self.savedir + prefix + path.sep
            path = self.savedir + prefix + "/"
            self.ui.status.setText("Scanning ...")
            #self.spectrum.make_scan(self.scan_store, path, self.button_searchonoff.get_active(), self.button_lockinonoff.get_active())
            self.spectrum.make_scan(self.posModel.getMatrix(), path, self.ui.checkBox_lockin.isChecked(), self.ui.checkBox_search.isChecked())
            self.disable_buttons()


    @pyqtSlot()
    def on_stop_clicked(self):
        self.ui.status.setText('Stopped')
        self.spectrum.stop_process()
        self.enable_buttons()

    @pyqtSlot()
    def on_reset_clicked(self):
        self.spectrum.reset()

    @pyqtSlot()
    def on_acquirelockin_clicked(self):
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
    def on_searchgrid_clicked(self):
        self.ui.status.setText("Searching Max.")
        self.spectrum.scan_search_max(self.posModel.getMatrix())
        self.disable_buttons()

    @pyqtSlot()
    def on_search_clicked(self):
        self.ui.status.setText("Searching Max.")
        self.spectrum.search_max()
        self.disable_buttons()

    @pyqtSlot()
    def on_save_clicked(self):
        self.ui.status.setText("Saving Data ...")
        prefix, ok = QInputDialog.getText(self, 'Save Folder',
            'Enter Folder to save spectra to:')
        if ok:
            try:
                # os.path.exists(prefix)
                os.mkdir(self.savedir+prefix)
            except:
                #print("Error creating directory ."+path.sep + prefix)
                print("Error creating directory ./" + prefix)
            #path = self.savedir + prefix + path.sep
            path = self.savedir + prefix + "/"
            self.spectrum.save_data(path)

    @pyqtSlot()
    def on_saveas_clicked(self):
        self.ui.status.setText("Saving Data ...")
        save_as = QFileDialog.getSaveFileName(self, "Save currently shown Spectrum as", './spectra/','CSV Files (*.csv)')
        print(save_as[0])
        #prefix, ok = QInputDialog.getText(self, 'Save Folder', 'Enter Folder to save spectra to:')
        if not self.spectrum.mean is None:
            try:
                self.spectrum.save_spectrum(self.spectrum.mean, save_as[0], None, False, True)
            except:
                print("Error Saving file " + save_as[0])

    @pyqtSlot()
    def on_settings_clicked(self):
        self.settings_dialog.show()

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
                #print("Error creating directory ."+path.sep + prefix)
                print("Error creating directory ./" + prefix)
            #path = self.savedir + prefix + path.sep
            path = self.savedir + prefix + "/"
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
    def on_loadlamp_clicked(self):
        buf = self._load_spectrum_from_file()
        if not buf is None:
            self.spectrum.lamp = buf

    @pyqtSlot()
    def on_loadbg_clicked(self):
        buf = self._load_spectrum_from_file()
        if not buf is None:
            self.spectrum.bg = buf

    @pyqtSlot()
    def on_lockin_clicked(self):
        pass

    @pyqtSlot()
    def on_aquirelockin_clicked(self):
        pass

    # ##---------------- END button connect functions ----------

    def _load_spectrum_from_file(self):
        #save_dir = QFileDialog.getOpenFileName(self, "Load Spectrum from CSV", os.path.expanduser('~'), 'CSV Files (*.csv)')
        #save_dir = QFileDialog.getOpenFileName(self, "Load Spectrum from CSV", '.'+path.sep+'spectra'+path.sep, 'CSV Files (*.csv)')
        save_dir = QFileDialog.getOpenFileName(self, "Load Spectrum from CSV", './spectra/', 'CSV Files (*.csv)')

        if len(save_dir[0])>1:
            save_dir = save_dir[0]
            #data = pandas.DataFrame(pandas.read_csv(save_dir,skiprows=8))
            #data = data['counts']
            data = np.genfromtxt(save_dir, delimiter=',',skip_header=8)
            data = data[:,1]
            return np.array(data)
        return None

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
    def update_plot(self):
        self.spec = self.spectrum.get_spec(self.ui.CheckBox_correct.isChecked())
        mask = (self._wl >= self.settings.min_wl) & (self._wl <= self.settings.max_wl)
        self.axes.plot(self._wl[mask], self.spec[mask])
        #self.axes.plot(self._wl, spec)
        self.Canvas.draw()
        self.show_pos()
        return True


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    main = SCNR()
    main.show()
    sys.exit(app.exec_())
