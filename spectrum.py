__author__ = 'sei'

from datetime import datetime
import math
# import seabreeze.spectrometers as sb
import oceanoptics
import pandas
from progress import *
from threads import *


class Spectrum(object):
    def __init__(self, stage, settings, status, progressbar, enable_buttons, disable_buttons):
        self.settings = settings
        self.stage = stage
        self.status = status
        self.progressbar = progressbar
        self.enable_buttons = enable_buttons
        self.disable_buttons = disable_buttons
        self._init_spectrometer()
        self._cycle_time_start = 60
        self._data = None
        self.prev_time = None

        # variables for storing the spectra
        self.lamp = None
        self.dark = None
        self.mean = None
        self.bg = None
        self.lockin = None

        self._spec = np.zeros(1024, dtype=np.float)
        self._spec = self._spectrometer.intensities(correct_nonlinearity=True)

        self.getspecthread = GetSpectrumThread(self._spectrometer)
        self.workingthread = None

    def __del__(self):
        self._spectrometer = None

    def _init_spectrometer(self):

        #    try:
        #        devices = sb.list_devices()
        #        self._spectrometer = sb.Spectrometer(devices[0])
        #        self._spectrometer.tec_set_temperature_C(-17)
        #        self._spectrometer.tec_set_enable(True)
        #        print(self._spectrometer.tec_get_temperature_C())
        #        self._spectrometer.integration_time_micros(100000)
        #        print("Spectrometer " + str(self._spectrometer.serial_number) + " initialized and working")
        #    except:
        #        print("Error opening Spectrometer, using Dummy instead")
        # self._spectrometer = oceanoptics.Dummy()
        self._spectrometer = oceanoptics.ParticleDummy(self.stage)
        self._spectrometer._set_integration_time(100)
        sp = self._spectrometer.spectrum()
        self._wl = np.array(sp[0], dtype=np.float)

    @pyqtSlot(float)
    def progressCallback(self, progress):
        self.progressbar.setValue(progress)

    @pyqtSlot(np.ndarray)
    def specCallback(self, spec):
        self._spec = spec

    def get_wl(self):
        return self._wl

    def get_spec(self, corrected=False):
        if corrected:
            if not self.dark is None:
                if not self.bg is None:
                    if not self.lamp is None:
                        return (self._spec - self.bg) / (self.lamp - self.dark)
                    return self._spec - self.bg
                else:
                    if not self.lamp is None:
                        return (self._spec - self.dark) / (self.lamp - self.dark)
                    return self._spec - self.dark
        return self._spec

    def move_stage(self, dist):
        x = self._startx + self.settings.amplitude / 2 * dist * self.settings.direction_x
        y = self._starty + self.settings.amplitude / 2 * dist * self.settings.direction_y
        z = self._startz + self.settings.amplitude / 2 * dist * self.settings.direction_z
        # print "X: {0:+8.4f} | Y: {1:8.4f} | Z: {2:8.4f} || X: {3:+8.4f} | Y: {4:8.4f} | Z: {5:8.4f}".format(x,y,z,self._startx,self._starty,self._startz)
        self.stage.moveabs(x=x, y=y, z=z)

    def calc_lockin(self):
        res = np.empty(1024)
        for i in range(1024):
            buf = self._data[:, i + 2]
            buf = buf * self._data[:, 1]
            buf = np.sum(buf)
            res[i] = buf
        return res

    def stop_process(self):
        self.workingthread.stop()
        self.workingthread = None
        # self.enable_buttons()

    def take_live(self):
        self.workingthread = LiveThread(self.getspecthread)
        self.workingthread.specSignal.connect(self.specCallback)
        self.workingthread.start(QThread.HighPriority)

    @pyqtSlot(np.ndarray)
    def finishedDarkCallback(self, spec):
        self.dark = spec
        self.enable_buttons()
        self.status.setText('Dark Spectrum acquired')
        self.workingthread = None

    @pyqtSlot(np.ndarray)
    def finishedLampCallback(self, spec):
        self.lamp = spec
        self.enable_buttons()
        self.status.setText('Lamp Spectrum acquired')
        self.workingthread = None

    @pyqtSlot(np.ndarray)
    def finishedMeanCallback(self, spec):
        self.mean = spec
        self.enable_buttons()
        self.status.setText('Mean Spectrum acquired')
        self.workingthread = None

    @pyqtSlot(np.ndarray)
    def finishedBGCallback(self, spec):
        self.bg = spec
        self.enable_buttons()
        self.status.setText('Background Spectrum acquired')
        self.workingthread = None

    def startMeanThread(self):
        self.workingthread = MeanThread(self.getspecthread, self.settings.number_of_samples)
        self.workingthread.specSignal.connect(self.specCallback)
        self.workingthread.progressSignal.connect(self.progressCallback)

    def take_dark(self):
        self.startMeanThread()
        self.workingthread.finishSignal.connect(self.finishedDarkCallback)
        self.workingthread.start(QThread.HighPriority)


    def take_lamp(self):
        self.startMeanThread()
        self.workingthread.finishSignal.connect(self.finishedLampCallback)
        self.workingthread.start(QThread.HighPriority)

    def take_mean(self):
        self.startMeanThread()
        self.workingthread.finishSignal.connect(self.finishedMeanCallback)
        self.workingthread.start(QThread.HighPriority)

    def take_bg(self):
        self.startMeanThread()
        self.workingthread.finishSignal.connect(self.finishedBGCallback)
        self.workingthread.start(QThread.HighPriority)

    def take_lockin(self):
        self.worker_mode = "lockin"
        # self.reset()
        self.lockin = None
        self._data = np.ones((self.settings.number_of_samples, 1026), dtype=np.float)
        self.start_process(self._lockin_spectrum)

    def search_max(self):
        self.workingthread = SearchThread(self.getspecthread, self.settings, self.stage)
        self.workingthread.specSignal.connect(self.specCallback)
        self.workingthread.progressSignal.connect(self.progressCallback)
        self.workingthread.finishSignal.connect(self.finishedSearchCallback)
        self.workingthread.start(QThread.HighPriority)

    def scan_search_max(self):
        self.stage.query_pos()
        x, y, z = self.stage.last_pos()
        pos = np.matrix([[x, y]])
        self.workingthread = ScanSearchThread(self.getspecthread, self.settings, pos, self.stage)
        self.workingthread.specSignal.connect(self.specCallback)
        self.workingthread.progressSignal.connect(self.progressCallback)
        self.workingthread.finishSignal.connect(self.finishedSearchCallback)
        self.workingthread.start(QThread.HighPriority)

    @pyqtSlot(np.ndarray)
    def finishedSearchCallback(self, pos):
        print(pos)
        self.enable_buttons()
        self.status.setText('Search finished')
        self.workingthread = None


    def take_series(self, path):
        self.series_path = path
        self.save_data(self.series_path)
        self.worker_mode = "series"
        self.series_count = 0
        self.start_process(self._live_spectrum)

    def _callback_scan(self):

        def start():
            self.scanner_point = self.scanner_points[self.scanner_index]
            self.stage.moveabs(x=self.scanner_point[0], y=self.scanner_point[1])
            # self.reset()
            if self.scanner_search:
                self.scanner_mode = "search"
                self.start_process(self._search_max_int)
            elif self.scanner_lockin:
                self.scanner_mode = "lockin"
                self.start_process(self._lockin_spectrum)
            else:
                self.scanner_mode = "mean"
                self.start_process(self._mean_spectrum)

        def finish():
            self.running.clear()
            map = np.ones((len(self.x), 4), dtype=np.float)
            map[:, 0] = self.x
            map[:, 1] = self.y
            map[:, 2] = self.map
            map[:, 3] = self.peakpos
            map = pandas.DataFrame(map, columns=('x', 'y', 'int', 'peak'))
            filename = self.scanner_path + 'map.csv'
            map.to_csv(filename, header=True, index=False)
            if not self.dark is None:
                self.save_spectrum(self.dark, self.scanner_path + "dark.csv")
            if not self.lamp is None:
                self.save_spectrum(self.lamp, self.scanner_path + "lamp.csv")
            if not self.bg is None:
                self.save_spectrum(self.bg, self.scanner_path + "background.csv")
            self.status.set_text("Scan complete")
            self.scanner_index += 1

        if self.scanner_mode is "start":
            start()

        if self.scanner_mode is "search":
            finished, self._progress_fraction, spec = self.conn_for_main.recv()
            if spec is not None:
                self._spec = spec
            if finished:
                self.worker.join(0.5)
                if self.scanner_lockin:
                    self.scanner_mode = "lockin"
                    self.start_process(self._lockin_spectrum)
                else:
                    self.scanner_mode = "mean"
                    self.start_process(self._mean_spectrum)

        if self.scanner_mode is "lockin":
            finished, self._progress_fraction, spec, ref, i = self.conn_for_main.recv()
            self._data[i, 0] = i
            self._data[i, 1] = ref
            self._data[i, 2:] = spec
            self._spec = spec
            if finished:
                self.worker.join(0.5)
                self.lockin = self.calc_lockin()
                self._spec = self.lockin
                smooth = self.smooth(self._spec)
                maxind = np.argmax(smooth)
                self.map.append(smooth[maxind])
                self.peakpos.append(self._wl[maxind])
                self.x.append(self.scanner_point[0])
                self.y.append(self.scanner_point[1])
                self.save_spectrum(self.lockin, self.scanner_path + str(self.scanner_index).zfill(5) + ".csv",
                                   self.scanner_point)
                self.scanner_index += 1
                self.progress.next()
                if self.scanner_index >= len(self.scanner_points):
                    finish()
                else:
                    start()

        if self.scanner_mode is "mean":
            finished, self._progress_fraction, spec = self.conn_for_main.recv()
            self._spec = spec
            if finished:
                self.worker.join(0.5)
                self.normal = spec
                self._spec = spec
                smooth = self.smooth(self._spec)
                maxind = np.argmax(smooth)
                self.map.append(smooth[maxind])
                self.peakpos.append(self._wl[maxind])

                if self.scanner_search:
                    pos = self.stage.last_pos()
                    self.x.append(pos[0])
                    self.y.append(pos[1])
                else:
                    self.x.append(self.scanner_point[0])
                    self.y.append(self.scanner_point[1])
                self.save_spectrum(self.normal, self.scanner_path + str(self.scanner_index).zfill(5) + ".csv",
                                   self.scanner_point)
                self.scanner_index += 1
                self.progress.next()
                if self.scanner_index >= len(self.scanner_points):
                    finish()
                else:
                    start()

        if not self.running.is_set():
            self.worker.join(0.5)
            self.enable_buttons()
            self.worker_mode = None
            self.scanner_mode = None

        self.status.showMessage("ETA: " + str(self.progress.eta_td))
        self.progressbar.setValue(self.scanner_index / (len(self.scanner_points)) * 100)

        return True

    def _lockin_spectrum(self, connection):
        f = self.settings.f
        self.stage.query_pos()
        pos = self.stage.last_pos()
        self._startx = pos[0]
        self._starty = pos[1]
        self._startz = pos[2]
        starttime = datetime.now()

        for i in range(self.settings.number_of_samples):
            ref = math.cos(2 * math.pi * i * f)
            self.move_stage(ref / 2)
            spec = self._spectrometer.intensities(correct_nonlinearity=True)
            progress_fraction = float(i + 1) / self.settings.number_of_samples
            connection.send([False, progress_fraction, spec, ref, i])
            if not self.running.is_set():
                return True

        print("%s spectra aquired" % self.settings.number_of_samples)
        print("time taken: %s s" % (self._millis(starttime) / 1000))
        self.stage.moveabs(x=self._startx, y=self._starty, z=self._startz)
        connection.send([True, 1., spec, ref, self.settings.number_of_samples - 1])
        return True

    def save_data(self, prefix):
        filename = self._gen_filename()
        if not self._data is None:
            cols = ('t', 'ref') + tuple(map(str, np.round(self._wl, 1)))
            data = pandas.DataFrame(self._data, columns=cols)
            data.to_csv(prefix + 'spectrum_' + filename, header=True, index=False)
        if not self.dark is None:
            self.save_spectrum(self.dark, prefix + 'dark_' + filename)
        if not self.lamp is None:
            self.save_spectrum(self.lamp, prefix + 'lamp_' + filename)
        if not self.normal is None:
            self.save_spectrum(self.normal, prefix + 'normal_' + filename)
        if not self.bg is None:
            self.save_spectrum(self.bg, prefix + 'background_' + filename)
        if not self.lockin is None:
            self.save_spectrum(self.lockin, prefix + 'lockin_' + filename, lockin=True)

    def save_spectrum(self, spec, filename, pos=None, lockin=False):
        data = np.append(np.round(self._wl, 1).reshape(self._wl.shape[0], 1), spec.reshape(spec.shape[0], 1), 1)

        f = open(filename, 'w')
        f.write(str(datetime.now().day).zfill(2) + "." + str(datetime.now().month).zfill(2) + "." + str(
            datetime.now().year) + "\r\n")
        f.write(str(datetime.now().hour).zfill(2) + ":" + str(datetime.now().minute).zfill(2) + ":" + str(
            datetime.now().second).zfill(2) + ":" + str(datetime.now().microsecond).zfill(2) + "\r\n")
        f.write("integration time [ms]" + "\r\n")
        f.write(str(self.settings.integration_time) + "\r\n")
        f.write("number of samples" + "\r\n")
        f.write(str(self.settings.number_of_samples) + "\r\n")
        if lockin:
            f.write("amplitude" + "\r\n")
            f.write(str(self.settings.amplitude) + "\r\n")
            f.write("frequency" + "\r\n")
            f.write(str(self.settings.f) + "\r\n")

        if pos is not None:
            f.write("x" + "\r\n")
            f.write(str(pos[0]) + "\r\n")
            f.write("y" + "\r\n")
            f.write(str(pos[1]) + "\r\n")

        f.write("\r\n")
        f.write("wavelength,counts" + "\r\n")
        for i in range(len(data)):
            f.write(str(data[i][0]) + "," + str(data[i][1]) + "\r\n")

        f.close()

    def on_reset_clicked(self, widget):
        self.dark = None
        self.lamp = None
        self.lockin = None
        self.mean = None

    @staticmethod
    def _gen_filename():
        return str(datetime.now().year) + str(datetime.now().month).zfill(2) \
               + str(datetime.now().day).zfill(2) + '_' + str(datetime.now().hour).zfill(2) + \
               str(datetime.now().minute).zfill(2) + str(datetime.now().second).zfill(2) + '.csv'
