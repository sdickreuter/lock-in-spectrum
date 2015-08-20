__author__ = 'sei'

from datetime import datetime
import math
import sys
#import seabreeze.spectrometers as sb
import oceanoptics
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas
from progress import *
from PyQt5.QtCore import pyqtSlot, QThread, QMutex, QMutexLocker, QWaitCondition , pyqtSignal

# modified from: http://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m#comment33999040_21566831
def gauss2D(pos, amplitude, xo, yo, fwhm, offset):
    sigma = fwhm / 2.3548
    xo = float(xo)
    yo = float(yo)
    g = offset + amplitude * np.exp(
        -( np.power(pos[0] - xo, 2.) + np.power(pos[1] - yo, 2.) ) / (2 * np.power(sigma, 2.)))
    return g.ravel()

def gauss(x, amplitude, xo, fwhm, offset):
    sigma = fwhm / 2.3548
    xo = float(xo)
    g = offset + amplitude * np.exp(-np.power(x - xo, 2.) / (2 * np.power(sigma, 2.)))
    return g.ravel()

def smooth(x):
    """
    modified from: http://wiki.scipy.org/Cookbook/SignalSmooth
    """
    window_len = 151

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]

    window = 'hanning'
    # window='flat'

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    y = y[(window_len / 2):-(window_len / 2)]
    return y

def _millis(starttime):
    dt = datetime.now() - starttime
    ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
    return ms


class GetSpectrumThread(QThread):

    dynamicSpecSignal = pyqtSignal(np.ndarray)

    def __init__(self, spectrometer, parent=None):
        if getattr(self.__class__, '_has_instance', False):
            print('Cannot create another instance')
            return None
        self.__class__._has_instance = True

        super(GetSpectrumThread, self).__init__(parent)
        self.spectrometer = spectrometer
        self.mutex = QMutex()
        self.condition = QWaitCondition()

    def __del__(self):
        self.mutex.lock()
        self.abort = True
        self.condition.wakeOne()
        self.mutex.unlock()

        self.wait()

    def getSpectrum(self):
    #    #locker = QMutexLocker(self.mutex)
        if not self.isRunning():
            self.start(QThread.HighPriority)

    def run(self):
        self.mutex.lock()
        spec = self.spectrometer.intensities()
        self.mutex.unlock()
        self.dynamicSpecSignal.emit(spec)


class MeasurementThread(QThread):

    specSignal = pyqtSignal(np.ndarray)
    progressSignal = pyqtSignal(float)
    finishSignal = pyqtSignal(np.ndarray)
    waitCondition = QWaitCondition()
    abort = False

    def __init__(self, getspecthread, parent=None):
        if getattr(self.__class__, '_has_instance', False):
            RuntimeError('Cannot create another instance')
        self.__class__._has_instance = True

        super(MeasurementThread, self).__init__(parent)
        self.getspecthread = getspecthread
        self.mutex = QMutex()
        self.getspecthread.dynamicSpecSignal.connect(self.specCallback)
        self.start(QThread.HighPriority)

    def __del__(self):
        self.mutex.lock()
        self.__class__.has_instance = False
        self.getspecthread.dynamicSpecSignal.disconnect(self.specCallback)
        try:
            self.progressSignal.disconnect()
            self.finishSignal.disconnect()
        except TypeError:
            pass
        self.abort = True
        self.waitCondition.wakeOne()
        self.mutex.unlock()
        self.wait()

    def stop(self):
        self.abort=True
        self.waitCondition.wakeOne()

    def work(self):
        #pass #print(self.spec)
        self.specSignal.emit(self.spec)

    def run(self):
        while True:
            if self.abort:
                return
            self.getspecthread.getSpectrum()
            self.mutex.lock()
            self.waitCondition.wait(self.mutex)
            self.mutex.unlock()
            try:
                self.work()
            except:
                (type, value, traceback) = sys.exc_info()
                sys.excepthook(type, value, traceback)

    @pyqtSlot(np.ndarray)
    def specCallback(self, spec):
        self.waitCondition.wakeOne()
        self.spec = spec


class MeanThread(MeasurementThread):
    def __init__(self, getspecthread, number_of_samples, parent=None):
        self.number_of_samples = number_of_samples
        self.mean = np.zeros(1024, dtype=np.float)
        self.i = 0
        super(MeanThread, self).__init__(getspecthread)

    def work(self):
        self.mutex.lock()
        print(self.i)
        self.mean = (self.mean + self.spec)  # / 2
        self.mutex.unlock()
        self.specSignal.emit(self.mean/(self.i+1))
        progressFraction = float(self.i + 1) / self.number_of_samples
        self.progressSignal.emit(progressFraction*100)
        self.mutex.lock()
        self.i += 1
        if self.i >= self.number_of_samples:
            self.abort = True
            self.finishSignal.emit(self.mean/(self.number_of_samples))
        self.mutex.unlock()



class ScanMeanThread(MeanThread):
    def __init__(self, getspecthread, settings, scanning_points,parent=None):
        self.points = scanning_points
        self.settings = settings
        super(ScanMeanThread, self).__init__(getspecthread, settings.number_of_samples)

    def work(self):
        super(ScanMeanThread, self).work()



class SearchThread(MeasurementThread):

    def __init__(self, getspecthread, settings, scanning_points, stage, parent=None):
        try:
            self.scanning_points = scanning_points
            self.settings = settings
            self.stage = stage
            self.i = 0
            self.n = scanning_points.shape[0]
            self.positions = np.zeros((self.n,2))
            super(SearchThread, self).__init__(getspecthread)
            self.spectrometer = self.getspecthread.spectrometer
        except:
            (type, value, traceback) = sys.exc_info()
            sys.excepthook(type, value, traceback)

    def work(self):
        self.mutex.lock()
        self.stage.moveabs(x=self.scanning_points[self.i,0],y=self.scanning_points[self.i,1])
        self.mutex.unlock()
        self.search()
        self.mutex.lock()
        x, y, z = self.stage.last_pos()
        self.positions[self.i,0] = x
        self.positions[self.i,1] = y
        progressFraction = float(self.i + 1) / self.n
        self.progressSignal.emit(progressFraction*100)
        print(self.positions)
        self.i += 1
        print(self.i)
        if self.i >= self.n:
            self.abort = True
            self.finishSignal.emit(self.positions)
        self.mutex.unlock()

    def search(self):
        self.mutex.lock()
        #self.spectrometer.integration_time_micros(self.settings.search_integration_time * 1000)
        spec = smooth(self.spectrometer.intensities())
        self.mutex.unlock()

        minval = np.min(spec)
        maxval = np.max(spec)

        d = np.linspace(-self.settings.rasterwidth, self.settings.rasterwidth, self.settings.rasterdim)

        repetitions = 4
        for j in range(repetitions):
            self.mutex.lock()
            self.stage.query_pos()
            origin = self.stage.last_pos()
            measured = np.zeros(self.settings.rasterdim)
            self.mutex.unlock()
            if j is 4:
                d /= 2
            if j % 2:
                pos = d + origin[0]
            else:
                pos = d + origin[1]

            for k in range(len(pos)):
                self.mutex.lock()
                if j % 2:
                    self.stage.moveabs(x=pos[k])
                else:
                    self.stage.moveabs(y=pos[k])
                spec = smooth(self.spectrometer.intensities(correct_nonlinearity=True))
                self.mutex.unlock()
                self.specSignal.emit(spec)
                measured[k] = np.max(spec)
                #print(k)
            maxind = np.argmax(measured)

            self.mutex.lock()
            initial_guess = (maxval - minval, pos[maxind], self.settings.sigma, minval)
            self.mutex.unlock()
            dx = origin[0]
            dy = origin[1]
            try:
                popt, pcov = opt.curve_fit(gauss, pos[2:(len(pos) - 1)], measured[2:(len(pos) - 1)], p0=initial_guess)
                perr = np.diag(pcov)
                if perr[1] > 1:
                    RuntimeError("Could not determine particle position: Variance to big")
            except RuntimeError as e:
                print(e)
                print("Could not determine particle position: Fit error")
            else:
                if j % 2:
                    dx=float(popt[1])
                else:
                    dy=float(popt[1])
                    # print(popt)
            self.mutex.lock()
            plt.figure()
            plt.plot(pos, measured,'bo')
            x = np.linspace(min(pos),max(pos))
            plt.plot(x, gauss(x,popt[0],popt[1],popt[2],popt[3]),'g-')
            plt.savefig("search_max/search" + str(j) + ".png")
            plt.close()
            self.stage.moveabs(x=dx,y=dy)
            self.mutex.unlock()
        self.mutex.lock()
        #self._spectrometer.integration_time_micros(self.settings.integration_time / 1000)
        spec = self.spectrometer.intensities(correct_nonlinearity=True)
        self.stage.query_pos()
        self.mutex.unlock()
        self.specSignal.emit(spec)



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
        #self._spectrometer = oceanoptics.Dummy()
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

    def get_spec(self, corrected = False):
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
        self.workingthread.wait()
        self.workingthread = None

    def take_live(self):
        self.workingthread = MeasurementThread(self.getspecthread)
        self.workingthread.specSignal.connect(self.specCallback)

    @pyqtSlot(np.ndarray)
    def finishedDarkCallback(self, spec):
        self.dark = spec

    @pyqtSlot(np.ndarray)
    def finishedLampCallback(self, spec):
        self.lamp = spec

    @pyqtSlot(np.ndarray)
    def finishedMeanCallback(self, spec):
        self.mean = spec

    @pyqtSlot(np.ndarray)
    def finishedBGCallback(self, spec):
        self.bg = spec

    def startMeanThread(self):
        self.workingthread = MeanThread(self.getspecthread,self.settings.number_of_samples)
        self.workingthread.specSignal.connect(self.specCallback)
        self.workingthread.progressSignal.connect(self.progressCallback)

    def take_dark(self):
        self.startMeanThread()
        self.workingthread.finishSignal.connect(self.finishedDarkCallback)

    def take_lamp(self):
        self.startMeanThread()
        self.workingthread.finishSignal.connect(self.finishedLampCallback)

    def take_mean(self):
        self.startMeanThread()
        self.workingthread.finishSignal.connect(self.finishedMeanCallback)

    def take_bg(self):
        self.startMeanThread()
        self.workingthread.finishSignal.connect(self.finishedBGCallback)

    def take_lockin(self):
        self.worker_mode = "lockin"
        # self.reset()
        self.lockin = None
        self._data = np.ones((self.settings.number_of_samples, 1026), dtype=np.float)
        self.start_process(self._lockin_spectrum)

    def search_max(self):
        self.stage.query_pos()
        x,y,z = self.stage.last_pos()
        pos = np.matrix([[x,y]])
        print(pos)
        #SearchThread(self, getspecthread, settings, scanning_points, stage, parent=None):
        self.workingthread = SearchThread(self.getspecthread,self.settings,pos,self.stage)
        self.workingthread.specSignal.connect(self.specCallback)
        self.workingthread.progressSignal.connect(self.progressCallback)
        self.workingthread.finishSignal.connect(self.finishedSearchCallback)

    @pyqtSlot(np.ndarray)
    def finishedSearchCallback(self, pos):
        self._positions = pos
        print(pos)

    def take_series(self, path):
        self.series_path = path
        self.save_data(self.series_path)
        self.worker_mode = "series"
        self.series_count = 0
        self.start_process(self._live_spectrum)

    def make_scan(self, points, path, search=False, lockin=False):
        self.scanner_lockin = lockin
        self.scanner_search = search
        self.scanner_points = points
        self.scanner_path = path
        self.scanner_index = 0
        self.worker_mode = "scan"
        self.scanner_mode = "start"
        self.map = list()
        self.peakpos = list()
        self.x = list()
        self.y = list()
        self.progress = Progress(max=len(self.scanner_points))
        self._callback_scan()

    @pyqtSlot()
    def callback(self):
        if self.worker_mode is "scan":
            self._callback_scan()
        else:
            self._callback_normal()
        return True

    def _callback_normal(self):

        if self.worker_mode is "lockin":
            finished, self._progress_fraction, spec, ref, i = self.conn_for_main.recv()
            self._data[i, 0] = i
            self._data[i, 1] = ref
            self._data[i, 2:] = spec
        elif self.worker_mode is "series":
            finished, self._progress_fraction, spec = self.conn_for_main.recv()
            self._progress_fraction = self.series_count / self.settings.number_of_samples
            self.series_count += 1
            self.save_spectrum(spec, self.series_path + str(self.series_count).zfill(5) + ".csv")
            if self.series_count >= self.settings.number_of_samples:
                finished = True
        else:
            finished, self._progress_fraction, spec = self.conn_for_main.recv()

        self.progressbar.set_fraction(self._progress_fraction)

        if spec is not None:
            self._spec = spec

        if finished:
            self.running.clear()

            if self.worker_mode is "lockin":
                self.lockin = self.calc_lockin()
                self._spec = self.lockin
                self.status.setText('Lock-In Spectrum acquired')
            elif self.worker_mode is "lamp":
                self.lamp = self._spec
                self.status.showMessage('Lamp Spectrum taken',5000)
            elif self.worker_mode is "dark":
                self.dark = self._spec
                self.status.showMessage('Dark Spectrum taken',5000)
            elif self.worker_mode is "normal":
                self.normal = self._spec
                self.status.showMessage('Normal Spectrum taken',5000)
            elif self.worker_mode is "bg":
                self.bg = self._spec
                self.status.showMessage('Background Spectrum taken',5000)
            elif self.worker_mode is "search":
                # self.show_pos()
                self.status.showMessage("Max. approached",5000)
            elif self.worker_mode is "series":
                # self.show_pos()
                self.status.showMessage("Time series finished",5000)

                # if not self.worker_mode is "lockin":
                #if not self.dark is None:
                #    self._spec = self._spec - self.dark
                #    if not self.lamp is None:
                #        # self._spec = self._spec / self.lamp
                #        self._spec = self._spec / (self.lamp - self.dark)

        if not self.running.is_set():
            self.worker.join(0.5)
            self.enable_buttons()
            self.worker_mode = None

        return True

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
        self.progressbar.setValue(self.scanner_index / (len(self.scanner_points))*100)

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

