__author__ = 'sei'

import sys

import numpy as np
import scipy.optimize as opt
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from PyQt5.QtCore import pyqtSlot, QThread, QMutex, QWaitCondition, pyqtSignal, QObject, QTimer
import progress
import pygamepad
import scipy.fftpack
from scipy import interpolate
from scipy import stats

class GamepadThread(QObject):
    ASignal = pyqtSignal()
    BSignal = pyqtSignal()
    XSignal = pyqtSignal()
    YSignal = pyqtSignal()

    def __init__(self, parent=None):
        if getattr(self.__class__, '_has_instance', False):
            RuntimeError('Cannot create another instance')
        self.__class__._has_instance = True
        try:
            super(GamepadThread, self).__init__(parent)
            self.abort = False
            self.thread = QThread()
            #try:
            self.pad = pygamepad.Gamepad()
            #except:
            #    print("Could not initialize Gamepad")
            #    self.pad = None
            #else:
            self.thread.started.connect(self.process)
            self.thread.finished.connect(self.stop)
            self.moveToThread(self.thread)
        except:
            (type, value, traceback) = sys.exc_info()
            sys.excepthook(type, value, traceback)

    def start(self):
        self.thread.start(QThread.HighPriority)

    @pyqtSlot()
    def stop(self):
        self.abort = True

    def check_Analog(self):
        return (self.pad.get_analogL_x(),self.pad.get_analogL_y())

    def __del__(self):
        self.__class__.has_instance = False
        try:
            self.ASignal.disconnect()
            self.BSignal.disconnect()
            self.XSignal.disconnect()
            self.YSignal.disconnect()
        except TypeError:
            pass
        self.abort = True

    def work(self):
        self.pad._read_gamepad()
        if self.pad.changed:
            if self.pad.A_was_released():
                self.ASignal.emit()
            if self.pad.B_was_released():
                self.BSignal.emit()
            if self.pad.X_was_released():
                self.XSignal.emit()
            if self.pad.Y_was_released():
                self.YSignal.emit()

    @pyqtSlot()
    def process(self):
        while not self.abort:
            try:
                self.work()
            except:
                (type, value, traceback) = sys.exc_info()
                sys.excepthook(type, value, traceback)

# modified from: http://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m#comment33999040_21566831
def gauss2D(pos, amplitude, xo, yo, fwhm, offset):
    sigma = fwhm / 2.3548
    xo = float(xo)
    yo = float(yo)
    g = offset + amplitude * np.exp(
        -(np.power(pos[0] - xo, 2.) + np.power(pos[1] - yo, 2.)) / (2 * np.power(sigma, 2.)))
    return g.ravel()


def gauss(x, amplitude, xo, fwhm, offset):
    sigma = fwhm / 2.3548
    xo = float(xo)
    g = offset + amplitude * np.exp(-np.power(x - xo, 2.) / (2 * np.power(sigma, 2.)))
    return g.ravel()


# def smooth(x):
#     """
#     modified from: http://wiki.scipy.org/Cookbook/SignalSmooth
#     """
#     window_len = 101
#     s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
#     window = 'hanning'
#     # window='flat'
#     if window == 'flat':  # moving average
#         w = np.ones(window_len, 'd')
#     else:
#         w = eval('np.' + window + '(window_len)')
#     y = np.convolve(w / w.sum(), s, mode='valid')
#     y = y[(window_len / 2):-(window_len / 2)]
#     #y = y*gauss(x,1,550,1000,0)
#     return y

# def smooth(y):
#     x = np.linspace(0,2*np.pi,len(y))
#     w = scipy.fftpack.rfft(y)
#     f = scipy.fftpack.rfftfreq(len(x), x[1]-x[0])
#     spectrum = w**2
#     cutoff_idx = spectrum < (spectrum.max()/5)
#     cutoff_idx[0:50] = False
#     w2 = w.copy()
#     w2[cutoff_idx] = 0
#
#     y2 = scipy.fftpack.irfft(w2)
#     return y2

def smooth(y):
    y = savgol_filter(y, 201, 1, mode='interp')
    #y[900:y.shape[0]] = y[900]
    x = np.linspace(0,len(y)-1,len(y))
    #ind = np.linspace(0,50,dtype=np.int)
    #ind = np.append(ind,np.linspace(980,1023,dtype=np.int))
    #slope, intercept, r_value, p_value, std_err = stats.linregress(x[ind],y[ind])
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    y = y -  (slope*x + intercept)
    y = y - np.min(y)
    y = y*gauss(x,1,500,300,0)
    y = savgol_filter(y, 91, 1, mode='interp')
    #s = interpolate.InterpolatedUnivariateSpline(np.linspace(200,900,len(x)),x)
    #return s(np.linspace(200,900,len(x)))
    return y

class MeasurementThread(QObject):
    specSignal = pyqtSignal(np.ndarray)
    progressSignal = pyqtSignal(float, str)
    finishSignal = pyqtSignal(np.ndarray)

    def __init__(self, spectrometer, parent=None):
        if getattr(self.__class__, '_has_instance', False):
            RuntimeError('Cannot create another instance')
        self.__class__._has_instance = True
        try:
            super(MeasurementThread, self).__init__(parent)
            self.spectrometer = spectrometer
            self.abort = False
            self.thread = QThread()
            self.moveToThread(self.thread)
            self.thread.started.connect(self.process)
            self.thread.finished.connect(self.stop)
        except:
            (type, value, traceback) = sys.exc_info()
            sys.excepthook(type, value, traceback)

    def start(self):
        self.thread.start(QThread.HighPriority)

    @pyqtSlot()
    def stop(self):
        self.abort = True

    def __del__(self):
        self.__class__.has_instance = False
        try:
            self.specSignal.disconnect()
            self.progressSignal.disconnect()
            self.finishSignal.disconnect()
        except TypeError:
            pass
        self.abort = True

    def work(self):
        self.specSignal.emit(self.spec)

    @pyqtSlot()
    def process(self):
        while not self.abort:
            try:
                self.spec = self.spectrometer.intensities()
                self.spec = self.spec[0:1024]
                self.work()
            except:
                (type, value, traceback) = sys.exc_info()
                sys.excepthook(type, value, traceback)


class MeanThread(MeasurementThread):

    def __init__(self, spectrometer, number_of_samples, parent=None):
        self.number_of_samples = number_of_samples
        self.init()
        super(MeanThread, self).__init__(spectrometer)

    def init(self):
        self.progress = progress.Progress(max=self.number_of_samples)
        self.mean = np.zeros(1024, dtype=np.float)
        self.i = 0
        self.abort = False

    def work(self):
        self.mean = (self.mean + self.spec)  # / 2
        self.specSignal.emit(self.mean / (self.i + 1))
        self.progress.next()
        progressFraction = float(self.i + 1) / self.number_of_samples
        self.progressSignal.emit(self.progress.percent, str(self.progress.eta_td))
        self.i += 1
        if self.i >= self.number_of_samples:
            self.abort = True
            self.finishSignal.emit(self.mean / (self.number_of_samples))


class SearchThread(MeasurementThread):
    def __init__(self, spectrometer, settings, stage, parent=None):
        try:
            self.settings = settings
            self.stage = stage
            super(SearchThread, self).__init__(spectrometer)
        except:
            (type, value, traceback) = sys.exc_info()
            sys.excepthook(type, value, traceback)

    @pyqtSlot()
    def process(self):
        while True:
            if self.abort:
                return
            try:
                self.work()
            except:
                (type, value, traceback) = sys.exc_info()
                sys.excepthook(type, value, traceback)

    def work(self):
        self.search()
        x, y, z = self.stage.last_pos()
        #self.specSignal.emit(self.spec)
        self.finishSignal.emit(np.array([x,y]))
        self.abort = True

    def stop(self):
        self.spectrometer.integration_time_micros(self.settings.integration_time * 1000)
        super(SearchThread, self).stop()

    def search(self):
        # self.mutex.lock()
        self.spectrometer.integration_time_micros(self.settings.search_integration_time * 1000)
        # self.mutex.unlock()
        spec = self.spectrometer.intensities()
        spec = spec[0:1024]
        sepc = smooth(spec)

        self.stage.query_pos()
        startpos = self.stage.last_pos()

        minval = np.min(spec)
        maxval = np.max(spec)

        d = np.linspace(-self.settings.rasterwidth, self.settings.rasterwidth, self.settings.rasterdim)

        repetitions = 4
        self.progress = progress.Progress(max=repetitions)
        for j in range(repetitions):
            self.stage.query_pos()
            origin = self.stage.last_pos()
            measured = np.zeros(self.settings.rasterdim)
            if j is 4:
                d /= 2
            if j % 2:
                pos = d + origin[0]
            else:
                pos = d + origin[1]

            for k in range(len(pos)):
                if j % 2:
                    self.stage.moveabs(x=pos[k])
                else:
                    self.stage.moveabs(y=pos[k])
                if self.abort:
                    self.stage.moveabs(x=startpos[0],y=startpos[1])
                    return False
                spec = self.spectrometer.intensities()
                spec = spec[0:1024]
                spec = smooth(spec)
                self.specSignal.emit(spec)
                measured[k] = np.max(spec)
            maxind = np.argmax(measured)

            initial_guess = (maxval - minval, pos[maxind], self.settings.sigma, minval)
            dx = origin[0]
            dy = origin[1]
            popt = None
            fitted = False
            try:
                popt, pcov = opt.curve_fit(gauss, pos[2:(len(pos) - 1)], measured[2:(len(pos) - 1)], p0=initial_guess)
                perr = np.diag(pcov)
                #print(perr)
                if perr[0] > 500 or perr[1] > 1e-1 or perr[2] > 1e-1 :
                    print("Could not determine particle position: Variance too big")
                elif popt[0] < 1e-1:
                    print("Could not determine particle position: Peak too small")
                elif popt[1] < (min(pos)-2) or popt[1] > (max(pos)+2):
                    print("Could not determine particle position: Peak outside bounds")
                else:
                    fitted = True
            except RuntimeError as e:
                print(e)
                print("Could not determine particle position: Fit error")

            if fitted:
                if j % 2:
                    dx = float(popt[1])
                else:
                    dy = float(popt[1])
            else:
                if j % 2:
                    dx = startpos[0]
                else:
                    dy = startpos[1]

            plt.figure()
            plt.plot(pos, measured, 'bo')
            x = np.linspace(min(pos), max(pos))
            if not popt is None:
                plt.plot(x, gauss(x, popt[0], popt[1], popt[2], popt[3]), 'g-')
            plt.savefig("search_max/search" + str(j) + ".png")
            plt.close()
            self.stage.moveabs(x=dx, y=dy)
            self.progress.next()
            self.progressSignal.emit(self.progress.percent, str(self.progress.eta_td))
        self.spectrometer.integration_time_micros(self.settings.integration_time * 1000)
        #self.stage.query_pos()
        # spec = self.getspec()
        # self.specSignal.emit(spec)



class ScanThread(MeasurementThread):
    def __init__(self, spectrometer, settings, scanning_points, stage, parent=None):
        try:
            self.spectrometer = spectrometer
            self.scanning_points = scanning_points
            self.settings = settings
            self.stage = stage
            self.i = 0
            self.n = scanning_points.shape[0]
            self.positions = np.zeros((self.n, 2))
            self.progress = progress.Progress(max=self.n)
            super(ScanThread, self).__init__(spectrometer)
        except:
            (type, value, traceback) = sys.exc_info()
            sys.excepthook(type, value, traceback)

    @pyqtSlot()
    def run(self):
        while True:
            if self.abort:
                return
            try:
                self.work()
            except:
                (type, value, traceback) = sys.exc_info()
                sys.excepthook(type, value, traceback)

    def intermediatework(self):
        pass

    def work(self):
        self.stage.moveabs(x=self.scanning_points[self.i, 0], y=self.scanning_points[self.i, 1])
        self.intermediatework()
        x, y, z = self.stage.last_pos()
        self.positions[self.i, 0] = x
        self.positions[self.i, 1] = y
        progressFraction = float(self.i + 1) / self.n
        self.progress.next()
        self.progressSignal.emit(self.progress.percent, str(self.progress.eta_td))
        self.i += 1
        if self.i >= self.n:
            self.stop()
            plt.plot(self.scanning_points[:, 0], self.scanning_points[:, 1], "r.")
            plt.plot(self.positions[:, 0], self.positions[:, 1], "bx")
            plt.savefig("search_max/grid.png")
            plt.close()
            self.finishSignal.emit(self.positions)
            #self.spec = self.getspec()
            #self.specSignal.emit(self.spec)

class ScanSearchThread(ScanThread):
    def __init__(self, spectrometer, settings, scanning_points, stage, parent=None):
        super(ScanSearchThread, self).__init__(spectrometer,settings,scanning_points,stage)
        self.searchthread = SearchThread(self.spectrometer,self.settings,self.stage,self)
        self.searchthread.specSignal.connect(self.specslot)

    def stop(self):
        super(ScanSearchThread, self).stop()
        self.searchthread.stop()

    def __del__(self):
        self.searchthread.specSignal.disconnect(self.specslot)
        super(ScanSearchThread, self).__del__()

    def intermediatework(self):
        self.searchthread.search()

    @pyqtSlot(np.ndarray)
    def specslot(self, spec):
        self.specSignal.emit(spec)


class ScanMeanThread(ScanThread):
    saveSignal = pyqtSignal(np.ndarray, str, np.ndarray, bool)

    def __init__(self, spectrometer, settings, scanning_points, stage, parent=None):
        super(ScanMeanThread, self).__init__(spectrometer, settings, scanning_points, stage)
        self.meanthread = MeanThread(spectrometer,settings.number_of_samples,self)
        self.meanthread.finishSignal.connect(self.meanfinished)
        self.meanthread.specSignal.connect(self.specslot)

    def stop(self):
        super(ScanMeanThread, self).stop()
        self.meanthread.stop()

    def __del__(self):
        self.meanthread.finishSignal.disconnect(self.meanfinished)
        self.meanthread.specSignal.disconnect(self.specslot)
        self.saveSignal.disconnect()
        super(ScanMeanThread, self).__del__()

    def intermediatework(self):
        self.meanthread.init()
        self.meanthread.process()

    @pyqtSlot(np.ndarray)
    def specslot(self, spec):
        self.specSignal.emit(spec)

    @pyqtSlot(np.ndarray)
    def meanfinished(self, spec):
        self.saveSignal.emit(spec, str(self.i).zfill(5) + ".csv", self.positions[self.i,:], False)

class ScanSearchMeanThread(ScanMeanThread):
    def __init__(self, spectrometer, settings, scanning_points, stage, parent=None):
        super(ScanSearchMeanThread, self).__init__(spectrometer, settings, scanning_points, stage)
        self.searchthread = SearchThread(self.spectrometer,self.settings,self.stage,self)
        self.searchthread.specSignal.connect(self.specslot)

    def stop(self):
        super(ScanMeanThread, self).stop()
        self.searchthread.stop()

    def __del__(self):
        self.searchthread.specSignal.disconnect()
        super(ScanMeanThread, self).__del__()

    def intermediatework(self):
        self.searchthread.search()
        self.meanthread.init()
        self.meanthread.process()

