__author__ = 'sei'

import sys

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from PyQt5.QtCore import pyqtSlot, QThread, QMutex, QWaitCondition, pyqtSignal


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


# def _millis(starttime):
#    dt = datetime.now() - starttime
#    ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
#    return ms


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
        self.stop()

    def getSpectrum(self):
        #    #locker = QMutexLocker(self.mutex)
        if not self.isRunning():
            self.start(QThread.HighPriority)

    def run(self):
        self.mutex.lock()
        spec = self.spectrometer.intensities()
        self.mutex.unlock()
        self.dynamicSpecSignal.emit(spec)

    def stop(self):
        self.mutex.lock()
        self.abort = True
        self.condition.wakeOne()
        self.mutex.unlock()
        self.wait()


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
        self.spec = None

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
        self.abort = True

    def work(self):
        self.specSignal.emit(self.spec)

    def run(self):
        while True:
            if self.abort:
                return
            self.getspecthread.getSpectrum()
            try:
                self.work()
            except:
                (type, value, traceback) = sys.exc_info()
                sys.excepthook(type, value, traceback)

    @pyqtSlot(np.ndarray)
    def specCallback(self, spec):
        self.spec = spec

class LiveThread(MeasurementThread):
    def __init__(self, getspecthread, parent=None):
        super(LiveThread, self).__init__(getspecthread, parent)

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


class MeanThread(LiveThread):
    def __init__(self, getspecthread, number_of_samples, parent=None):
        self.number_of_samples = number_of_samples
        self.mean = np.zeros(1024, dtype=np.float)
        self.i = 0
        super(MeanThread, self).__init__(getspecthread)

    def work(self):
        self.mutex.lock()
        self.mean = (self.mean + self.spec)  # / 2
        self.mutex.unlock()
        self.specSignal.emit(self.mean / (self.i + 1))
        progressFraction = float(self.i + 1) / self.number_of_samples
        self.progressSignal.emit(progressFraction * 100)
        self.mutex.lock()
        self.i += 1
        if self.i >= self.number_of_samples:
            self.abort = True
            self.finishSignal.emit(self.mean / (self.number_of_samples))
        self.mutex.unlock()


class SearchThread(LiveThread):
    def __init__(self, getspecthread, settings, stage, parent=None):
        try:
            self.settings = settings
            self.stage = stage
            super(SearchThread, self).__init__(getspecthread)
            self.spectrometer = self.getspecthread.spectrometer
        except:
            (type, value, traceback) = sys.exc_info()
            sys.excepthook(type, value, traceback)

    def run(self):
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
        self.mutex.lock()
        x, y, z = self.stage.last_pos()
        self.finishSignal.emit(np.array([x,y]))
        self.specSignal.emit(self.spec)
        self.mutex.unlock()
        self.stop()

    def getspec(self):
        self.mutex.lock()
        self.getspecthread.getSpectrum()
        self.waitCondition.wait(self.mutex)
        self.mutex.unlock()
        return self.spec

    def search(self):
        # self.mutex.lock()
        # self.spectrometer.integration_time_micros(self.settings.search_integration_time * 1000)
        # self.mutex.unlock()
        spec = smooth(self.getspec())

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
                if self.abort:
                    self.stage.moveabs(x=origin[0],y=origin[1])
                    return False
                self.mutex.unlock()
                spec = smooth(self.getspec())
                self.specSignal.emit(spec)
                measured[k] = np.max(spec)
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
                    dx = float(popt[1])
                else:
                    dy = float(popt[1])
                    # print(popt)
            self.mutex.lock()
            plt.figure()
            plt.plot(pos, measured, 'bo')
            x = np.linspace(min(pos), max(pos))
            plt.plot(x, gauss(x, popt[0], popt[1], popt[2], popt[3]), 'g-')
            plt.savefig("search_max/search" + str(j) + ".png")
            plt.close()
            self.stage.moveabs(x=dx, y=dy)
            self.mutex.unlock()
        self.mutex.lock()
        # self._spectrometer.integration_time_micros(self.settings.integration_time / 1000)
        self.stage.query_pos()
        self.mutex.unlock()
        # spec = self.getspec()
        # self.specSignal.emit(spec)



class ScanSearchThread(MeasurementThread):
    def __init__(self, getspecthread, settings, scanning_points, stage, parent=None):
        try:
            self.scanning_points = scanning_points
            self.settings = settings
            self.stage = stage
            self.i = 0
            self.n = scanning_points.shape[0]
            self.positions = np.zeros((self.n, 2))
            super(ScanSearchThread, self).__init__(getspecthread)
        except:
            (type, value, traceback) = sys.exc_info()
            sys.excepthook(type, value, traceback)

    def run(self):
        while True:
            if self.abort:
                return
            try:
                self.work()
            except:
                (type, value, traceback) = sys.exc_info()
                sys.excepthook(type, value, traceback)

    def work(self):
        self.mutex.lock()
        self.stage.moveabs(x=self.scanning_points[self.i, 0], y=self.scanning_points[self.i, 1])
        self.mutex.unlock()
        self.searchthread = SearchThread(self.getspecthread,self.settings,self.stage,self)
        self.searchthread.finishSignal.connect(self.searchfinished)
        self.searchthread.specSignal.connect(self.specslot)
        self.searchthread.search()
        self.mutex.lock()
        print('search started')
        self.waitCondition.wait(self.mutex)
        print('done waiting')
        self.searchthread = None
        x, y, z = self.stage.last_pos()
        self.positions[self.i, 0] = x
        self.positions[self.i, 1] = y
        progressFraction = float(self.i + 1) / self.n
        self.progressSignal.emit(progressFraction * 100)
        self.i += 1
        if self.i >= self.n:
            self.abort = True
            plt.plot(self.positions[:, 0], self.positions[:, 1], "bx")
            plt.plot(self.scanning_points[:, 0], self.scanning_points[:, 1], "r.")
            plt.savefig("search_max/grid.png")
            plt.close()
            self.finishSignal.emit(self.positions)
            #self.spec = self.getspec()
            #self.specSignal.emit(self.spec)
        self.mutex.unlock()

    @pyqtSlot(np.ndarray)
    def searchfinished(self, spec):
        self.waitCondition.wakeOne()

    @pyqtSlot(np.ndarray)
    def specslot(self, spec):
        self.specSignal.emit(spec)

    @pyqtSlot(np.ndarray)
    def specCallback(self, spec):
        self.spec = spec


class SearchScanMeanThread(SearchThread):
    def __init__(self, getspecthread, settings, scanning_points, stage, parent=None):
        super(SearchScanMeanThread, self).__init__(getspecthread, settings, scanning_points, stage, parent)

    def work(self):
        super(SearchScanMeanThread, self).work()
        self.meanthread = MeanThread(self.getspecthread,self.settings.number_of_samples,self)
        self.meanthread.specSignal.connect(self.specCallback)
        self.meanthread.progressSignal.connect(self.progressCallback)

    def stop(self):
        super(SearchScanMeanThread, self).stop()
        self.meanthread.stop()
