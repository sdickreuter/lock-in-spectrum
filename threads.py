__author__ = 'sei'

import sys

import numpy as np
import scipy.optimize as opt
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from PyQt5.QtCore import pyqtSlot, QThread, QMutex, QWaitCondition, pyqtSignal, QObject, QTimer
import progress
import os, struct, array
from fcntl import ioctl

# These constants were borrowed from linux/input.h
axis_names = {
    0x00: 'x',
    0x01: 'y',
    0x02: 'z',
    0x03: 'rx',
    0x04: 'ry',
    0x05: 'rz',
    0x06: 'trottle',
    0x07: 'rudder',
    0x08: 'wheel',
    0x09: 'gas',
    0x0a: 'brake',
    0x10: 'hat0x',
    0x11: 'hat0y',
    0x12: 'hat1x',
    0x13: 'hat1y',
    0x14: 'hat2x',
    0x15: 'hat2y',
    0x16: 'hat3x',
    0x17: 'hat3y',
    0x18: 'pressure',
    0x19: 'distance',
    0x1a: 'tilt_x',
    0x1b: 'tilt_y',
    0x1c: 'tool_width',
    0x20: 'volume',
    0x28: 'misc',
}

button_names = {
    0x120: 'X',
    0x121: 'A',
    0x122: 'B',
    0x123: 'Y',
    0x124: 'LB',
    0x125: 'RB',
    0x126: 'LT',
    0x127: 'RT',
    0x128: 'BACK',
    0x129: 'START',
    0x12a: 'base5',
    0x12b: 'base6',
    0x12f: 'dead',
    0x130: 'a',
    0x131: 'b',
    0x132: 'c',
    0x133: 'x',
    0x134: 'y',
    0x135: 'z',
    0x136: 'tl',
    0x137: 'tr',
    0x138: 'tl2',
    0x139: 'tr2',
    0x13a: 'select',
    0x13b: 'start',
    0x13c: 'mode',
    0x13d: 'thumbl',
    0x13e: 'thumbr',

    0x220: 'dpad_up',
    0x221: 'dpad_down',
    0x222: 'dpad_left',
    0x223: 'dpad_right',

    # XBox 360 controller uses these codes.
    0x2c0: 'dpad_left',
    0x2c1: 'dpad_right',
    0x2c2: 'dpad_up',
    0x2c3: 'dpad_down',
}

class GamepadThread(QObject):
    ASignal = pyqtSignal()
    BSignal = pyqtSignal()
    XSignal = pyqtSignal()
    YSignal = pyqtSignal()
    xaxisSignal = pyqtSignal(float)
    yaxisSignal = pyqtSignal(float)

    axis_map = []
    button_map = []

    axis_states = {}
    button_states = {}

    def __init__(self, parent=None):
        if getattr(self.__class__, '_has_instance', False):
            RuntimeError('Cannot create another instance')
        self.__class__._has_instance = True
        self.isinitialized = False
        super(GamepadThread, self).__init__(parent)
        try:
            self._initialize()
            self.isinitialized = True
        except:
            (type, value, traceback) = sys.exc_info()
            sys.excepthook(type, value, traceback)

        if self.isinitialized:
            self.abort = False
            self.thread = QThread()
            self.thread.started.connect(self.process)
            self.thread.finished.connect(self.stop)
            self.moveToThread(self.thread)

    def _initialize(self):
        # Open the joystick device.
        fn = '/dev/input/js0'
        print('Opening %s...' % fn)
        self.jsdev = open(fn, 'rb')

        # Get the device name.
        # buf = bytearray(63)
        buf = array.array('b', [0] * 64)
        ioctl(self.jsdev, 0x80006a13 + (0x10000 * len(buf)), buf)  # JSIOCGNAME(len)
        buf = bytes(buf)
        js_name = buf.decode()
        js_name = js_name.strip(b'\x00'.decode())
        print('%s initialized' % js_name)

        # Get number of axes and buttons.
        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a11, buf)  # JSIOCGAXES
        self.num_axes = buf[0]

        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a12, buf)  # JSIOCGBUTTONS
        self.num_buttons = buf[0]

        # Get the axis map.
        buf = array.array('B', [0] * 0x40)
        ioctl(self.jsdev, 0x80406a32, buf)  # JSIOCGAXMAP

        for axis in buf[:self.num_axes]:
            axis_name = axis_names.get(axis, 'unknown(0x%02x)' % axis)
            self.axis_map.append(axis_name)
            self.axis_states[axis_name] = 0.0

        # Get the button map.
        buf = array.array('H', [0] * 200)
        ioctl(self.jsdev, 0x80406a34, buf)  # JSIOCGBTNMAP

        for btn in buf[:self.num_buttons]:
            btn_name = button_names.get(btn, 'unknown(0x%03x)' % btn)
            self.button_map.append(btn_name)
            self.button_states[btn_name] = 0


    def start(self):
        self.thread.start(QThread.HighPriority)

    @pyqtSlot()
    def stop(self):
        self.abort = True

    def work(self):
        evbuf = self.jsdev.read(8)
        if evbuf:
            time, value, type, number = struct.unpack('IhBB', evbuf)

            if not type & 0x80:
                if type & 0x01:
                    button = self.button_map[number]
                    if button:
                        self.button_states[button] = value
                        if value == 0:
                            if button is 'A':
                                self.ASignal.emit()
                            if button is 'B':
                                self.BSignal.emit()
                            if button is 'X':
                                self.XSignal.emit()
                            if button is 'Y':
                                self.YSignal.emit()

                if type & 0x02:
                    axis = self.axis_map[number]
                    if axis:
                        self.axis_states[axis] = value
                        fvalue = value / 32767.0
                        if axis is 'x':
                            self.xaxisSignal.emit(fvalue)
                        if axis is 'y':
                            self.yaxisSignal.emit(fvalue)

    @pyqtSlot()
    def process(self):
        while not self.abort:
            try:
                self.work()
            except:
                (type, value, traceback) = sys.exc_info()
                sys.excepthook(type, value, traceback)

    def __del__(self):
        self.__class__.has_instance = False
        try:
            self.ASignal.disconnect()
            self.BSignal.disconnect()
            self.XSignal.disconnect()
            self.YSignal.disconnect()
            self.xaxisSignal.disconnect()
            self.yaxisSignal.disconnect()
        except TypeError as e:
            print(e)
            (type, value, traceback) = sys.exc_info()
            sys.excepthook(type, value, traceback)
        self.abort = True





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

def smooth(x,y):
    buf = y.copy()
    buf = savgol_filter(buf, 137, 1, mode='interp')
    buf = savgol_filter(buf, 137, 1, mode='interp')
    #y = savgol_filter(y, 31, 1, mode='interp')
    #y[900:y.shape[0]] = y[900]
    #ind = np.linspace(0,19,20,dtype=np.int)
    #ind = np.append(ind,np.linspace(1003,1023,20,dtype=np.int))
    #slope, intercept, r_value, p_value, std_err = stats.linregress(x[ind],y[ind])
    #print((slope, intercept))
    #slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    #y = y - (-slope*x+intercept)
    #slope = (buf[1023] - buf[200]) / (x[1023] -x[200])
    #intercept = buf[200] - slope*x[200]
    #res = buf - (slope*x  + intercept)

    ind1 = np.linspace(300,319,20,dtype=np.int)
    ind2 = np.linspace(1003,1023,10,dtype=np.int)
    slope = (np.mean(buf[ind2]) - np.mean(buf[ind1])) / (np.mean(x[ind2]) - np.mean(x[ind1]))
    intercept = np.mean(buf[ind1]) - slope*np.mean(x[ind1])
    #print((np.mean(y[ind1]),np.mean(x[ind1])))
    #print((slope, intercept))
    l = slope*x + intercept
    buf = np.subtract(buf,l)
    #slope = -(np.mean(y[ind2]) - np.mean(y[ind1])) / (np.mean(x[ind2]) - np.mean(x[ind1]))
    #intercept = np.mean(y[ind1]) - slope*np.mean(x[ind1])
    #l = slope*x + intercept
    #y = l # np.subtract(y,l)
    #y = y - np.min(y)
    buf = buf*gauss(x,1,600,500,0)
    #s = interpolate.InterpolatedUnivariateSpline(np.linspace(200,900,len(x)),x)
    #return s(np.linspace(200,900,len(x)))
    return buf

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
            self.thread = QThread(parent)
            self.moveToThread(self.thread)
            self.thread.started.connect(self.process)
            self.thread.finished.connect(self.stop)
        except:
            (type, value, traceback) = sys.exc_info()
            print(type)
            print(value)
            print(traceback)
            sys.excepthook(type, value, traceback)

    def start(self):
        self.thread.start(QThread.HighPriority)

    @pyqtSlot()
    def stop(self):
        self.abort = True
        self.thread.quit()
        self.thread.wait(5000)


    def __del__(self):
        self.__class__.has_instance = False
        try:
            self.specSignal.disconnect()
            self.progressSignal.disconnect()
            self.finishSignal.disconnect()
        except:
            (type, value, traceback) = sys.exc_info()
            print(type)
            print(value)
            print(traceback)
            sys.excepthook(type, value, traceback)

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
                print(type)
                print(value)
                print(traceback)
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

class LockinThread(MeasurementThread):

    def __init__(self, spectrometer, settings, stage, parent=None):
        self.number_of_samples = settings.number_of_samples
        self.stage = stage
        self.settings = settings
        self.init()
        super(MeanThread, self).__init__(spectrometer)

    def init(self):
        self.progress = progress.Progress(max=self.number_of_samples)
        self.lockin = np.zeros((1024,self.number_of_samples), dtype=np.float)
        self.i = 0
        self.stage.query_pos()
        self.startpos = self.stage.last_pos()
        self.abort = False

    def move_stage(self, dist):
        x = self.startpos[0] + self.settings.amplitude / 2 * dist * self.settings.direction_x
        y = self.startpos[1] + self.settings.amplitude / 2 * dist * self.settings.direction_y
        z = self.startpos[2] + self.settings.amplitude / 2 * dist * self.settings.direction_z
        # print "X: {0:+8.4f} | Y: {1:8.4f} | Z: {2:8.4f} || X: {3:+8.4f} | Y: {4:8.4f} | Z: {5:8.4f}".format(x,y,z,self._startx,self._starty,self._startz)
        self.stage.moveabs(x=x, y=y, z=z)

    def calc_lockin(self):
        res = np.zeros(1024)
        for i in range(1024):
            ref = np.cos(2 * np.pi * np.arange(0,self.number_of_samples) * self.settings.f)
            buf = ref * self.lockin[:, 1]
            buf = np.sum(buf)
            res[i] = buf
        return res

    def work(self):

        ref = np.cos(2 * np.pi * self.i * self.settings.f)
        self.move_stage(ref / 2)
        spec = self._spectrometer.intensities(correct_nonlinearity=True)
        self.lockin[:,self.i] = spec[0:1024]

        self.specSignal.emit(self.lockin[:,self.i])
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
            self.wl = self.spectrometer.wavelengths()
            self.wl = self.wl[0:1024]
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
        self.spectrometer.intensities()
        self.finishSignal.emit(np.array([x,y]))
        self.abort = True

    def stop(self):
        #self.spectrometer.integration_time_micros(self.settings.integration_time * 1000)
        super(SearchThread, self).stop()

    def search(self):
        # self.mutex.lock()
        self.spectrometer.integration_time_micros(self.settings.search_integration_time * 1000)
        # self.mutex.unlock()
        spec = self.spectrometer.intensities()
        spec = spec[0:1024]
        spec = smooth(self.wl,spec)

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
                spec = smooth(self.wl,spec)
                self.specSignal.emit(spec)
                #initial_guess = (np.max(spec), 600, 200, 0)
                #try:
                #    #def gauss(x, amplitude, xo, fwhm, offset):
                #    popt, pcov = opt.curve_fit(gauss, np.linspace(0,1023,1024), spec, p0=initial_guess)
                #    #print(popt)
                #    measured[k] = popt[0]
                #except RuntimeError as e:
                #    print(e)
                measured[k] = np.max(spec[400:800])

            maxind = np.argmax(measured[2:(len(pos))])

            initial_guess = (maxval - minval, pos[maxind], self.settings.sigma, minval)
            dx = origin[0]
            dy = origin[1]
            popt = None
            fitted = False
            try:
                popt, pcov = opt.curve_fit(gauss, pos[2:(len(pos))], measured[2:(len(pos))], p0=initial_guess)
                #popt, pcov = opt.curve_fit(gauss, pos, measured, p0=initial_guess)
                perr = np.diag(pcov)
                #print(perr)
                if perr[0] > 10000 or perr[1] > 1 or perr[2] > 1 :
                    print("Could not determine particle position: Variance too big")
                elif popt[0] < 1e-1:
                    print("Could not determine particle position: Peak too small")
                elif popt[1] < (min(pos)-0.5) or popt[1] > (max(pos)+0.5):
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

            self.stage.moveabs(x=dx, y=dy)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(pos, measured, 'bo')
            x = np.linspace(min(pos), max(pos))
            if not popt is None:
                ax.text(0.1, 0.9, str(popt[1])+' +- '+str(perr[1]) , ha='left', va='center', transform=ax.transAxes)
                ax.plot(x, gauss(x, popt[0], popt[1], popt[2], popt[3]), 'g-')
            plt.savefig("search_max/search" + str(j) + ".png")
            plt.close()
            self.progress.next()
            self.progressSignal.emit(self.progress.percent, str(self.progress.eta_td))
        self.spectrometer.integration_time_micros(self.settings.integration_time * 1000)
        self.spectrometer.intensities()
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
            #plt.plot(self.scanning_points[:, 0], self.scanning_points[:, 1], "r.")
            #plt.plot(self.positions[:, 0], self.positions[:, 1], "bx")
            #plt.savefig("search_max/grid.png")
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
        self.searchthread.stop()
        super(ScanSearchThread, self).stop()

    def __del__(self):
        self.searchthread.specSignal.disconnect(self.specslot)
        super(ScanSearchThread, self).__del__()

    def intermediatework(self):
        self.searchthread.search()

    @pyqtSlot(np.ndarray)
    def specslot(self, spec):
        self.specSignal.emit(spec)


class ScanLockinThread(ScanThread):
    saveSignal = pyqtSignal(np.ndarray, str, np.ndarray, bool, bool)

    def __init__(self, spectrometer, settings, scanning_points, stage, parent=None):
        super(ScanMeanThread, self).__init__(spectrometer, settings, scanning_points, stage)
        #__init__(self, spectrometer, settings, stage, parent=None)
        self.meanthread = LockinThread(spectrometer,settings,stage,self)
        self.meanthread.finishSignal.connect(self.lockinfinished)
        self.meanthread.specSignal.connect(self.specslot)

    def stop(self):
        self.meanthread.stop()
        super(ScanMeanThread, self).stop()

    def __del__(self):
        self.meanthread.finishSignal.disconnect(self.lockinfinished)
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
    def lockinfinished(self, spec):
        self.saveSignal.emit(self.lockin, str(self.i).zfill(5) + "_lockin.csv", self.positions[self.i,:], False,False)


class ScanMeanThread(ScanThread):
    saveSignal = pyqtSignal(np.ndarray, str, np.ndarray, bool, bool)

    def __init__(self, spectrometer, settings, scanning_points, stage, parent=None):
        super(ScanMeanThread, self).__init__(spectrometer, settings, scanning_points, stage)
        self.meanthread = MeanThread(spectrometer,settings.number_of_samples,self)
        self.meanthread.finishSignal.connect(self.meanfinished)
        self.meanthread.specSignal.connect(self.specslot)

    def stop(self):
        self.meanthread.stop()
        super(ScanMeanThread, self).stop()

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
        self.saveSignal.emit(spec, str(self.i).zfill(5) + ".csv", self.positions[self.i,:], False,False)


class ScanSearchMeanThread(ScanMeanThread):
    def __init__(self, spectrometer, settings, scanning_points, stage, parent=None):
        super(ScanSearchMeanThread, self).__init__(spectrometer, settings, scanning_points, stage)
        self.searchthread = SearchThread(self.spectrometer,self.settings,self.stage,self)
        self.searchthread.specSignal.connect(self.specslot)

    def stop(self):
        self.searchthread.stop()
        super(ScanMeanThread, self).stop()

    def __del__(self):
        self.searchthread.specSignal.disconnect()
        super(ScanMeanThread, self).__del__()

    def intermediatework(self):
        self.searchthread.search()
        self.meanthread.init()
        self.meanthread.process()

