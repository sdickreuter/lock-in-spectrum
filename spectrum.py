__author__ = 'sei'

from datetime import datetime
import math
import PIStage
import oceanoptics
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import multiprocessing
import pandas
from scipy.interpolate import griddata

class Spectrum(object):
    def __init__(self, stage, settings, status, progress, enable_buttons, disable_buttons):
        self.settings = settings
        self.stage = stage
        self.status = status
        self.progress = progress
        self.enable_buttons = enable_buttons
        self.disable_buttons = disable_buttons
        self._init_spectrometer()
        self._cycle_time_start = 250
        self._startx = 10
        self._starty = 10
        self._startz = 10
        self._starttime = None
        self._juststarted = True
        self._spectrum_ready = False
        self._data = np.ones((self.settings.number_of_samples, 1026), dtype=np.float)

        # variables for storing the spectra
        self.lamp = None
        self.dark = None
        self.normal = None
        self.lockin = None

        self._progress_fraction = 0.0

        self.worker = None
        self.worker_mode = None
        self.conn_for_main, self._conn_for_worker = multiprocessing.Pipe()
        self.running = multiprocessing.Event()
        self.running.clear()
        self._spec = self._spectrometer.intensities()


    def _init_spectrometer(self):
        try:
            #self._spectrometer = oceanoptics.QE65000()
            self._spectrometer = oceanoptics.ParticleDummy(stage=self.stage)
            #self._spectrometer = oceanoptics.ParticleDummy(stage=self.stage,particles = [[10, 10], [11, 10],[12, 10],[14, 10],[11, 14],[11, 12],[14, 13],[15, 15]])
            self._spectrometer.integration_time(self.settings.integration_time/1000)
            #print(self._spectrometer._query_status())
            sp = self._spectrometer.spectrum()
            self._wl = np.array(sp[0], dtype=np.float)
            print("Spectrometer initialized and working")
        except:
            raise RuntimeError("Error opening spectrometer. Exiting...")

    def start_process(self, target):
        self.worker = multiprocessing.Process(target=target, args=(self._conn_for_worker,))
        self.worker.daemon = True
        self.running.set()
        self.worker.start()

    def stop_process(self):
        self.running.clear()

    def _millis(self):
        dt = datetime.now() - self._starttime
        ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        return ms

    def get_wl(self):
        return self._wl

    def get_spec(self):
        return self._spec

    def reset(self):
        self._spectrometer.integration_time(self.settings.integration_time/1000)
        self._spectrum_ready = False
        self._juststarted = True

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
            if not self.dark is None:
                buf = buf - self.dark[i]
                if not self.lamp is None:
                    buf = buf / (self.lamp[i])
            buf = buf * self._data[:, 1]
            buf = np.sum(buf)
            res[i] = buf
        return res

    def take_live(self):
        self.worker = "live"
        self.start_process(self._live_spectrum)

    def take_dark(self):
        self.worker_mode = "dark"
        self.start_process(self._mean_spectrum)

    def take_lamp(self):
        self.worker_mode = "lamp"
        self.start_process(self._mean_spectrum)

    def take_normal(self):
        self.worker_mode = "normal"
        self.start_process(self._mean_spectrum)

    def take_lockin(self):
        self.worker_mode = "lockin"
        self.reset()
        self.lockin = None
        self.start_process(self._lockin_spectrum)

    def search_max(self):
        self.worker_mode = "search"
        self.start_process(self._search_max_int)

    def make_scan(self, points, path, search = False, lockin = False):
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
        self._callback_scan()

    def callback(self, io, condition):
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
        else:
            finished, self._progress_fraction, spec = self.conn_for_main.recv()

        self.progress.set_fraction(self._progress_fraction)

        if spec is not None:
            self._spec = spec

        if finished:
            self.running.clear()

            if self.worker_mode is "lockin":
                self.lockin = self.calc_lockin()
                self._spec = self.lockin
                self.status.set_label('Lock-In Spectrum acquired')
            elif self.worker_mode is "lamp":
                self.lamp = self._spec
                self.status.set_label('Lamp Spectrum taken')
            elif self.worker_mode is "dark":
                self.dark = self._spec
                self.status.set_label('Dark Spectrum taken')
            elif self.worker_mode is "normal":
                self.normal = self._spec
                self.status.set_label('Normal Spectrum taken')
            elif self.worker_mode is "search":
                # self.show_pos()
                self.status.set_text("Max. approached")

        if not self.running.is_set():
            self.worker.join(0.5)
            self.enable_buttons()
            self.worker_mode = None

        return True

    def _callback_scan(self):

        def start():
            self.scanner_point = self.scanner_points[self.scanner_index]
            self.stage.moveabs(x=self.scanner_point[0], y=self.scanner_point[1])
            self.reset()
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
            self.status.set_label('Scanning done')
            plt.figure()
            area = self.map - np.min(self.map)
            area = area/np.max(self.map)
            area = (3000 * area) + 10
            plt.scatter(self.x, self.y, c=self.peakpos, s=area, cmap=plt.cm.jet, edgecolors='None', alpha=0.75)
            plt.ylabel('Y [um]')
            plt.xlabel('X [um]')
            plt.axis('equal')
            bar = plt.colorbar()
            bar.set_label('Peak Wavelength [nm]', rotation=270)
            plt.savefig(self.scanner_path+"scanning_map.png")
            plt.close()
            self.status.set_text("Scan complete")

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
                filename = self.scanner_path + 'lockin_' + 'x_{0:3.2f}um_y_{1:3.2f}um'.format( self.scanner_point[0], self.scanner_point[1]) + '.csv'
                data = np.append(np.round(self._wl, 1).reshape(self._wl.shape[0], 1), self.lockin.reshape(self.lockin.shape[0], 1), 1)
                data = pandas.DataFrame(data, columns=('wavelength', 'intensity'))
                data.to_csv(filename, header=True, index=False)
                self.scanner_index += 1
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
                self._spec = self.normal
                smooth = self.smooth(self._spec)
                maxind = np.argmax(smooth)
                self.map.append(smooth[maxind])
                self.peakpos.append(self._wl[maxind])
                self.x.append(self.scanner_point[0])
                self.y.append(self.scanner_point[1])
                filename = self.scanner_path + 'mean_' + 'x_{0:3.2f}um_y_{1:3.2f}um'.format( self.scanner_point[0], self.scanner_point[1]) + '.csv'
                data = np.append(np.round(self._wl, 1).reshape(self._wl.shape[0], 1), self.normal.reshape(self.normal.shape[0], 1), 1)
                data = pandas.DataFrame(data, columns=('wavelength', 'intensity'))
                data.to_csv(filename, header=True, index=False)
                self.scanner_index += 1
                if self.scanner_index >= len(self.scanner_points):
                    finish()
                else:
                    start()

        if not self.running.is_set():
            self.worker.join(0.5)
            self.enable_buttons()
            self.worker_mode = None
            self.scanner_mode = None

        self.progress.set_fraction((self.scanner_index+1)/len(self.scanner_points))

        return True


    def _lockin_spectrum(self, connection):
        self._cycle_factor = -1.0 / (
            7.0 * self.settings.number_of_samples / 1000)  # cycle time is calculated using this factor
        self._spectrum_ready = False
        pos = self.stage.query_pos()
        self._startx = pos[0]
        self._starty = pos[1]
        self._startz = pos[2]
        self._starttime = datetime.now()

        for i in range(self.settings.number_of_samples):
            self._cycle_time = self._cycle_factor * i + self._cycle_time_start
            ref = math.cos(2 * math.pi * i / self._cycle_time)
            self.move_stage((-ref + 1) / 2)
            spec = self._spectrometer.intensities()
            progress_fraction = float(i + 1) / self.settings.number_of_samples
            connection.send([False, progress_fraction, spec, ref, i])
            if not self.running.is_set():
                return True

        print("%s spectra aquired" % (i + 1))
        print("time taken: %s s" % (self._millis() / 1000))
        self.stage.moveabs(x=self._startx, y=self._starty, z=self._startz)
        self._spectrum_ready = True
        connection.send([True, 1., spec, ref, self.settings.number_of_samples - 1])
        return True

    def _mean_spectrum(self, connection):
        spec = self._spectrometer.intensities()
        for i in range(self.settings.number_of_samples - 1):
            spec = (spec + self._spectrometer.intensities())  # / 2
            progress_fraction = float(i + 1) / self.settings.number_of_samples
            connection.send([False, progress_fraction, spec / i])
            if not self.running.is_set():
                return True
        connection.send([True, 1., spec / i])
        return True

    def _live_spectrum(self, connection):
        while self.running.is_set():
            spec = self._spectrometer.intensities()
            if not self.dark is None:
                spec = spec - self.dark
                if not self.lamp is None:
                    spec = spec / self.lamp
            connection.send([False, 0., spec])
        return True

    def _search_max_int(self, connection):

        def update_connection(progress):
            connection.send([False, progress, None])
            if not self.running.is_set():
                return True

        # use position of stage as origin
        #origin = self.stage.query_pos()
        self.stage.query_pos()
        origin = self.stage.last_pos()

        spec = self.smooth(self._spectrometer.intensities())
        min = np.min(spec)
        max = np.max(spec)

        d = np.linspace(-self.settings.rasterwidth, self.settings.rasterwidth, self.settings.rasterdim)

        self.stage.query_pos()
        origin = self.stage.last_pos()

        pos = d+origin[0]

        repetitions = 6

        for j in range(repetitions):
            measured = np.zeros(self.settings.rasterdim)
            for i in range(len(pos)):
                if j%2:
                    self.stage.moveabs(x=origin[0]+d[i])
                else:
                    self.stage.moveabs(y=origin[0]+d[i])
                spec = self.smooth(self._spectrometer.intensities())
                measured[i] = np.max(spec)
            maxind = np.argmax(measured)

            initial_guess = (max - min, pos[maxind], self.settings.sigma, min)

            update_connection(repetitions/i)

            popt = None
            try:
                popt, pcov = opt.curve_fit(self.gauss, pos, measured, p0=initial_guess)
                if popt[0] < 20:
                    RuntimeError("Peak is to small")
            except RuntimeError as e:
                print(e)
                print("Could not determine particle position")
                self.stage.moveabs(x=origin[0],y=origin[1])
                return True
            else:
                if j%2:
                    self.stage.moveabs(x=float(popt[1]))
                else:
                    self.stage.moveabs(y=float(popt[1]))
            print(popt)

        connection.send([True, 1.0, None])
        return True

    def save_data(self, prefix):
        filename = self._gen_filename()
        cols = ('t', 'ref') + tuple(map(str, np.round(self._wl, 1)))
        data = pandas.DataFrame(self._data, columns=cols)
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

    @staticmethod
    def _gen_filename():
        return str(datetime.now().year) + str(datetime.now().month).zfill(2) \
               + str(datetime.now().day).zfill(2) + '_' + str(datetime.now().hour).zfill(2) + \
               str(datetime.now().minute).zfill(2) + str(datetime.now().second).zfill(2) + '.csv'

    # modified from: http://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m#comment33999040_21566831
    @staticmethod
    def gauss2D(pos, amplitude, xo, yo, fwhm, offset):
        sigma = fwhm / 2.3548
        xo = float(xo)
        yo = float(yo)
        g = offset + amplitude * np.exp(
            -( np.power(pos[0] - xo, 2.) + np.power(pos[1] - yo, 2.) ) / (2 * np.power(sigma, 2.)))
        return g.ravel()

    @staticmethod
    def gauss(x, amplitude, xo, fwhm, offset):
        sigma = fwhm / 2.3548
        xo = float(xo)
        g = offset + amplitude * np.exp( -np.power(x - xo, 2.) / (2 * np.power(sigma, 2.)))
        return g.ravel()


    @staticmethod
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
