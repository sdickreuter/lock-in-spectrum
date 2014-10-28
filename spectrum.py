__author__ = 'sei'

from datetime import datetime
import math
import PIStage
import oceanoptics
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import billiard
import pandas

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

        self.worker = None
        self.worker_mode = None
        self.conn_for_main, self._conn_for_worker = billiard.Pipe()
        self.running = False
        self._spec = self._spectrometer.intensities()


    def _init_spectrometer(self):
        try:
            # self._spectrometer = oceanoptics.QE65000()
            self._spectrometer = oceanoptics.ParticleDummy(stage=self.stage)
            self._spectrometer.integration_time(self.settings.integration_time*1000)
            sp = self._spectrometer.spectrum()
            self._wl = np.array(sp[0], dtype=np.float)
        except:
            raise RuntimeError("Error opening spectrometer. Exiting...")

    def start_process(self, target):
        self.worker = billiard.Process(target=target,args=(self._conn_for_worker,))
        self.running = True
        self.worker.daemon = True
        self.worker.start()

    def stop_process(self):
        self.running = False

    def _millis(self):
        dt = datetime.now() - self._starttime
        ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        return ms

    def get_wl(self):
        return self._wl

    def get_spec(self):
        return self._spec

    def reset(self):
        self._spectrometer.integration_time(self.settings.integration_time*1000)
        self._spectrum_ready = False
        self._juststarted = True

    def move_stage(self, dist):
        x = self._startx + self.settings.amplitude / 2 * dist * self.settings.direction_x
        y = self._starty + self.settings.amplitude / 2 * dist * self.settings.direction_y
        z = self._startz + self.settings.amplitude / 2 * dist * self.settings.direction_z
        # print "X: {0:+8.4f} | Y: {1:8.4f} | Z: {2:8.4f} || X: {3:+8.4f} | Y: {4:8.4f} | Z: {5:8.4f}".format(x,y,z,self._startx,self._starty,self._startz)
        self.stage.moveabs(x=x, y=y, z=z)

    def get_spec(self):
        return self._spec

    def calc_lockin(self):
        res = np.empty(1024)
        for i in range(1024):
            buf = self._data[:, i+2]
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

    def callback(self, io, condition):

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

        if not finished:
            self.conn_for_main.send(self.running)

        if not self.running:
            self.status.set_label('Stopped')
            self.worker.join(0.5)
            self.enable_buttons()
            self.worker_mode = None

        if finished :
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
                #self.show_pos()
                self.status.set_text("Max. approached")

            self.worker.join(0.5)
            self.enable_buttons()
            self.worker_mode = None

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
            connection.send([False,progress_fraction,spec,ref,i])
            running = connection.recv()
            if not running:
                return True

        print("%s spectra aquired" % (i+1))
        print("time taken: %s s" % (self._millis() / 1000))
        self.stage.moveabs(self._startx, self._starty, self._startz)
        self._spectrum_ready = True
        connection.send([True,1.,spec,ref,self.settings.number_of_samples-1])
        return True

    def _mean_spectrum(self, connection):
        spec = self._spectrometer.intensities()
        for i in range(self.settings.number_of_samples - 1):
            spec = (spec + self._spectrometer.intensities())# / 2
            progress_fraction = float(i + 1) / self.settings.number_of_samples
            connection.send([False,progress_fraction,spec/i])
            running = connection.recv()
            if not running:
                return True
        connection.send([True,1.,spec/i])
        return True

    def _live_spectrum(self, connection):
        running = True
        while running:
            spec = self._spectrometer.intensities()
            if not self.dark is None:
                spec = spec - self.dark
                if not self.lamp is None:
                    spec = spec / self.lamp
            connection.send([False,0.,spec])
            running = connection.recv()
        return True


    def _search_max_int(self, connection):

        def update_connection(progress):
            connection.send([False,progress,None])
            running = connection.recv()
            if not running:
                return True

        # use position of stage as origin
        origin = self.stage.query_pos()
        # round origin position to 1um, so that grid will be regular for all possible origins
        x_or = round(origin[0])
        y_or = round(origin[1])
        update_connection(0.1)
        # make scanning raster
        x = np.linspace(-self.settings.rasterwidth, self.settings.rasterwidth, self.settings.rasterdim)
        y = np.linspace(-self.settings.rasterwidth, self.settings.rasterwidth, self.settings.rasterdim)
        # add origin to raster to get absolute positions
        x += x_or
        y += y_or

        int = np.zeros((len(x), len(y)))  # matrix for saving the scanned maximum intensities

        # take spectra and get min and max values for use as values for the inital guess
        spec = self.smooth(self._spectrometer.intensities())
        min = np.min(spec)
        max = np.max(spec)

        # iterate through the raster, take spectrum and save maximum value of smoothed spectrum to int
        for xi in range(len(x)):
            for yi in range(len(y)):
                self.stage.moveabs(x[xi], y[yi])
                int[xi, yi] = np.max(self.smooth(self._spectrometer.intensities()))

        update_connection(0.3)

        # find max value of int and use this as the inital value for the position
        maxind = np.argmax(int)
        maxind = np.unravel_index(maxind, int.shape)

        int = int.ravel()

        initial_guess = (max - min, x[maxind[1]], y[maxind[0]], self.settings.sigma, min)
        x, y = np.meshgrid(x, y)
        positions = np.vstack((x.ravel(), y.ravel()))

        popt = None
        try:
            popt, pcov = opt.curve_fit(self.gauss2D, positions, int, p0=initial_guess)
            #print popt
            if popt[0] < 20:
                RuntimeError("Peak is to small")
        except RuntimeError as e:
            print(e)
            print("Could not determine particle position")
            self.stage.moveabs(origin[0], origin[1], origin[2])
            return True
        else:
            self.stage.moveabs(float(popt[1]), float(popt[2]))
            #print "Position of Particle: {0:+2.2f}, {1:+2.2f}".format(popt[1],popt[2])

        #------------ Plot scanned map and fitted 2dgauss to file
        # modified from: http://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m#comment33999040_21566831
        plt.figure()
        plt.imshow(int.reshape(self.settings.rasterdim, self.settings.rasterdim))
        #plt.colorbar()
        if popt is not None:
            data_fitted = self.gauss2D((x, y), *popt)
        else:
            data_fitted = self.gauss2D((x, y), *initial_guess)
            print(initial_guess)

        fig, ax = plt.subplots(1, 1)
        ax.hold(True)
        ax.imshow(int.reshape(self.settings.rasterdim, self.settings.rasterdim), cmap=plt.cm.jet, origin='bottom',
                  extent=(x.min(), x.max(), y.min(), y.max()), interpolation='nearest')
        ax.contour(x, y, data_fitted.reshape(self.settings.rasterdim, self.settings.rasterdim), 8, colors='w')
        plt.savefig("map_particle_search.png")
        #------------ END Plot scanned map and fitted 2dgauss to file
        plt.close()

        measured = np.zeros(7)
        self.stage.moverel(dx=-0.4)
        for x in range(7):
            self.stage.moverel(dx=0.1)
            spec = self.smooth(self._spectrometer.intensities())
            measured[x] = np.max(spec)
        maxind = np.argmax(measured)
        self.stage.moverel(dx=-maxind * 0.1)

        update_connection(0.4)

        measured = np.zeros(7)
        self.stage.moverel(dy=-0.4)
        for y in range(7):
            self.stage.moverel(dy=0.1)
            measured[y] = np.max(self.smooth(self._spectrometer.intensities()))
        maxind = np.argmax(measured)
        self.stage.moverel(dy=-maxind * 0.1)

        update_connection(0.7)

        measured = np.zeros(7)
        self.stage.moverel(dx=-0.4)
        for x in range(7):
            self.stage.moverel(dx=0.1)
            measured[x] = np.max(self.smooth(self._spectrometer.intensities()))
        maxind = np.argmax(measured)
        self.stage.moverel(dx=-maxind * 0.1)

        update_connection(1.0)

        connection.send([True,0.0,None])
        return True

    def save_data(self,prefix):
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