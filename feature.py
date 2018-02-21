import abc
import math
from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

_ = Axes3D


def wrap_around_circle(freq, time):
    return np.exp(-2*np.pi*1j*time*freq)


class Feature(abc.ABC):
    def __init__(self, freq, time, amp, fig, *args, **kwargs):
        self.freq = freq
        self.time = time
        self.amp = amp
        self.fig = fig

        self.plot_args = args
        self.plot_kwargs = kwargs

    @abc.abstractmethod
    def plot(self):
        pass


class FourierSlice(Feature):
    def __init__(self, *args, steps=100, axes_limit=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.steps = steps
        self.ax = None
        self.axes_limit = axes_limit
        self.freq_set_to = 0
        self.scroll_amount = 0

    def get_points(self):
        return self.amp * wrap_around_circle(self.freq[self.freq_set_to], self.time)

    def plot(self):

        pts = self.get_points()

        if not self.ax:
            self.ax = self.fig.add_subplot(*self.plot_args, **self.plot_kwargs)
        self.ax.clear()

        self.ax.set_xlabel("Real")
        self.ax.set_ylabel("Imaginary")

        if self.axes_limit:
            self.ax.set_xlim(-self.axes_limit, self.axes_limit)
            self.ax.set_ylim(-self.axes_limit, self.axes_limit)
        self.ax.set_aspect('equal')
        self.ax.set_title("Wrapped function\nFrequency = {:.2f}".format(self.freq[self.freq_set_to]))
        self.ax.plot(pts.real, pts.imag)
        self.ax.plot(0, 0, 'r.')
        pts_avg = pts.sum() / len(self.time)
        pts_avg_artist, = self.ax.plot(pts_avg.real, pts_avg.imag, 'go', label="Average of all points")
        self.ax.legend()

        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        return self.ax

    def on_scroll(self, event):
        self.scroll_amount += event.step

    def update(self):
        self.freq_set_to = int(self.freq_set_to +
                               math.copysign(self.scroll_amount**2 * self.freq.size / self.steps, -self.scroll_amount)) \
                           % self.freq.size
        self.scroll_amount = 0
        self.plot()


class FourierFeature(Feature):
    @abc.abstractmethod
    def plot(self):
        raise NotImplemented()

    def get_points(self):
        partial_wrap = partial(wrap_around_circle, time=self.time)
        return np.vectorize(lambda freq_: (self.amp * partial_wrap(freq=freq_)).sum() / len(self.time))(self.freq)


class AveragePoint(FourierFeature):
    def plot(self):
        pts = self.get_points()

        ax = self.fig.add_subplot(projection='3d', *self.plot_args, **self.plot_kwargs)

        ax.set_title("Change in average over frequency")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Real")
        ax.set_zlabel("Imaginary")

        ax.plot(self.freq, pts.real, pts.imag, 'g', linewidth=1)
        ax.mouse_init()
        return ax


class Frequency(FourierFeature):
    def plot(self):
        pts = self.get_points()

        ax = self.fig.add_subplot(*self.plot_args, **self.plot_kwargs)

        ax.set_title("Discrete Fourier transform")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Value of average")

        ax.plot(self.freq, pts.real, 'r', label="Real")
        ax.plot(self.freq, pts.imag, 'b', label="Imaginary")

        ax.legend()


class FrequencySummed(FourierFeature):
    def plot(self):
        pts = self.get_points()

        ax = self.fig.add_subplot(*self.plot_args, **self.plot_kwargs)

        ax.set_title("Distance of the average from center")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Distance of average")

        ax.plot(self.freq, np.hypot(pts.real, pts.imag), c=(.75, 0, .75))


class OriginalFunction(Feature):
    def plot(self):
        ax = self.fig.add_subplot(*self.plot_args, **self.plot_kwargs)

        ax.set_title("Original function")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")

        ax.hlines(y=0, xmin=self.time.min(), xmax=self.time.max(), color='r')
        ax.plot(self.time, self.amp)


class Description(Feature):
    text = """This program plots different features of the fourier transform,
     in an attempt to make the transform more intuitive.
Change the function displayed in the config, as well as other settings.
    
Some of the features are interactive!
Scroll in order to change the frequency of the slice.
Drag the 3D plot to rotate it."""

    def plot(self):
        ax: plt.Axes = self.fig.add_subplot(*self.plot_args, **self.plot_kwargs)
        ax.axis('off')
        ax.text(.5, .5, self.text,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                wrap=True)


class FeatureManager:
    def __init__(self, freq, time, amp, fig, grid):
        self.freq = freq
        self.time = time
        self.amp = amp
        self.fig = fig
        self.grid = grid

        self.features = []

    def add_feature(self, feature: type, *args, **kwargs):
        self.features.append(feature(self.freq, self.time, self.amp, self.fig, *self.grid, *args, **kwargs))
        return self.features[-1]

    def plot(self):
        return [feature.plot() for feature in self.features]

    def update(self):
        for feature in self.features:
            try:
                feature.update()
            except AttributeError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.close(self.fig)
        del self.features
        del self.fig
