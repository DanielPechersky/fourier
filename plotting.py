import matplotlib.pyplot as plt

import config
from feature import *


class InteractivePlot:
    def __init__(self, fm):
        plt.ion()
        self.done = False
        fm.fig.canvas.mpl_connect('close_event', lambda *_: self.finish())
        self.fm = fm
        self.main_loop()

    def main_loop(self):
        while not self.done:
            self.fm.update()
            plt.draw()
            plt.pause(.001)
        plt.close()

    def finish(self):
        self.done = True


plt.style.use('seaborn')

freq = np.linspace(**config.freq_limit, num=config.point_count)
time = np.linspace(**config.time_limit, num=config.point_count)
amp = config.amp

with FeatureManager(freq, time, amp, fig=plt.figure(figsize=config.figsize, dpi=80), grid=config.grid) as fm:
    for i, feature in enumerate(config.features, 1):
        args, kwargs = config.params[feature]
        fm.add_feature(feature, i, *args, **kwargs)

    fm.plot()

    interactive = True  # interactive = False window does not close

    if interactive:
        p = InteractivePlot(fm)
    else:
        plt.show()


