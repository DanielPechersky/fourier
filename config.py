from collections import defaultdict

from feature import *

params = defaultdict(lambda: ([], {}))
features = [
    OriginalFunction,
    FourierSlice,
    AveragePoint,
    Frequency,
    FrequencySummed,
    Description
]
params[FourierSlice] = [], {'axes_limit': 1.5}

grid = (2, 3)

figsize = (15, 10)


freq_limit = {'start': 0, 'stop': 10}
time_limit = {'start': 0, 'stop': 5}
point_count = 1000

time = np.linspace(**time_limit, num=point_count)


# add functions here
amps = {
    'log': np.log2(time+.001),
    'tan': np.tan(1*2*np.pi*time),
    'multi': np.cos(3*2*np.pi*time)*2 + np.sin(2*2*np.pi*time) + np.cos(7*2*np.pi*time)/2 + np.cos(1/2*2*np.pi*time)*3,
    'cos': np.cos(2*2*np.pi*time),
    'sin': np.sin(2*2*np.pi*time),
    'trianglewave': time % 1,
    'squarewave': np.sign(np.sin(2*np.pi*time)),
    'phaseshift': np.cos(2*np.pi*(time+1/6)),
    'quadratic': 2*time**2 + 2,
    'linear': time,
    'exp': 2**time
}

# change function here
amp = amps['multi']
