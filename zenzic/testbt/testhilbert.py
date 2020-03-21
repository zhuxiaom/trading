import numpy as np
import pandas as pd
from talib.abstract import *
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp

def hl_period(signal, loopback, fs):
    sz = len(signal)
    res = [0 for i in range(0, sz)]
    for i in range(loopback - 1, sz):
        s = i - loopback + 1
        e = i + 1
        analytic_signal = hilbert(signal[s:e])
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)
        res[i] = instantaneous_frequency[-1]
    return res

duration = 400.0
fs = 400.0
samples = int(fs*duration)
t = np.arange(samples) / fs
signal = chirp(t, 20.0, t[-1], 100.0)
signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )
analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = hl_period(signal, 62, fs)
input = {'close': signal}
period = HT_DCPHASE(input)
fig = plt.figure()
# ax0 = fig.add_subplot(211)
# ax0.plot(t, signal, label='signal')
# ax0.plot(t, amplitude_envelope, label='envelope')
# ax0.set_xlabel("time in seconds")
# ax0.legend()

# ax0 = fig.add_subplot(211)
# ax0.plot(t[0:], period)
# ax1 = fig.add_subplot(212)
# ax1.plot(t[1:], instantaneous_frequency)
# ax1.set_xlabel("time in seconds")
# ax1.set_ylim(0.0, 120.0)

print(instantaneous_frequency)
print(period)
print("Successful!")