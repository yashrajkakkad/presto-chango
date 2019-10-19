from pydub import AudioSegment
from scipy.signal import decimate, hamming, spectrogram, lfilter, butter, filtfilt, resample
from scipy.fftpack import fft
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage import util

def stereo_to_mono(audiodata):
    return audiodata.sum(axis=1) / 2

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y
    
def downsample_signal(data, factor):
    downsampled_data = decimate(data, factor)
    return downsampled_data

def generate_window(window_size):
    return hamming(window_size, sym=False)

def do_fft(data, window_size, window, sampling_rate):
    fft_data = fft(data[:window_size]*window)
    fft_freq = np.fft.fftfreq(window_size//2)
    power = np.abs(fft_data[:window_size//2])
    plt.plot(np.abs(fft_freq*sampling_rate), power)
    plt.show()

rate, audiodata = wavfile.read('modem.wav')
print(rate)

audiodata = stereo_to_mono(audiodata)

audiodata = butter_lowpass_filter(audiodata, 5000, 44100)

audiodata = downsample_signal(audiodata, 4)

window_size = 1024
hamming_window = generate_window(window_size)

# sampling_rate = 11025
# do_fft(audiodata, window_size, hamming_window, sampling_rate)

# slices = util.view_as_windows(audiodata, window_shape=(window_size,), step=100)
# slices = slices * hamming_window
# slices = slices.T

# spectrum = np.fft.fft(slices, axis=0)[:window_size // 2 + 1:-1]
# spectrum = np.abs(spectrum)

# f, ax = plt.subplots(figsize=(4.8, 2.4))

# S = np.abs(spectrum)
# S = 20 * np.log10(S / np.max(S))

# L = audiodata.shape[0]/rate

# ax.imshow(S, origin='lower', cmap='viridis',
#           extent=(0, L, 0, rate / 2 / 1000))
# ax.axis('tight')
# ax.set_ylabel('Frequency [kHz]')
# ax.set_xlabel('Time [s]');

# DO IT DIRECTLY
freqs, times, Sx = spectrogram(audiodata, fs=11025, window='hamming', nperseg=1024, detrend=False, scaling='spectrum')
f, ax = plt.subplots(figsize=(4.8, 2.4))
ax.pcolormesh(times, freqs / 1000, 10 * np.log10(Sx), cmap='viridis')
ax.set_ylabel('Frequency [kHz]')
ax.set_xlabel('Time [s]');

plt.show()