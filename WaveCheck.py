from pydub import AudioSegment
from scipy.signal import decimate, hamming, spectrogram, lfilter, butter
from scipy.fftpack import fft
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np
from skimage import util

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

rate, audiodata = wavfile.read('new.wav')
print(rate)

audiodata = audiodata.sum(axis=1)/2

window_size = 4096
hamming_window = hamming(window_size, sym=False)

fft_data_orig = fft(audiodata[:4096]*hamming_window)
fft_freq = np.fft.fftfreq(4096)
power = np.abs(fft_data_orig[:2048])
print(fft_freq.min(), fft_freq.max())

tempdata = butter_lowpass_filter(audiodata, 500, 44100, order=8)
audiodata = butter_lowpass_filter(audiodata, 500, 44100)

audiodata = decimate(audiodata, 4)
tempdata = decimate(tempdata, 4)
window_size = 1024
hamming_window = hamming(window_size, sym=False)

fft_data = fft(audiodata[:window_size]*hamming_window)
fft_freq = np.fft.fftfreq(window_size//2)
power = np.abs(fft_data[:window_size//2])
print(fft_freq.min(), fft_freq.max())
plt.plot(fft_freq[:window_size//2]*11025, power)
# plt.show()

fft_data = fft(tempdata[:window_size]*hamming_window)
fft_freq = np.fft.fftfreq(window_size//2)
power = np.abs(fft_data[:window_size//2])
print(fft_freq.min(), fft_freq.max())
plt.plot(fft_freq[:window_size//2]*11025, power)

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
# freqs, times, Sx = spectrogram(audiodata, fs=11025, window='hamming', nperseg=1024, detrend=False, scaling='spectrum')
# f, ax = plt.subplots(figsize=(4.8, 2.4))
# ax.pcolormesh(times, freqs / 1000, 10 * np.log10(Sx), cmap='viridis')
# ax.set_ylabel('Frequency [kHz]')
# ax.set_xlabel('Time [s]');

plt.show()